import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import shutil
import setproctitle
import numpy as np

import models, wideresnet
from PGD_Attack import L2PGDAttack, LinfPGDAttack, MuonL2PGDAttack
from Muon import SingleDeviceMuon, SingleDeviceMuonWithAuxAdam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=200)
    parser.add_argument('--data', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'mnist', 'svhn'))

    parser.add_argument('--method', type=str, default='vanilla', choices=('vanilla', 'fast', 'free'))
    parser.add_argument('--attack', type=str, default='L2', choices=('Linf', 'L2', 'L2muon'))
    parser.add_argument('--eps', type=float, default=128.0)
    parser.add_argument('--model', type=str, default='res18', choices=('res18', 'wrn34'))

    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--fast_lr', type=float, default=64.0)
    parser.add_argument('--free_lr', type=float, default=128.0)
    parser.add_argument('--free_step', type=int, default=4)

    parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adam', 'muon', 'muon_aux'))
    parser.add_argument('--muon_lr', type=float, default=1e-1)
    parser.add_argument('--muon_momentum', type=float, default=0.95)
    parser.add_argument('--adam_lr', type=float, default=1e-3)
    parser.add_argument('--adam_betas', nargs=2, type=float, default=[0.9, 0.95])
    parser.add_argument('--adam_eps', type=float, default=1e-8)

    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    args.eps = args.eps / 255.0
    args.fast_lr = args.fast_lr / 255.0
    args.free_lr = args.free_lr / 255.0

    args.save_path = os.path.join('model_pth', args.save_path)
    setproctitle.setproctitle(args.save_path)
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)

    # 根据数据集类型设置不同的数据转换
    if args.data == 'mnist':
        trainTransform = transforms.Compose([
            transforms.ToTensor(),
        ])
        testTransform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        trainTransform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        testTransform = transforms.Compose([
            transforms.ToTensor()
        ])

    kwargs = {'num_workers': 0, 'pin_memory': True} 

    if args.data == 'cifar10':
        trainDataset = dset.CIFAR10(root='cifar10', train=True, download=True, transform=trainTransform)
        testDataset = dset.CIFAR10(root='cifar10', train=False, download=True, transform=testTransform)
    elif args.data == 'cifar100':
        trainDataset = dset.CIFAR100(root='cifar100', train=True, download=True, transform=trainTransform)
        testDataset = dset.CIFAR100(root='cifar100', train=False, download=True, transform=testTransform)
    elif args.data == 'mnist':
        trainDataset = dset.MNIST(root='mnist', train=True, download=True, transform=trainTransform)
        testDataset = dset.MNIST(root='mnist', train=False, download=True, transform=testTransform)
    elif args.data == 'svhn':
        trainDataset = dset.SVHN(root='svhn', split='train', download=True, transform=trainTransform)
        testDataset = dset.SVHN(root='svhn', split='test', download=True, transform=testTransform)

    trainLoader = DataLoader(trainDataset, batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(testDataset, batch_size=args.batchSz, shuffle=False, **kwargs)
   
    lossFunc = nn.CrossEntropyLoss(reduction="mean") 

    if args.data == 'cifar10' or args.data == 'svhn' or args.data == 'mnist':
        num_classes = 10
    elif args.data == 'cifar100':
        num_classes = 100

    if args.model == 'res18':
        # 对于MNIST数据集，需要修改网络的第一层卷积以适应单通道输入
        if args.data == 'mnist':
            net = models.ResNet18(num_classes=num_classes)
            # 修改第一层卷积层以适应MNIST的单通道输入
            net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            net = models.ResNet18(num_classes=num_classes)
    elif args.model == 'wrn34':
        net = models.WideResNet(34, num_classes=num_classes, widen_factor=10, dropRate=0.0)
        # 对于MNIST数据集，需要修改WideResNet的第一层卷积以适应单通道输入
        if args.data == 'mnist':
            net.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)

    net = net.cuda()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.adam_lr, betas=args.adam_betas, eps=args.adam_eps, weight_decay=args.weight_decay)
    # elif args.optimizer == 'muon':
    #     # Use only Muon optimizer for all parameters (recommended only for matrix parameters)
    #     optimizer = SingleDeviceMuon(net.parameters(), lr=args.muon_lr, momentum=args.muon_momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'muon_aux':
        # Use Muon for matrix parameters and Adam for scalar parameters
        hidden_matrix_params = [p for p in net.parameters() if p.ndim >= 2]
        scalar_params = [p for p in net.parameters() if p.ndim < 2]
        
        muon_group = dict(params=hidden_matrix_params, lr=args.muon_lr, momentum=args.muon_momentum, weight_decay=args.weight_decay, use_muon=True)
        adam_group = dict(params=scalar_params, lr=args.adam_lr, betas=args.adam_betas, eps=args.adam_eps, weight_decay=args.weight_decay, use_muon=False)
        param_groups = [muon_group, adam_group]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    trainF = open(os.path.join(args.save_path, 'train.csv'), 'w')
    testF = open(os.path.join(args.save_path, 'test.csv'), 'w')

    if args.attack == 'Linf':
        PGD_Attacker = LinfPGDAttack(net, eps=args.eps, alpha=args.eps/4, steps=10, random_start=True)
    elif args.attack == 'L2':
        PGD_Attacker = L2PGDAttack(net, eps=args.eps, alpha=args.eps/4, steps=10, random_start=True)
    elif args.attack == 'L2muon':
        PGD_Attacker = MuonL2PGDAttack(net, eps=args.eps, alpha=args.eps/4, steps=10, random_start=True)
    
    for epoch in range(1, args.nEpochs + 1):
        if args.method == 'vanilla':
            train(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, PGD_Attacker)
        elif args.method == 'fast':
            if args.attack == 'Linf':
                Fast_Attacker = LinfPGDAttack(net, eps=args.eps, alpha=args.fast_lr, steps=1, random_start=True)
            elif args.attack == 'L2':
                Fast_Attacker = L2PGDAttack(net, eps=args.eps, alpha=args.fast_lr, steps=1, random_start=True)
            elif args.attack == 'L2muon':
                Fast_Attacker = MuonL2PGDAttack(net, eps=args.eps, alpha=args.fast_lr, steps=1, random_start=True)
            train(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, Fast_Attacker)
        elif args.method == 'free':
            train_free(args, epoch, net, trainLoader, optimizer, trainF, lossFunc)

        adjust_opt(args, optimizer, epoch)
        torch.save(net, os.path.join(args.save_path, 'latest.pth'))

        print('\nClean Train error: ')
        test(args, epoch, net, trainLoader, testF, lossFunc, None)
        print('Train error against PGD attack: ')
        test(args, epoch, net, trainLoader, testF, lossFunc, PGD_Attacker)

        print('\nClean Test error: ')
        test(args, epoch, net, testLoader, testF, lossFunc, None)
        print('Test error against PGD attack: ')
        test(args, epoch, net, testLoader, testF, lossFunc, PGD_Attacker)


    trainF.close()
    testF.close()



def train(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, attacker):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()

        if attacker is not None:
            data = attacker(data, target)

        net.train()
        optimizer.zero_grad()
        output = net(data)
        loss = lossFunc(output, target)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] 
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        if batch_idx % (len(trainLoader)//10) == 0: 
            print('Train Epoch: {:.2f} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.item(), err))
        trainF.flush()



def test(args, epoch, net, testLoader, testF, lossFunc, attacker):
    net.eval()
    test_loss = 0
    nProcessed = 0
    incorrect = 0
    
    for data, target in testLoader:
        data, target = data.cuda(), target.cuda()

        if attacker is not None:
            data = attacker(data, target)

        with torch.no_grad():
            output = net(data)

        lossFunc_ = nn.CrossEntropyLoss(reduction="sum") 
        test_loss += lossFunc_(output, target).item()
        pred = output.data.max(1)[1] 
        incorrect += pred.ne(target.data).cpu().sum()
        nProcessed += len(data)

    test_loss /= nProcessed 
    err = 100.*incorrect/nProcessed
    print('Average loss: {:.4f}, Error: {}/{} ({:.1f}%)\n'.format(
        test_loss, incorrect, nProcessed, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()


    return err



def adjust_opt(args, optimizer, epoch):
    if epoch < args.nEpochs//2: lr = args.lr
    elif epoch < args.nEpochs*3//4: lr = args.lr * 0.1
    else: lr = args.lr * 0.01
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def train_free(args, epoch, net, trainLoader, optimizer, trainF, lossFunc):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()

        # 初始化对抗扰动
        if args.attack == 'L2':
            delta = torch.empty_like(data).normal_()
            d_flat = delta.view(data.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(data.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*args.eps
        elif args.attack == 'Linf':
            delta = torch.empty_like(data).uniform_(-args.eps, args.eps)
        
        delta = delta.cuda()
        delta.requires_grad = True

        net.train()

        for _ in range(args.free_step):
            output = net(torch.clamp(data + delta, min=0.0, max=1.0))
            loss = lossFunc(output, target)

            optimizer.zero_grad()
            loss.backward()

            delta_grad = delta.grad.detach()

            if args.attack == 'L2':
                delta_grad_norm = torch.norm(delta_grad, p=2,dim=(1,2,3)).view(-1, 1,1,1).repeat(1, data.size(1), data.size(2), data.size(3)) + 1e-12
                delta.data = delta.data + args.free_lr * delta_grad / delta_grad_norm

                delta_norm = torch.norm(delta.data, p=2,dim=(1,2,3)).view(-1, 1,1,1).repeat(1, data.size(1), data.size(2), data.size(3)) + 1e-12
                delta.data = ~(delta_norm > args.eps) * delta.data + args.eps * delta.data * (delta_norm > args.eps) / delta_norm
            elif args.attack == 'Linf': 
                delta.data = delta.data + args.free_lr * torch.sign(delta_grad) 
                delta.data = torch.clamp(delta.data, min=-args.eps, max=args.eps)

            delta.grad.zero_()
            optimizer.step()

        nProcessed += len(data)
        pred = output.data.max(1)[1] 
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        if batch_idx % (len(trainLoader)//10) == 0: 
            print('Train Epoch: {:.2f} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.item(), err))
        trainF.flush()




if __name__=='__main__':
    main()