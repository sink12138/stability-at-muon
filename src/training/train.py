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
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models import models
from src.attacks.pgd_attack import L2PGDAttack, LinfPGDAttack, MuonL2PGDAttack
from src.optimizers.muon import SingleDeviceMuonWithAuxAdam


def create_attacker(net, attack_type, eps, alpha, steps=10, random_start=True):
    if attack_type == 'linf':
        return LinfPGDAttack(net, eps=eps, alpha=alpha, steps=steps, random_start=random_start)
    elif attack_type == 'l2':
        return L2PGDAttack(net, eps=eps, alpha=alpha, steps=steps, random_start=random_start)
    elif attack_type == 'l2muon':
        return MuonL2PGDAttack(net, eps=eps, alpha=alpha, steps=steps, random_start=random_start)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def adjust_opt(args, optimizer, epoch):
    if epoch < args.nEpochs//2: lr = args.lr
    elif epoch < args.nEpochs*3//4: lr = args.lr * 0.1
    else: lr = args.lr * 0.01
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, attacker, start_time):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    epoch_start_time = time.time()

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
        
        # 计算训练时间
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if batch_idx % (len(trainLoader)//10) == 0: 
            print('Train Epoch: {:.2f} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tError: {:.6f}\tTime: {:.2f}s'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), err, elapsed_time))

        trainF.write('{},{},{},{}\n'.format(partialEpoch, loss.item(), err, elapsed_time))
        trainF.flush()
    
    epoch_time = time.time() - epoch_start_time
    print(f'Epoch {epoch} training completed in {epoch_time:.2f} seconds')


def train_free(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, start_time):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    epoch_start_time = time.time()

    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()

        # 初始化对抗扰动
        if args.attack == 'l2':
            delta = torch.empty_like(data).normal_()
            d_flat = delta.view(data.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(data.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*args.eps
        elif args.attack == 'l2muon':
            delta = torch.empty_like(data).normal_()
            d_flat = delta.view(data.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(data.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*args.eps
        elif args.attack == 'linf':
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

            if args.attack == 'l2':
                delta_grad_norm = torch.norm(delta_grad, p=2,dim=(1,2,3)).view(-1, 1,1,1).repeat(1, data.size(1), data.size(2), data.size(3)) + 1e-12
                delta.data = delta.data + args.free_lr * delta_grad / delta_grad_norm

                delta_norm = torch.norm(delta.data, p=2,dim=(1,2,3)).view(-1, 1,1,1).repeat(1, data.size(1), data.size(2), data.size(3)) + 1e-12
                delta.data = ~(delta_norm > args.eps) * delta.data + args.eps * delta.data * (delta_norm > args.eps) / delta_norm
            elif args.attack == 'l2muon':
                delta_grad_norm = torch.norm(delta_grad, p=2,dim=(1,2,3)).view(-1, 1,1,1).repeat(1, data.size(1), data.size(2), data.size(3)) + 1e-12
                delta.data = delta.data + args.free_lr * delta_grad / delta_grad_norm

                delta_norm = torch.norm(delta.data, p=2,dim=(1,2,3)).view(-1, 1,1,1).repeat(1, data.size(1), data.size(2), data.size(3)) + 1e-12
                delta.data = ~(delta_norm > args.eps) * delta.data + args.eps * delta.data * (delta_norm > args.eps) / delta_norm
            elif args.attack == 'linf': 
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

        # 计算训练时间
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        trainF.write('{},{},{},{}\n'.format(partialEpoch, loss.item(), err, elapsed_time))
        trainF.flush()


def test(args, epoch, net, testLoader, testF, lossFunc, attacker, start_time):
    net.eval()
    test_loss = 0
    nProcessed = 0
    incorrect = 0
    test_start_time = time.time()
    
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
    
    # 计算测试时间
    current_time = time.time()
    elapsed_time = current_time - start_time
    test_time = current_time - test_start_time
    
    print('Average loss: {:.4f}, Error: {}/{} ({:.1f}%), Test time: {:.2f}s, Total time: {:.2f}s\n'.format(
        test_loss, incorrect, nProcessed, err, test_time, elapsed_time))

    testF.write('{},{},{},{}\n'.format(epoch, test_loss, err, elapsed_time))
    testF.flush()

    return err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=200)
    parser.add_argument('--save_path', type=str)

    parser.add_argument('--data', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'fashionmnist'))
    parser.add_argument('--method', type=str, default='vanilla', choices=('vanilla', 'fast', 'free'))
    parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adam', 'muon'))
    parser.add_argument('--attack', type=str, default='l2', choices=('linf', 'l2', 'l2muon'))
    parser.add_argument('--model', type=str, default='res18', choices=('res18', 'wrn34'))

    parser.add_argument('--eps', type=float, default=128.0)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--fast_lr', type=float, default=64.0)
    parser.add_argument('--free_lr', type=float, default=128.0)
    parser.add_argument('--free_step', type=int, default=4)
    parser.add_argument('--adam_lr', type=float, default=1e-3)
    parser.add_argument('--adam_betas', nargs=2, type=float, default=[0.9, 0.95])
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--muon_lr', type=float, default=1e-1)
    parser.add_argument('--muon_momentum', type=float, default=0.95)
    parser.add_argument('--muon_beta', type=float, default=0.95)
    parser.add_argument('--ns_steps', type=int, default=5)

    args = parser.parse_args()

    args.eps = args.eps / 255.0
    args.fast_lr = args.fast_lr / 255.0
    args.free_lr = args.free_lr / 255.0

    args.save_path = os.path.join('model_pth', args.save_path)
    setproctitle.setproctitle(args.save_path)
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)

    if args.data == 'fashionmnist':
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
    elif args.data == 'fashionmnist':
        trainDataset = dset.FashionMNIST(root='fashionmnist', train=True, download=True, transform=trainTransform)
        testDataset = dset.FashionMNIST(root='fashionmnist', train=False, download=True, transform=testTransform)

    trainLoader = DataLoader(trainDataset, batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(testDataset, batch_size=args.batchSz, shuffle=False, **kwargs)
   
    lossFunc = nn.CrossEntropyLoss(reduction="mean") 

    if args.data == 'cifar10' or args.data == 'fashionmnist':
        num_classes = 10
    elif args.data == 'cifar100':
        num_classes = 100

    # Fashion-MNIST数据集，需要修改网络的第一层卷积以适应单通道输入
    if args.model == 'res18':
        net = models.ResNet18(num_classes=num_classes)
        if args.data == 'fashionmnist':
            net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    elif args.model == 'wrn34':
        net = models.WideResNet(34, num_classes=num_classes, widen_factor=10, dropRate=0.0)
        if args.data == 'fashionmnist':
            net.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)

    net = net.cuda()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.adam_lr, betas=args.adam_betas, eps=args.adam_eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'muon':
        # Use Muon for matrix parameters and Adam for scalar parameters
        hidden_matrix_params = [p for p in net.parameters() if p.ndim >= 2]
        scalar_params = [p for p in net.parameters() if p.ndim < 2]
        
        muon_group = dict(params=hidden_matrix_params, lr=args.muon_lr, momentum=args.muon_momentum, weight_decay=args.weight_decay, use_muon=True)
        adam_group = dict(params=scalar_params, lr=args.adam_lr, betas=args.adam_betas, eps=args.adam_eps, weight_decay=args.weight_decay, use_muon=False)
        param_groups = [muon_group, adam_group]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    trainF = open(os.path.join(args.save_path, 'train.csv'), 'w')
    testF = open(os.path.join(args.save_path, 'test.csv'), 'w')

    PGD_Attacker = create_attacker(net, args.attack, args.eps, args.eps/4, steps=10, random_start=True)
    
    # 记录训练开始时间
    start_time = time.time()
    print(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(1, args.nEpochs + 1):
        epoch_start_time = time.time()
        print(f"\n=== Epoch {epoch}/{args.nEpochs} 开始 ===")
        
        if args.method == 'vanilla':
            train(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, PGD_Attacker, start_time)
        elif args.method == 'fast':
            Fast_Attacker = create_attacker(net, args.attack, args.eps, args.fast_lr, steps=1, random_start=True)
            train(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, Fast_Attacker, start_time)
        elif args.method == 'free':
            train_free(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, start_time)

        adjust_opt(args, optimizer, epoch)
        torch.save(net, os.path.join(args.save_path, 'latest.pth'))

        print('\nClean Train error: ')
        test(args, epoch, net, trainLoader, testF, lossFunc, None, start_time)
        print('Train error against PGD attack: ')
        test(args, epoch, net, trainLoader, testF, lossFunc, PGD_Attacker, start_time)

        print('\nClean Test error: ')
        test(args, epoch, net, testLoader, testF, lossFunc, None, start_time)
        print('Test error against PGD attack: ')
        test(args, epoch, net, testLoader, testF, lossFunc, PGD_Attacker, start_time)
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        print(f"=== Epoch {epoch} 完成，耗时: {epoch_time:.2f}s，总耗时: {total_time:.2f}s ===")

    trainF.close()
    testF.close()


if __name__=='__main__':
    main()
