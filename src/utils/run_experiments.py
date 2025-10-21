import subprocess
import itertools
import os

def main():
    methods = ['vanilla']
    optimizers = ['sgd']
    attacks = ['L2']
    
    args = [
        '--nEpochs', '20',
        '--data', 'cifar10',
        '--lr', '0.1', # sgd muon
    ]
    
    for method, optimizer, attack in itertools.product(methods, optimizers, attacks):
        print(f"Running experiment: method={method}, optimizer={optimizer}, attack={attack}")
        
        cmd_args = [
            'python', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training/train.py'),
            '--method', method,
            '--optimizer', optimizer,
            '--attack', attack,
            '--save_path', f'{method}_{optimizer}_{attack}'
        ] + args
        
        print(f"Executing: {' '.join(cmd_args)}")
        try:
            subprocess.run(cmd_args)
            subprocess.run([f'python', 'visualize_training.py', '--path', f'{method}_{optimizer}_{attack}'])
        except subprocess.CalledProcessError as e:
            print(f"Error running {method}_{optimizer}_{attack}: {e}")
            print(e.stderr)

if __name__ == '__main__':
    main()