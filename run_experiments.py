import subprocess
import itertools

def main():
    methods = ['vanilla', 'fast', 'free']
    optimizers = ['sgd', 'adam', 'muon_aux']
    
    args = [
        '--nEpochs', '30',
        '--lr', '0.1', # sgd muon
    ]
    
    for method, optimizer in itertools.product(methods, optimizers):
        print(f"Running experiment: method={method}, optimizer={optimizer}")
        
        cmd_args = [
            'python', 'train.py',
            '--method', method,
            '--optimizer', optimizer,
            '--save_path', f'{method}_{optimizer}'
        ] + args
        
        print(f"Executing: {' '.join(cmd_args)}")
        try:
            subprocess.run(cmd_args)
            subprocess.run([f'python', 'visualize_training.py', '--path', f'{method}_{optimizer}'])
        except subprocess.CalledProcessError as e:
            print(f"Error running {method}_{optimizer}: {e}")
            print(e.stderr)

if __name__ == '__main__':
    main()