import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, header=None, names=['epoch', 'loss', 'error'])
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def smooth_curve(data, window_size=10):
    if len(data) < window_size:
        return data
    smoothed = pd.Series(data).rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values

def plot_data_and_save(path, save_dir):
    data = load_data(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    base_name = os.path.splitext(os.path.basename(path))[0]
    folder_name = os.path.basename(save_dir)
    
    # loss
    ax1.plot(data['epoch'], data['loss'], 'b-', linewidth=1.0, alpha=0.3, label='Original Loss')
    smoothed_loss = smooth_curve(data['loss'].values)
    ax1.plot(data['epoch'], smoothed_loss, 'b-', linewidth=2, label='Smoothed Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{folder_name} - {base_name} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # error
    ax2.plot(data['epoch'], data['error'], 'r-', linewidth=1.0, alpha=0.3, label='Original Error')
    smoothed_error = smooth_curve(data['error'].values)
    ax2.plot(data['epoch'], smoothed_error, 'r-', linewidth=2, label='Smoothed Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.set_title(f'{folder_name} - {base_name} - Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{base_name}_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

def plot_compare(paths):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, path in enumerate(paths):
        data = load_data(path)
        if data is not None:
            color = colors[i % len(colors)]
            # label = path.split(os.sep)[1]
            label = path.split('\\')[1].split('.')[0].upper()
            
            # ax1.plot(data['epoch'], data['loss'], color=color, linewidth=1.0, alpha=0.3)
            smoothed_loss = smooth_curve(data['loss'].values)
            ax1.plot(data['epoch'], smoothed_loss, color=color, linewidth=1, label=f'{label} Loss')
            
            # ax2.plot(data['epoch'], data['error'], color=color, linewidth=1.0, alpha=0.3)
            smoothed_error = smooth_curve(data['error'].values)
            ax2.plot(data['epoch'], smoothed_error, color=color, linewidth=1, label=f'{label} Error')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.set_title('Error Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', help='Paths to the CSV files or directories containing the CSV files')
    parser.add_argument('--compare', action='store_true', help='Compare multiple methods')
    args = parser.parse_args()
    
    if args.compare:
        train_csv_files = []
        test_csv_files = []
        for path in args.paths:
            train_file = os.path.join('vis', f'{path}.csv')
            # test_file = os.path.join('model_pth', path, 'test.csv')
            train_csv_files.append(train_file)
            # test_csv_files.append(test_file)
        
        plot_compare(train_csv_files)
        plt.savefig('1.png')
        plt.close()

        # plot_compare(test_csv_files)
        # plt.savefig('test_comparison.png')
        # plt.close()
    else:
        path = args.paths[0] if args.paths else '.'
        save_dir = os.path.join('model_pth', path)
        train_path = os.path.join('model_pth', path, 'train.csv')
        test_path = os.path.join('model_pth', path, 'test.csv')
        print(train_path, path)

        plot_data_and_save(train_path, save_dir)
        plot_data_and_save(test_path, save_dir)

if __name__ == "__main__":
    main()