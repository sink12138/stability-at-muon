import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

def load_training_data(file_path):
    try:
        data = pd.read_csv(file_path, header=None, names=['epoch', 'loss', 'error'])
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_training_curves(data, title="Training Curves", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(data['epoch'], data['loss'], 'b-', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(data['epoch'], data['error'], 'r-', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (%)')
    ax2.set_title(f'{title} - Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_combined_curves(data_list, labels, title="Training Comparison", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (data, label) in enumerate(zip(data_list, labels)):
        color = colors[i % len(colors)]
        ax1.plot(data['epoch'], data['loss'], color=color, linewidth=1.5, label=label)
        ax2.plot(data['epoch'], data['error'], color=color, linewidth=1.5, label=label)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (%)')
    ax2.set_title(f'{title} - Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def smooth_curve(data, window_size=5):
    if len(data) < window_size:
        return data
    # 使用pandas的rolling方法进行平滑处理，避免长度变化问题
    smoothed = pd.Series(data).rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values

def plot_smoothed_curves(data, window_size=5, title="Smoothed Training Curves", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    smoothed_loss = smooth_curve(data['loss'].values, window_size)
    ax1.plot(data['epoch'], data['loss'], 'b-', linewidth=1.0, alpha=0.3, label='Original')
    ax1.plot(data['epoch'], smoothed_loss, 'b-', linewidth=2, label=f'Smoothed (window={window_size})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    smoothed_error = smooth_curve(data['error'].values, window_size)
    ax2.plot(data['epoch'], data['error'], 'r-', linewidth=1.0, alpha=0.3, label='Original')
    ax2.plot(data['epoch'], smoothed_error, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (%)')
    ax2.set_title(f'{title} - Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_combined_smoothed_curves(data_list, labels, window_size=10, title="Smoothed Training Comparison", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (data, label) in enumerate(zip(data_list, labels)):
        color = colors[i % len(colors)]
        
        # 对比图中只显示平滑后的曲线，保持与plot_smoothed_curves一致的处理方式
        smoothed_loss = smooth_curve(data['loss'].values, window_size)
        smoothed_error = smooth_curve(data['error'].values, window_size)
        
        ax1.plot(data['epoch'], smoothed_loss, color=color, linewidth=2, label=label)
        ax2.plot(data['epoch'], smoothed_error, color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (%)')
    ax2.set_title(f'{title} - Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def analyze_training_data(data, name):
    file = open(os.path.join('model_pth', name, 'result.txt'), 'w')
    file.write(f"Total epochs: {data['epoch'].max():.2f}\n")
    file.write(f"Initial loss: {data['loss'].iloc[0]:.4f}\n")
    file.write(f"Final loss: {data['loss'].iloc[-1]:.4f}\n")
    file.write(f"Initial error: {data['error'].iloc[0]:.2f}%\n")
    file.write(f"Final error: {data['error'].iloc[-1]:.2f}%\n")
    file.write(f"Best error: {data['error'].min():.2f}% at epoch {data.loc[data['error'].idxmin(), 'epoch']:.2f}\n")
    file.write(f"Best loss: {data['loss'].min():.4f} at epoch {data.loc[data['loss'].idxmin(), 'epoch']:.2f}")
    file.close()

def plot_method_comparison(method, optimizers=['sgd', 'adam','muon_aux'], smooth=False):
    data_list = []
    labels = []
    
    for optimizer in optimizers:
        folder_name = f"{method}_{optimizer}"
        file_path = os.path.join('model_pth', folder_name, 'train.csv')
        
        if os.path.exists(file_path):
            data = load_training_data(file_path)
            if data is not None:
                data_list.append(data)
                labels.append(optimizer.upper())
                print(f"Loaded data for {folder_name}")
        else:
            print(f"File not found: {file_path}")
    
    if data_list:
        if smooth:
            plot_combined_smoothed_curves(data_list, labels, window_size=10,
                                        title=f"Smoothed {method.upper()} Method Comparison (Training Error)",
                                        save_path=f"{method}_smoothed_comparison.png")
            print(f"Saved smoothed comparison plot for {method} method")
        else:
            plot_combined_curves(data_list, labels, 
                               title=f"{method.upper()} Method Comparison (Training Error)",
                               save_path=f"{method}_comparison.png")
            print(f"Saved comparison plot for {method} method")
    else:
        print(f"No data found for {method} method")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--compare', type=str, choices=['fast', 'free', 'vanilla'])
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the curves')
    args = parser.parse_args()
    path = args.path
    compare_method = args.compare
    smooth = args.smooth
    
    if compare_method:
        plot_method_comparison(compare_method, smooth=smooth)
        return
    
    train_data = load_training_data(os.path.join('model_pth', path, 'train.csv'))
    test_data = load_training_data(os.path.join('model_pth', path, 'test.csv')) 

    if train_data is not None:
        analyze_training_data(train_data, path)
        plot_training_curves(train_data, title=f"{path} train curves", 
                            save_path=os.path.join('model_pth', path, 'training_curves.png'))
        plot_smoothed_curves(train_data, window_size=10, title=f"smoothed {path} train curves",
                            save_path=os.path.join('model_pth', path, 'smoothed_training_curves.png'))
    else:
        print("Failed to load testing data.")

    if test_data is not None:
        analyze_training_data(test_data, path)
        plot_training_curves(test_data, title=f"{path} test curves", 
                            save_path=os.path.join('model_pth', path, 'testing_curves.png'))
        plot_smoothed_curves(test_data, window_size=10, title=f"smoothed {path} test curves",
                            save_path=os.path.join('model_pth', path, 'smoothed_testing_curves.png'))
    else:
        print("Failed to load testing data.")

if __name__ == "__main__":
    main()