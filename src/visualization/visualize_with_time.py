import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def load_test_data_with_time(csv_path):
    """
    Load test.csv file with time data
    Format: epoch, loss, error, elapsed_time
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test CSV file not found: {csv_path}")
    
    data = []
    try:
        with open(csv_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split(',')
                if len(parts) == 4:  # New format: epoch, loss, error, time
                    epoch, loss, error, elapsed_time = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    data.append([epoch, loss, error, elapsed_time])
                elif len(parts) == 3:  # Old format: epoch, loss, error
                    epoch, loss, error = int(parts[0]), float(parts[1]), float(parts[2])
                    data.append([epoch, loss, error, 0.0])  # Set time to 0
                else:
                    print(f"Warning: Line {line_num} in {csv_path} has {len(parts)} columns, expected 3 or 4. Skipping...")
                    continue
    except Exception as e:
        raise IOError(f"Error reading file {csv_path}: {e}")
    
    if not data:
        raise ValueError(f"No valid data found in {csv_path}")
    
    df = pd.DataFrame(data, columns=['epoch', 'loss', 'error', 'elapsed_time'])
    
    # Determine test type for each row
    df['test_type'] = df.index % 4
    df['test_name'] = df['test_type'].map({
        0: 'clean_train',
        1: 'adv_train', 
        2: 'clean_test',
        3: 'adv_test'
    })
    
    print(f"Loaded {len(df)} data points from {csv_path}")
    return df

def plot_test_errors_by_time(csv_path):
    """Plot test errors by time"""
    df = load_test_data_with_time(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # Plot each test type
    for test_type in df['test_name'].unique():
        subset = df[df['test_name'] == test_type]
        times = subset['elapsed_time']
        errors = subset['error']
        
        linestyle = '-' if 'clean' in test_type else '--'
        color = 'blue' if 'train' in test_type else 'red'
        
        plt.plot(times, errors, label=test_type, linestyle=linestyle, color=color, marker='o', markersize=4)
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Error Rate (%)')
    plt.title('Test Error Rate Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_accuracies_by_time(csv_path):
    """Plot accuracies by time"""
    df = load_test_data_with_time(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # Convert to accuracy
    df['accuracy'] = 100 - df['error']
    
    # Plot each test type
    for test_type in df['test_name'].unique():
        subset = df[df['test_name'] == test_type]
        times = subset['elapsed_time']
        accuracies = subset['accuracy']
        
        linestyle = '-' if 'clean' in test_type else '--'
        color = 'blue' if 'train' in test_type else 'red'
        
        plt.plot(times, accuracies, label=f'{test_type}_accuracy', linestyle=linestyle, color=color, marker='o', markersize=4)
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_comparison_by_time(csv_path):
    """Plot clean vs adversarial performance comparison by time"""
    df = load_test_data_with_time(csv_path)
    
    plt.figure(figsize=(15, 10))
    
    # Separate clean and adversarial results
    clean_train = df[df['test_name'] == 'clean_train']
    adv_train = df[df['test_name'] == 'adv_train']
    clean_test = df[df['test_name'] == 'clean_test']
    adv_test = df[df['test_name'] == 'adv_test']
    
    # Clean vs adversarial training
    plt.subplot(2, 2, 1)
    plt.plot(clean_train['elapsed_time'], clean_train['error'], label='Clean Train', marker='o', markersize=4)
    plt.plot(adv_train['elapsed_time'], adv_train['error'], label='Adversarial Train', linestyle='--', marker='s', markersize=4)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Error Rate (%)')
    plt.title('Training: Clean vs Adversarial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Clean vs adversarial testing
    plt.subplot(2, 2, 2)
    plt.plot(clean_test['elapsed_time'], clean_test['error'], label='Clean Test', marker='o', markersize=4)
    plt.plot(adv_test['elapsed_time'], adv_test['error'], label='Adversarial Test', linestyle='--', marker='s', markersize=4)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Error Rate (%)')
    plt.title('Testing: Clean vs Adversarial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy comparison training
    plt.subplot(2, 2, 3)
    plt.plot(clean_train['elapsed_time'], 100 - clean_train['error'], label='Clean Train Acc', marker='o', markersize=4)
    plt.plot(adv_train['elapsed_time'], 100 - adv_train['error'], label='Adversarial Train Acc', linestyle='--', marker='s', markersize=4)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy: Clean vs Adversarial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy comparison testing
    plt.subplot(2, 2, 4)
    plt.plot(clean_test['elapsed_time'], 100 - clean_test['error'], label='Clean Test Acc', marker='o', markersize=4)
    plt.plot(adv_test['elapsed_time'], 100 - adv_test['error'], label='Adversarial Test Acc', linestyle='--', marker='s', markersize=4)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy (%)')
    plt.title('Testing Accuracy: Clean vs Adversarial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_epoch_vs_time(csv_path):
    """Plot epoch vs time relationship"""
    df = load_test_data_with_time(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # Only use clean_test data to avoid duplication
    clean_test = df[df['test_name'] == 'clean_test']
    
    plt.plot(clean_test['epoch'], clean_test['elapsed_time'], marker='o', markersize=6, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative Training Time (seconds)')
    plt.title('Epoch vs Training Time Relationship')
    plt.grid(True, alpha=0.3)
    
    # Add time annotations
    for i, (epoch, time_val) in enumerate(zip(clean_test['epoch'], clean_test['elapsed_time'])):
        if i % 5 == 0:  # Annotate every 5 epochs
            plt.annotate(f'{time_val:.0f}s', (epoch, time_val), textcoords="offset points", xytext=(0,10), ha='center')
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='Visualize test results with time data')
    parser.add_argument('--csv_path', type=str, default='model_pth/free_muon_aux_L2muon/test.csv', 
                        help='Path to test.csv file')
    parser.add_argument('--output_dir', type=str, default='results_viz_time', 
                        help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: {args.csv_path} does not exist!")
        return
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create visualizations
    print("Creating error rate over time plot...")
    plt1 = plot_test_errors_by_time(args.csv_path)
    plt1.savefig(os.path.join(args.output_dir, 'test_errors_by_time.png'), dpi=300, bbox_inches='tight')
    plt1.show()
    
    print("Creating accuracy over time plot...")
    plt2 = plot_accuracies_by_time(args.csv_path)
    plt2.savefig(os.path.join(args.output_dir, 'accuracies_by_time.png'), dpi=300, bbox_inches='tight')
    plt2.show()
    
    print("Creating time comparison plot...")
    plt3 = plot_comparison_by_time(args.csv_path)
    plt3.savefig(os.path.join(args.output_dir, 'comparison_by_time.png'), dpi=300, bbox_inches='tight')
    plt3.show()
    
    print("Creating epoch vs time relationship plot...")
    plt4 = plot_epoch_vs_time(args.csv_path)
    plt4.savefig(os.path.join(args.output_dir, 'epoch_vs_time.png'), dpi=300, bbox_inches='tight')
    plt4.show()
    
    print(f"Visualization images saved to {args.output_dir}/")

if __name__ == "__main__":
    main()