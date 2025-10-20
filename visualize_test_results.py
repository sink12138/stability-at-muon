import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_test_data(csv_path):
    """
    Load test.csv data and parse it into a structured format
    Each epoch has 4 entries in sequence: clean train, adversarial train, clean test, adversarial test
    """
    data = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                epoch, loss, error = int(parts[0]), float(parts[1]), float(parts[2])
                data.append([epoch, loss, error])
    
    df = pd.DataFrame(data, columns=['epoch', 'loss', 'error'])
    
    # Determine which type of test each row represents
    df['test_type'] = df.index % 4
    df['test_name'] = df['test_type'].map({
        0: 'clean_train',
        1: 'adv_train', 
        2: 'clean_test',
        3: 'adv_test'
    })
    
    return df

def plot_test_errors(csv_path):
    """Plot the four different test errors over epochs"""
    df = load_test_data(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # Plot each test type separately
    for test_type in df['test_name'].unique():
        subset = df[df['test_name'] == test_type]
        epochs = subset['epoch']
        errors = subset['error']
        
        linestyle = '-' if 'clean' in test_type else '--'
        color = 'blue' if 'train' in test_type else 'red'
        
        plt.plot(epochs, errors, label=test_type, linestyle=linestyle, color=color, marker='o', markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate (%)')
    plt.title('Test Error Rates Over Training Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_accuracies(csv_path):
    """Plot accuracies instead of error rates"""
    df = load_test_data(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # Convert error rates to accuracies
    df['accuracy'] = 100 - df['error']
    
    # Plot each test type separately
    for test_type in df['test_name'].unique():
        subset = df[df['test_name'] == test_type]
        epochs = subset['epoch']
        accuracies = subset['accuracy']
        
        linestyle = '-' if 'clean' in test_type else '--'
        color = 'blue' if 'train' in test_type else 'red'
        
        plt.plot(epochs, accuracies, label=f'{test_type}_accuracy', linestyle=linestyle, color=color, marker='o', markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Over Training Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_comparison(csv_path):
    """Plot clean vs adversarial performance"""
    df = load_test_data(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # Separate clean and adversarial results
    clean_train = df[df['test_name'] == 'clean_train']
    adv_train = df[df['test_name'] == 'adv_train']
    clean_test = df[df['test_name'] == 'clean_test']
    adv_test = df[df['test_name'] == 'adv_test']
    
    # Clean vs adversarial train
    plt.subplot(2, 2, 1)
    plt.plot(clean_train['epoch'], clean_train['error'], label='Clean Train', marker='o', markersize=4)
    plt.plot(adv_train['epoch'], adv_train['error'], label='Adversarial Train', linestyle='--', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate (%)')
    plt.title('Train: Clean vs Adversarial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Clean vs adversarial test
    plt.subplot(2, 2, 2)
    plt.plot(clean_test['epoch'], clean_test['error'], label='Clean Test', marker='o', markersize=4)
    plt.plot(adv_test['epoch'], adv_test['error'], label='Adversarial Test', linestyle='--', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate (%)')
    plt.title('Test: Clean vs Adversarial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy comparison train
    plt.subplot(2, 2, 3)
    plt.plot(clean_train['epoch'], 100 - clean_train['error'], label='Clean Train Acc', marker='o', markersize=4)
    plt.plot(adv_train['epoch'], 100 - adv_train['error'], label='Adversarial Train Acc', linestyle='--', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train Accuracy: Clean vs Adversarial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy comparison test
    plt.subplot(2, 2, 4)
    plt.plot(clean_test['epoch'], 100 - clean_test['error'], label='Clean Test Acc', marker='o', markersize=4)
    plt.plot(adv_test['epoch'], 100 - adv_test['error'], label='Adversarial Test Acc', linestyle='--', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy: Clean vs Adversarial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def main():
    # Assuming the test.csv is in the same directory or in the model_pth subdirectory
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize test results from adversarial training')
    parser.add_argument('--csv_path', type=str, default='model_pth/free_muon_aux_L2muon/test.csv', 
                        help='Path to test.csv file')
    parser.add_argument('--output_dir', type=str, default='results_viz', 
                        help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: {args.csv_path} not found!")
        return
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create visualizations
    print("Creating error rate plot...")
    plt1 = plot_test_errors(args.csv_path)
    plt1.savefig(os.path.join(args.output_dir, 'test_errors.png'), dpi=300, bbox_inches='tight')
    plt1.show()
    
    print("Creating accuracy plot...")
    plt2 = plot_accuracies(args.csv_path)
    plt2.savefig(os.path.join(args.output_dir, 'accuracies.png'), dpi=300, bbox_inches='tight')
    plt2.show()
    
    print("Creating comparison plot...")
    plt3 = plot_comparison(args.csv_path)
    plt3.savefig(os.path.join(args.output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    plt3.show()
    
    print(f"Visualizations saved to {args.output_dir}/")

if __name__ == "__main__":
    main()