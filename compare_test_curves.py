import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

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

def compare_test_curves():
    """
    Compare test curves across different models and optimizers, removing accuracy plots
    """
    # Define the paths to compare - including more model types from different directories
    model_paths = {
        # Fast models
        'fast_sgd_L2': 'model_pth/attack/fast_sgd_L2/test.csv',
        'fast_muon_aux_L2muon': 'model_pth/attack/fast_muon_aux_L2muon/test.csv',
        # Free models
        'free_sgd_L2': 'model_pth/attack/free_sgd_L2/test.csv',
        'free_muon_aux_L2muon': 'model_pth/attack/free_muon_aux_L2muon/test.csv',
        # Vanilla models
        'vanilla_sgd_L2': 'model_pth/attack/vanilla_sgd_L2/test.csv',
    }
    
    # Check if paths exist and filter the dictionary
    existing_model_paths = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            existing_model_paths[name] = path
        else:
            print(f"Info: {path} does not exist, skipping...")
    
    if not existing_model_paths:
        print("No test.csv files found in the specified paths!")
        return
    
    # Load data for each model
    model_data = {}
    for name, path in existing_model_paths.items():
        try:
            df = load_test_data(path)
            model_data[name] = df
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if not model_data:
        print("No data could be loaded from the available test.csv files!")
        return
    
    # Create comparison plots - removing accuracy plots, keeping error plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'green', 'red', 'orange', 'skyblue']
    
    # Plot 1: Clean Test Error Comparison
    ax1 = axes[0, 0]
    for i, (name, df) in enumerate(model_data.items()):
        clean_test = df[df['test_name'] == 'clean_test']
        if not clean_test.empty:  # Check if data exists
            ax1.plot(clean_test['epoch'], clean_test['error'], 
                     label=name, color=colors[i % len(colors)], 
                     linestyle='-', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Clean Test Error (%)')
    ax1.set_title('Clean Test Error Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Adversarial Test Error Comparison
    ax2 = axes[0, 1]
    for i, (name, df) in enumerate(model_data.items()):
        adv_test = df[df['test_name'] == 'adv_test']
        if not adv_test.empty:  # Check if data exists
            ax2.plot(adv_test['epoch'], adv_test['error'], 
                     label=name, color=colors[i % len(colors)], 
                     linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Adversarial Test Error (%)')
    ax2.set_title('Adversarial Test Error Comparison')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Error Comparison
    ax3 = axes[1, 0]
    for i, (name, df) in enumerate(model_data.items()):
        clean_train = df[df['test_name'] == 'clean_train']
        if not clean_train.empty:  # Check if data exists
            ax3.plot(clean_train['epoch'], clean_train['error'], 
                     label=name, color=colors[i % len(colors)], 
                     linestyle='-', linewidth=1.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Clean Training Error (%)')
    ax3.set_title('Clean Training Error Comparison')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Adversarial Training Error Comparison
    ax4 = axes[1, 1]
    for i, (name, df) in enumerate(model_data.items()):
        adv_train = df[df['test_name'] == 'adv_train']
        if not adv_train.empty:  # Check if data exists
            ax4.plot(adv_train['epoch'], adv_train['error'], 
                     label=name, color=colors[i % len(colors)], 
                     linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Adversarial Training Error (%)')
    ax4.set_title('Adversarial Training Error Comparison')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('test_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create another plot showing the robustness (gap between clean and adversarial)
    plt.figure(figsize=(12, 8))
    
    for i, (name, df) in enumerate(model_data.items()):
        clean_test = df[df['test_name'] == 'clean_test']
        adv_test = df[df['test_name'] == 'adv_test']
        
        # Align the epochs (they should be the same but let's make sure)
        common_epochs = sorted(set(clean_test['epoch']).intersection(set(adv_test['epoch'])))
        if len(common_epochs) > 0:
            clean_errors = [clean_test[clean_test['epoch'] == e]['error'].iloc[0] for e in common_epochs]
            adv_errors = [adv_test[adv_test['epoch'] == e]['error'].iloc[0] for e in common_epochs]
            
            generalization_gap = [adv_err - clean_err for clean_err, adv_err in zip(clean_errors, adv_errors)]
            
            plt.plot(common_epochs, generalization_gap, label=f'{name}', 
                     color=colors[i % len(colors)], 
                     linestyle='-.', linewidth=1.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Robustness Gap (Adv - Clean Error) (%)')
    plt.title('Robustness Gap Comparison Across Models')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the robustness plot
    plt.savefig('robustness_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    compare_test_curves()
    print("Comparison plots saved as 'test_curves_comparison.png' and 'robustness_comparison.png'")

if __name__ == "__main__":
    main()