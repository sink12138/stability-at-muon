import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from pathlib import Path

def load_test_data(csv_path):
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
                if len(parts) != 3:
                    print(f"Warning: Line {line_num} in {csv_path} has {len(parts)} columns, expected 3. Skipping...")
                    continue
                
                try:
                    epoch, loss, error = int(parts[0]), float(parts[1]), float(parts[2])
                    data.append([epoch, loss, error])
                except ValueError as e:
                    print(f"Warning: Line {line_num} in {csv_path} has invalid data: {e}. Skipping...")
                    continue
    except Exception as e:
        raise IOError(f"Error reading file {csv_path}: {e}")
    
    if not data:
        raise ValueError(f"No valid data found in {csv_path}")
    
    df = pd.DataFrame(data, columns=['epoch', 'loss', 'error'])
    
    # Determine which type of test each row represents
    df['test_type'] = df.index % 4
    df['test_name'] = df['test_type'].map({
        0: 'clean_train',
        1: 'adv_train', 
        2: 'clean_test',
        3: 'adv_test'
    })
    
    print(f"Loaded {len(df)} data points from {csv_path}")
    return df

def discover_experiments(experiments_dir="experiments"):
    if not os.path.exists(experiments_dir):
        print(f"Error: {experiments_dir} directory does not exist!")
        return {}
    
    experiment_paths = {}
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)
        test_csv_path = os.path.join(exp_path, "test.csv")
        
        if os.path.isdir(exp_path) and os.path.exists(test_csv_path):
            experiment_paths[exp_dir] = test_csv_path
            print(f"Found experiment: {exp_dir}")
    
    return experiment_paths


def categorize_experiments(experiment_names):
    categories = {
        'fast': [],
        'free': [],
        'vanilla': [],
        'other': []
    }
    
    for name in experiment_names:
        if name.startswith('fast_'):
            categories['fast'].append(name)
        elif name.startswith('free_'):
            categories['free'].append(name)
        elif name.startswith('vanilla_'):
            categories['vanilla'].append(name)
        else:
            categories['other'].append(name)
    
    return categories


def compare_test_curves(experiments_dir="experiments", output_dir="results/plots", show_plots=False):
    # Automatically discover all experiments
    model_paths = discover_experiments(experiments_dir)
    
    if not model_paths:
        print("No experiments found in the experiments directory!")
        return
    
    print(f"Found {len(model_paths)} experiments to compare")
    
    # Group by type and display
    categories = categorize_experiments(model_paths.keys())
    for category, experiments in categories.items():
        if experiments:
            print(f"  {category.capitalize()} experiments: {', '.join(experiments)}")
    
    # Load data for each model
    model_data = {}
    for name, path in model_paths.items():
        try:
            df = load_test_data(path)
            model_data[name] = df
            print(f"Successfully loaded data for {name}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if not model_data:
        print("No data could be loaded from the available test.csv files!")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Save the figure to results/plots directory
    output_path = os.path.join(output_dir, 'test_curves_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved test curves comparison to: {output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()
    
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
    
    # Save the robustness plot to results/plots directory
    robustness_output_path = os.path.join(output_dir, 'robustness_comparison.png')
    plt.savefig(robustness_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved robustness comparison to: {robustness_output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare test curves across different experiments')
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                        help='Directory containing experiment results (default: experiments)')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                        help='Directory to save output plots (default: results/plots)')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots interactively (default: False)')
    
    args = parser.parse_args()
    
    # Run comparison using command line arguments
    compare_test_curves(args.experiments_dir, args.output_dir, args.show_plots)
    print(f"Comparison plots saved to {args.output_dir}/ directory")

if __name__ == "__main__":
    main()