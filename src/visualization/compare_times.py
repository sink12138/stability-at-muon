import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path


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


def discover_experiments_with_time(experiments_dir="model_pth", specific_dirs=None):
    """
    Discover all experiment directories in the model_pth directory with test.csv files
    """
    if not os.path.exists(experiments_dir):
        print(f"Error: {experiments_dir} directory does not exist!")
        return {}
    
    experiment_paths = {}
    
    if specific_dirs:
        # Use specific directories if provided
        dirs = specific_dirs
    else:
        # Discover all available directories
        dirs = []
        for item in os.listdir(experiments_dir):
            item_path = os.path.join(experiments_dir, item)
            test_csv_path = os.path.join(item_path, "test.csv")
            if os.path.isdir(item_path) and os.path.exists(test_csv_path):
                dirs.append(item)
    
    print(f"Found directories to process: {dirs}")
    for exp_dir in dirs:
        exp_path = os.path.join(experiments_dir, exp_dir)
        test_csv_path = os.path.join(exp_path, "test.csv")
        
        if os.path.isdir(exp_path) and os.path.exists(test_csv_path):
            experiment_paths[exp_dir] = test_csv_path
            print(f"Found experiment: {exp_dir}")
    
    return experiment_paths


def categorize_experiments(experiment_names):
    """
    Categorize experiments by their method (fast, free, vanilla) and optimizer (sgd, adam, muon)
    """
    categories = {
        'fast': [],
        'free': [],
        'vanilla': [],
        'sgd': [],
        'adam': [],
        'muon': [],
        'l2': [],
        'l2muon': [],
        'other': []
    }
    
    for name in experiment_names:
        if name.startswith('fast_'):
            categories['fast'].append(name)
        elif name.startswith('free_'):
            categories['free'].append(name)
        elif name.startswith('vanilla_'):
            categories['vanilla'].append(name)
        elif '_sgd_' in name or name.endswith('_sgd'):
            categories['sgd'].append(name)
        elif '_adam_' in name or name.endswith('_adam'):
            categories['adam'].append(name)
        elif '_muon_' in name or name.endswith('_muon'):
            categories['muon'].append(name)
        elif '_l2_' in name or name.endswith('_l2'):
            categories['l2'].append(name)
        elif '_l2muon' in name or name.endswith('_l2muon'):
            categories['l2muon'].append(name)
        else:
            categories['other'].append(name)
    
    return categories


def group_experiments_for_comparison(experiment_names):
    """
    Group experiments into meaningful comparison sets
    """
    groups = {
        'training_method_comparison': {
            'name': 'Training Method Comparison',
            'description': 'Compare Fast, Free, and Vanilla training methods with same optimizer and attack',
            'sets': [
                {
                    'name': 'SGD + L2 Attack',
                    'experiments': ['fast_sgd_l2', 'free_sgd_l2', 'vanilla_sgd_l2'],
                    'color_styles': ['blue', 'red', 'green']
                },
                {
                    'name': 'SGD + L2MUON Attack',
                    'experiments': ['fast_sgd_l2muon', 'free_sgd_l2muon', 'vanilla_sgd_l2muon'],
                    'color_styles': ['blue', 'red', 'green']
                },
                {
                    'name': 'Muon + L2 Attack',
                    'experiments': ['fast_muon_l2', 'free_muon_l2', 'vanilla_muon_l2'],
                    'color_styles': ['blue', 'red', 'green']
                },
                {
                    'name': 'Muon + L2MUON Attack',
                    'experiments': ['fast_muon_l2muon', 'free_muon_l2muon', 'vanilla_muon_l2muon'],
                    'color_styles': ['blue', 'red', 'green']
                }
            ]
        },
        'optimizer_attack_comparison': {
            'name': 'Optimizer Comparison',
            'description': 'Compare SGD and Muon optimizers with same training method and attack',
            'sets': [
                {
                    'name': 'Vanilla',
                    'experiments': ['vanilla_sgd_l2', 'vanilla_muon_l2', 'vanilla_sgd_l2muon', 'vanilla_muon_l2muon'],
                    'color_styles': ['blue', 'red', 'green', 'orange']
                },
                {
                    'name': 'Fast',
                    'experiments': ['fast_sgd_l2', 'fast_muon_l2', 'fast_sgd_l2muon', 'fast_muon_l2muon'],
                    'color_styles': ['blue', 'red', 'green', 'orange']
                },
                {
                    'name': 'Free',
                    'experiments': ['free_sgd_l2', 'free_muon_l2', 'free_sgd_l2muon', 'free_muon_l2muon'],
                    'color_styles': ['blue', 'red', 'green', 'orange']
                }
            ]
        },
        'summary': {
            'name': 'Summary',
            'description': 'Overall comparison of all experiments',
            'sets': [
                {
                    'name': 'All Experiments',
                    'experiments': ['vanilla_sgd_l2', 'fast_sgd_l2', 'free_sgd_l2', 'fast_muon_l2muon', 'free_muon_l2muon'],
                    'color_styles': ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
                }
            ]

        }
    }
    
    return groups


def compare_test_curves_by_time(experiments_dir="model_pth", output_dir="results_viz_time", show_plots=False, filter_method=None, filter_optimizer=None, specific_experiments=None):
    """
    Compare test curves across different experiments using time as the x-axis
    """
    # Automatically discover all experiments in model_pth directory
    model_paths = discover_experiments_with_time(experiments_dir, specific_experiments)
    
    # Apply filters if specified
    if filter_method or filter_optimizer:
        filtered_paths = {}
        for exp_name, path in model_paths.items():
            include = True
            if filter_method and not exp_name.startswith(filter_method):
                include = False
            if filter_optimizer and f"_{filter_optimizer}_" not in exp_name and not exp_name.endswith(f"_{filter_optimizer}"):
                include = False
            if include:
                filtered_paths[exp_name] = path
        model_paths = filtered_paths
    
    if not model_paths:
        print("No experiments found in the model_pth directory!")
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
            df = load_test_data_with_time(path)
            model_data[name] = df
            print(f"Successfully loaded data for {name}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if not model_data:
        print("No data could be loaded from the available test.csv files!")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create grouped comparisons
    comparison_groups = group_experiments_for_comparison(model_paths.keys())
    
    # Generate plots for each comparison group
    for group_key, group_info in comparison_groups.items():
        print(f"\nGenerating plots for: {group_info['name']}")
        print(f"Description: {group_info['description']}")
        
        # Create a separate plot for each comparison set within the group
        for i, comparison_set in enumerate(group_info['sets']):
            set_name = comparison_set['name']
            experiments = comparison_set['experiments']
            colors = comparison_set['color_styles']
            
            print(f"  Creating comparison plot: {set_name}")
            
            # Filter data to only include experiments in this set
            filtered_data = {name: model_data[name] for name in experiments if name in model_data}
            
            if len(filtered_data) == 0:
                print(f"    No data available for this comparison set")
                continue
            
            # Create comparison plots using time as the x-axis
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{group_info["name"]}: {set_name}', fontsize=16)
            
            # Plot 1: Clean Test Error Comparison by Time
            ax1 = axes[0, 0]
            for j, (name, df) in enumerate(filtered_data.items()):
                clean_test = df[df['test_name'] == 'clean_test']
                if not clean_test.empty:  # Check if data exists
                    ax1.plot(clean_test['elapsed_time'], clean_test['error'], 
                             label=name, color=colors[j % len(colors)], 
                             linestyle='-', linewidth=1.5)
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Clean Test Error (%)')
            ax1.set_title('Clean Test Error Comparison by Time')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Adversarial Test Error Comparison by Time
            ax2 = axes[0, 1]
            for j, (name, df) in enumerate(filtered_data.items()):
                adv_test = df[df['test_name'] == 'adv_test']
                if not adv_test.empty:  # Check if data exists
                    ax2.plot(adv_test['elapsed_time'], adv_test['error'], 
                             label=name, color=colors[j % len(colors)], 
                             linestyle='-', linewidth=1.5)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Adversarial Test Error (%)')
            ax2.set_title('Adversarial Test Error Comparison by Time')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Training Error Comparison by Time
            ax3 = axes[1, 0]
            for j, (name, df) in enumerate(filtered_data.items()):
                clean_train = df[df['test_name'] == 'clean_train']
                if not clean_train.empty:  # Check if data exists
                    ax3.plot(clean_train['elapsed_time'], clean_train['error'], 
                             label=name, color=colors[j % len(colors)], 
                             linestyle='-', linewidth=1.5)
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Clean Training Error (%)')
            ax3.set_title('Clean Training Error Comparison by Time')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Adversarial Training Error Comparison by Time
            ax4 = axes[1, 1]
            for j, (name, df) in enumerate(filtered_data.items()):
                adv_train = df[df['test_name'] == 'adv_train']
                if not adv_train.empty:  # Check if data exists
                    ax4.plot(adv_train['elapsed_time'], adv_train['error'], 
                             label=name, color=colors[j % len(colors)], 
                             linestyle='-', linewidth=1.5)
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Adversarial Training Error (%)')
            ax4.set_title('Adversarial Training Error Comparison by Time')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the figure to results_viz_time directory with descriptive name
            safe_filename = set_name.replace(' ', '_').replace('+', '_').lower()
            output_path = os.path.join(output_dir, f'{group_key}_{safe_filename}_comparison_by_time.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    Saved comparison plot to: {output_path}")
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    # Also create the original all-in-one comparison plot
    create_original_comparison_plot(model_data, output_dir, show_plots)
    
    print(f"\nAll comparison plots saved to {output_dir}/ directory")


def create_original_comparison_plot(model_data, output_dir, show_plots):
    """
    Create the original all-in-one comparison plot
    """
    # Create comparison plots using time as the x-axis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']
    
    # Plot 1: Clean Test Error Comparison by Time
    ax1 = axes[0, 0]
    for i, (name, df) in enumerate(model_data.items()):
        clean_test = df[df['test_name'] == 'clean_test']
        if not clean_test.empty:  # Check if data exists
            ax1.plot(clean_test['elapsed_time'], clean_test['error'], 
                     label=name, color=colors[i % len(colors)], 
                     linestyle='-', linewidth=1.5)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Clean Test Error (%)')
    ax1.set_title('Clean Test Error Comparison by Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Adversarial Test Error Comparison by Time
    ax2 = axes[0, 1]
    for i, (name, df) in enumerate(model_data.items()):
        adv_test = df[df['test_name'] == 'adv_test']
        if not adv_test.empty:  # Check if data exists
            ax2.plot(adv_test['elapsed_time'], adv_test['error'], 
                     label=name, color=colors[i % len(colors)], 
                     linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Adversarial Test Error (%)')
    ax2.set_title('Adversarial Test Error Comparison by Time')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Error Comparison by Time
    ax3 = axes[1, 0]
    for i, (name, df) in enumerate(model_data.items()):
        clean_train = df[df['test_name'] == 'clean_train']
        if not clean_train.empty:  # Check if data exists
            ax3.plot(clean_train['elapsed_time'], clean_train['error'], 
                     label=name, color=colors[i % len(colors)], 
                     linestyle='-', linewidth=1.5)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Clean Training Error (%)')
    ax3.set_title('Clean Training Error Comparison by Time')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Adversarial Training Error Comparison by Time
    ax4 = axes[1, 1]
    for i, (name, df) in enumerate(model_data.items()):
        adv_train = df[df['test_name'] == 'adv_train']
        if not adv_train.empty:  # Check if data exists
            ax4.plot(adv_train['elapsed_time'], adv_train['error'], 
                     label=name, color=colors[i % len(colors)], 
                     linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Adversarial Training Error (%)')
    ax4.set_title('Adversarial Training Error Comparison by Time')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure to results_viz_time directory
    output_path = os.path.join(output_dir, 'test_curves_comparison_by_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved original test curves comparison by time to: {output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Create another plot showing the robustness (gap between clean and adversarial) over time
    plt.figure(figsize=(12, 8))
    
    for i, (name, df) in enumerate(model_data.items()):
        clean_test = df[df['test_name'] == 'clean_test']
        adv_test = df[df['test_name'] == 'adv_test']
        
        if not clean_test.empty and not adv_test.empty:
            # Create a common time base by combining all unique times and sorting them
            all_times = sorted(set(clean_test['elapsed_time'].tolist() + adv_test['elapsed_time'].tolist()))
            
            if len(all_times) > 0:
                # Interpolate clean and adversarial errors at common time points
                from scipy.interpolate import interp1d
                
                # Handle cases where there might be duplicate time values by averaging
                clean_time_to_error = {}
                for _, row in clean_test.iterrows():
                    t, err = row['elapsed_time'], row['error']
                    if t in clean_time_to_error:
                        # Average duplicate time values
                        clean_time_to_error[t] = (clean_time_to_error[t] + err) / 2
                    else:
                        clean_time_to_error[t] = err
                
                adv_time_to_error = {}
                for _, row in adv_test.iterrows():
                    t, err = row['elapsed_time'], row['error']
                    if t in adv_time_to_error:
                        # Average duplicate time values
                        adv_time_to_error[t] = (adv_time_to_error[t] + err) / 2
                    else:
                        adv_time_to_error[t] = err
                
                # Create sorted lists for interpolation functions
                clean_times = sorted(clean_time_to_error.keys())
                clean_errors = [clean_time_to_error[t] for t in clean_times]
                adv_times = sorted(adv_time_to_error.keys())
                adv_errors = [adv_time_to_error[t] for t in adv_times]
                
                # Create interpolation functions
                import numpy as np
                if len(clean_times) > 1:
                    # Use numpy interpolation for clean errors
                    clean_interp_errors = np.interp(all_times, clean_times, clean_errors)
                else:
                    # If only one point, repeat it for all time points
                    clean_interp_errors = [clean_errors[0]] * len(all_times)
                
                if len(adv_times) > 1:
                    # Use numpy interpolation for adversarial errors
                    adv_interp_errors = np.interp(all_times, adv_times, adv_errors)
                else:
                    # If only one point, repeat it for all time points
                    adv_interp_errors = [adv_errors[0]] * len(all_times)
                
                # Calculate robustness gaps at common time points
                robustness_gaps = [adv_err - clean_err for clean_err, adv_err in zip(clean_interp_errors, adv_interp_errors)]
                
                plt.plot(all_times, robustness_gaps, label=f'{name}', 
                         color=colors[i % len(colors)], 
                         linestyle='-',
                         linewidth=1.5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Robustness Gap (Adv - Clean Error) (%)')
    plt.title('Robustness Gap Comparison Across Models by Time')
    plt.legend(loc='upper left', fontsize='xx-small')
    plt.grid(True, alpha=0.3)
    
    # Save the robustness plot to results_viz_time directory
    robustness_output_path = os.path.join(output_dir, 'robustness_comparison_by_time.png')
    plt.savefig(robustness_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved robustness comparison by time to: {robustness_output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Create a performance comparison plot showing final accuracy by time
    plt.figure(figsize=(12, 8))
    
    final_results = {}
    for name, df in model_data.items():
        # Get the final test errors for each model
        try:
            model_df = df
            clean_test = model_df[model_df['test_name'] == 'clean_test']
            adv_test = model_df[model_df['test_name'] == 'adv_test']
            
            if not clean_test.empty and not adv_test.empty:
                # Get the latest time and corresponding errors
                final_clean_error = clean_test.loc[clean_test['elapsed_time'].idxmax()]['error'] if not clean_test.empty else None
                final_adv_error = adv_test.loc[adv_test['elapsed_time'].idxmax()]['error'] if not adv_test.empty else None
                final_time = max(clean_test['elapsed_time'].max(), adv_test['elapsed_time'].max()) if not clean_test.empty and not adv_test.empty else None
                
                if final_clean_error is not None and final_adv_error is not None and final_time is not None:
                    final_results[name] = {
                        'clean_acc': 100 - final_clean_error,
                        'adv_acc': 100 - final_adv_error,
                        'time': final_time
                    }
        except Exception as e:
            print(f"Error processing final results for {name}: {e}")
    
    if final_results:
        names = list(final_results.keys())
        clean_accs = [final_results[name]['clean_acc'] for name in names]
        adv_accs = [final_results[name]['adv_acc'] for name in names]
        times = [final_results[name]['time'] for name in names]
        
        # Plot accuracy vs time
        plt.scatter(clean_accs, times, label='Clean Accuracy', marker='o', s=60, alpha=0.7)
        plt.scatter(adv_accs, times, label='Adversarial Accuracy', marker='s', s=60, alpha=0.7)
        
        # Add labels to each point
        for i, name in enumerate(names):
            plt.annotate(name, (clean_accs[i], times[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            if i < len(adv_accs):  # Check if there's a corresponding adversarial accuracy
                plt.annotate(name, (adv_accs[i], times[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
        
        plt.xlabel('Accuracy (%)')
        plt.ylabel('Time to Convergence (seconds)')
        plt.title('Accuracy vs Time to Convergence Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the performance plot to results_viz_time directory
        performance_output_path = os.path.join(output_dir, 'performance_comparison.png')
        plt.savefig(performance_output_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison to: {performance_output_path}")
        if show_plots:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare test curves across different experiments by time')
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                        help='Directory containing experiment results (default: experiments)')
    parser.add_argument('--output_dir', type=str, default='results_viz_time',
                        help='Directory to save output plots (default: results_viz_time)')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots interactively (default: False)')
    parser.add_argument('--filter_method', type=str, choices=['fast', 'free', 'vanilla'],
                        help='Filter experiments by method (fast, free, vanilla)')
    parser.add_argument('--filter_optimizer', type=str, choices=['sgd', 'adam', 'muon'],
                        help='Filter experiments by optimizer (sgd, adam, muon)')
    parser.add_argument('--specific_experiments', nargs='*', default=['fast_muon_l2', 'fast_muon_l2muon', 'fast_sgd_l2', 'fast_sgd_l2muon', 'free_muon_l2', 'free_muon_l2muon', 'free_sgd_l2', 'free_sgd_l2muon', 'vanilla_muon_l2', 'vanilla_muon_l2muon', 'vanilla_sgd_l2', 'vanilla_sgd_l2muon'],
                        help='Specific experiment directories to compare (e.g., fast_sgd_l2 free_adam_l2muon)')
    
    args = parser.parse_args()
    
    # Run comparison using command line arguments
    compare_test_curves_by_time(
        args.experiments_dir, 
        args.output_dir, 
        args.show_plots,
        args.filter_method,
        args.filter_optimizer,
        args.specific_experiments
    )
    print(f"Comparison plots saved to {args.output_dir}/ directory")


if __name__ == "__main__":
    main()