import pandas as pd
import os
import argparse
from pathlib import Path

def load_test_data_with_time(csv_path):
    """
    加载test.csv文件，包含时间数据
    格式: epoch, loss, error, elapsed_time
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test CSV file not found: {csv_path}")
    
    data = []
    try:
        with open(csv_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(',')
                if len(parts) == 4:  # 新格式: epoch, loss, error, time
                    epoch, loss, error, elapsed_time = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    data.append([epoch, loss, error, elapsed_time])
                elif len(parts) == 3:  # 旧格式: epoch, loss, error
                    epoch, loss, error = int(parts[0]), float(parts[1]), float(parts[2])
                    data.append([epoch, loss, error, 0.0])  # 时间设为0
                else:
                    print(f"Warning: Line {line_num} in {csv_path} has {len(parts)} columns, expected 3 or 4. Skipping...")
                    continue
    except Exception as e:
        raise IOError(f"Error reading file {csv_path}: {e}")
    
    if not data:
        raise ValueError(f"No valid data found in {csv_path}")
    
    df = pd.DataFrame(data, columns=['epoch', 'loss', 'error', 'elapsed_time'])
    
    # 确定每行代表的测试类型
    df['test_type'] = df.index % 4
    df['test_name'] = df['test_type'].map({
        0: 'clean_train',
        1: 'adv_train', 
        2: 'clean_test',
        3: 'adv_test'
    })
    
    return df

def discover_experiments(experiments_dir="experiments"):
    """发现所有实验"""
    if not os.path.exists(experiments_dir):
        print(f"Error: {experiments_dir} directory does not exist!")
        return {}
    
    experiment_paths = {}
    # 自动发现所有包含test.csv的目录
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)
        test_csv_path = os.path.join(exp_path, "test.csv")
        
        if os.path.isdir(exp_path) and os.path.exists(test_csv_path):
            experiment_paths[exp_dir] = test_csv_path
    
    return experiment_paths

def extract_final_results(experiments_dir="experiments", output_file="results/final_results.csv"):
    """
    提取所有模型的最终结果
    返回包含速度、准确率、鲁棒性的DataFrame
    """
    model_paths = discover_experiments(experiments_dir)
    
    if not model_paths:
        print("No experiments found in the experiments directory!")
        return None
    
    print(f"Found {len(model_paths)} experiments")
    
    results = []
    
    for name, path in model_paths.items():
        try:
            df = load_test_data_with_time(path)
            
            # 获取最后一个epoch的数据
            max_epoch = df['epoch'].max()
            final_data = df[df['epoch'] == max_epoch]
            
            if len(final_data) < 4:
                print(f"Warning: {name} doesn't have complete data for epoch {max_epoch}")
                continue
            
            # 提取各测试类型的最终结果
            clean_test = final_data[final_data['test_name'] == 'clean_test']
            adv_test = final_data[final_data['test_name'] == 'adv_test']
            clean_train = final_data[final_data['test_name'] == 'clean_train']
            adv_train = final_data[final_data['test_name'] == 'adv_train']
            
            if clean_test.empty or adv_test.empty:
                print(f"Warning: {name} missing test data")
                continue
            
            # 提取指标
            clean_test_error = clean_test['error'].iloc[0]
            adv_test_error = adv_test['error'].iloc[0]
            clean_test_accuracy = 100.0 - clean_test_error
            adv_test_accuracy = 100.0 - adv_test_error
            
            # 鲁棒性 = 对抗测试误差 - 干净测试误差
            robustness_gap = adv_test_error - clean_test_error
            
            # 速度 = 最后一个epoch的总训练时间（秒）
            total_time = clean_test['elapsed_time'].iloc[0] if not clean_test.empty else 0.0
            
            # 解析实验名称
            parts = name.split('_')
            training_method = parts[0] if len(parts) > 0 else 'unknown'
            optimizer = parts[1] if len(parts) > 1 else 'unknown'
            attack_type = parts[2] if len(parts) > 2 else 'unknown'
            
            results.append({
                'model_name': name,
                'training_method': training_method,
                'optimizer': optimizer,
                'attack_type': attack_type,
                'final_epoch': max_epoch,
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60.0,
                'total_time_hours': total_time / 3600.0,
                'clean_test_error': clean_test_error,
                'clean_test_accuracy': clean_test_accuracy,
                'adv_test_error': adv_test_error,
                'adv_test_accuracy': adv_test_accuracy,
                'robustness_gap': robustness_gap,
                'clean_train_error': clean_train['error'].iloc[0] if not clean_train.empty else None,
                'adv_train_error': adv_train['error'].iloc[0] if not adv_train.empty else None,
            })
            
            print(f"✓ Extracted results for {name}")
            
        except Exception as e:
            print(f"✗ Error processing {name}: {e}")
            continue
    
    if not results:
        print("No results could be extracted!")
        return None
    
    # 创建DataFrame
    results_df = pd.DataFrame(results)
    
    # 按训练方法、优化器、攻击类型排序
    results_df = results_df.sort_values(['training_method', 'optimizer', 'attack_type'])
    
    # 保存结果
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"\n✓ Results saved to: {output_file}")
    
    # 打印摘要
    print("\n" + "="*120)
    print("最终结果摘要")
    print("="*120)
    print(f"\n{'模型名称':<30} {'训练方法':<12} {'优化器':<10} {'攻击类型':<12} {'总时间(小时)':<15} "
          f"{'干净准确率(%)':<15} {'对抗准确率(%)':<15} {'鲁棒性差距(%)':<15}")
    print("-"*120)
    
    for _, row in results_df.iterrows():
        time_str = f"{row['total_time_hours']:.2f}" if row['total_time_hours'] > 0 else "N/A"
        print(f"{row['model_name']:<30} {row['training_method']:<12} {row['optimizer']:<10} {row['attack_type']:<12} "
              f"{time_str:<15} {row['clean_test_accuracy']:<15.2f} {row['adv_test_accuracy']:<15.2f} "
              f"{row['robustness_gap']:<15.2f}")
    
    print("\n" + "="*100)
    print("按训练方法分组统计")
    print("="*100)
    
    for method in results_df['training_method'].unique():
        method_data = results_df[results_df['training_method'] == method]
        print(f"\n{method.upper()} 方法:")
        print(f"  平均训练时间: {method_data['total_time_hours'].mean():.2f} 小时")
        print(f"  平均干净准确率: {method_data['clean_test_accuracy'].mean():.2f}%")
        print(f"  平均对抗准确率: {method_data['adv_test_accuracy'].mean():.2f}%")
        print(f"  平均鲁棒性差距: {method_data['robustness_gap'].mean():.2f}%")
    
    print("\n" + "="*100)
    print("按优化器分组统计")
    print("="*100)
    
    for optimizer in results_df['optimizer'].unique():
        opt_data = results_df[results_df['optimizer'] == optimizer]
        print(f"\n{optimizer.upper()} 优化器:")
        print(f"  平均训练时间: {opt_data['total_time_hours'].mean():.2f} 小时")
        print(f"  平均干净准确率: {opt_data['clean_test_accuracy'].mean():.2f}%")
        print(f"  平均对抗准确率: {opt_data['adv_test_accuracy'].mean():.2f}%")
        print(f"  平均鲁棒性差距: {opt_data['robustness_gap'].mean():.2f}%")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='提取不同模型的最终结果（速度、准确率、鲁棒性）')
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                        help='实验数据目录 (default: experiments)')
    parser.add_argument('--output_file', type=str, default='results/final_results.csv',
                        help='输出CSV文件路径 (default: results/final_results.csv)')
    
    args = parser.parse_args()
    
    extract_final_results(args.experiments_dir, args.output_file)

# # 使用默认参数
# python src/visualization/extract_results.py
# # 指定实验目录和输出文件
# python src/visualization/extract_results.py --experiments_dir experiments --output_file results/final_results.csv
if __name__ == "__main__":
    main()

