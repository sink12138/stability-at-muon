import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def load_test_data_with_time(csv_path):
    """
    加载包含时间数据的test.csv文件
    格式：epoch, loss, error, elapsed_time
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到测试CSV文件: {csv_path}")
    
    data = []
    try:
        with open(csv_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                parts = line.split(',')
                if len(parts) == 4:  # 新格式：epoch, loss, error, time
                    epoch, loss, error, elapsed_time = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    data.append([epoch, loss, error, elapsed_time])
                elif len(parts) == 3:  # 旧格式：epoch, loss, error
                    epoch, loss, error = int(parts[0]), float(parts[1]), float(parts[2])
                    data.append([epoch, loss, error, 0.0])  # 时间设为0
                else:
                    print(f"警告: {csv_path}第{line_num}行有{len(parts)}列，期望3或4列。跳过...")
                    continue
    except Exception as e:
        raise IOError(f"读取文件{csv_path}时出错: {e}")
    
    if not data:
        raise ValueError(f"在{csv_path}中未找到有效数据")
    
    df = pd.DataFrame(data, columns=['epoch', 'loss', 'error', 'elapsed_time'])
    
    # 确定每行代表的测试类型
    df['test_type'] = df.index % 4
    df['test_name'] = df['test_type'].map({
        0: 'clean_train',
        1: 'adv_train', 
        2: 'clean_test',
        3: 'adv_test'
    })
    
    print(f"从{csv_path}加载了{len(df)}个数据点")
    return df

def plot_test_errors_by_time(csv_path):
    """按时间绘制测试错误率"""
    df = load_test_data_with_time(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制每种测试类型
    for test_type in df['test_name'].unique():
        subset = df[df['test_name'] == test_type]
        times = subset['elapsed_time']
        errors = subset['error']
        
        linestyle = '-' if 'clean' in test_type else '--'
        color = 'blue' if 'train' in test_type else 'red'
        
        plt.plot(times, errors, label=test_type, linestyle=linestyle, color=color, marker='o', markersize=4)
    
    plt.xlabel('训练时间 (秒)')
    plt.ylabel('错误率 (%)')
    plt.title('测试错误率随时间变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_accuracies_by_time(csv_path):
    """按时间绘制准确率"""
    df = load_test_data_with_time(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # 转换为准确率
    df['accuracy'] = 100 - df['error']
    
    # 绘制每种测试类型
    for test_type in df['test_name'].unique():
        subset = df[df['test_name'] == test_type]
        times = subset['elapsed_time']
        accuracies = subset['accuracy']
        
        linestyle = '-' if 'clean' in test_type else '--'
        color = 'blue' if 'train' in test_type else 'red'
        
        plt.plot(times, accuracies, label=f'{test_type}_accuracy', linestyle=linestyle, color=color, marker='o', markersize=4)
    
    plt.xlabel('训练时间 (秒)')
    plt.ylabel('准确率 (%)')
    plt.title('模型准确率随时间变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_comparison_by_time(csv_path):
    """按时间绘制干净vs对抗性能对比"""
    df = load_test_data_with_time(csv_path)
    
    plt.figure(figsize=(15, 10))
    
    # 分离干净和对抗结果
    clean_train = df[df['test_name'] == 'clean_train']
    adv_train = df[df['test_name'] == 'adv_train']
    clean_test = df[df['test_name'] == 'clean_test']
    adv_test = df[df['test_name'] == 'adv_test']
    
    # 干净vs对抗训练
    plt.subplot(2, 2, 1)
    plt.plot(clean_train['elapsed_time'], clean_train['error'], label='Clean Train', marker='o', markersize=4)
    plt.plot(adv_train['elapsed_time'], adv_train['error'], label='Adversarial Train', linestyle='--', marker='s', markersize=4)
    plt.xlabel('训练时间 (秒)')
    plt.ylabel('错误率 (%)')
    plt.title('训练: 干净 vs 对抗')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 干净vs对抗测试
    plt.subplot(2, 2, 2)
    plt.plot(clean_test['elapsed_time'], clean_test['error'], label='Clean Test', marker='o', markersize=4)
    plt.plot(adv_test['elapsed_time'], adv_test['error'], label='Adversarial Test', linestyle='--', marker='s', markersize=4)
    plt.xlabel('训练时间 (秒)')
    plt.ylabel('错误率 (%)')
    plt.title('测试: 干净 vs 对抗')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 准确率对比训练
    plt.subplot(2, 2, 3)
    plt.plot(clean_train['elapsed_time'], 100 - clean_train['error'], label='Clean Train Acc', marker='o', markersize=4)
    plt.plot(adv_train['elapsed_time'], 100 - adv_train['error'], label='Adversarial Train Acc', linestyle='--', marker='s', markersize=4)
    plt.xlabel('训练时间 (秒)')
    plt.ylabel('准确率 (%)')
    plt.title('训练准确率: 干净 vs 对抗')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 准确率对比测试
    plt.subplot(2, 2, 4)
    plt.plot(clean_test['elapsed_time'], 100 - clean_test['error'], label='Clean Test Acc', marker='o', markersize=4)
    plt.plot(adv_test['elapsed_time'], 100 - adv_test['error'], label='Adversarial Test Acc', linestyle='--', marker='s', markersize=4)
    plt.xlabel('训练时间 (秒)')
    plt.ylabel('准确率 (%)')
    plt.title('测试准确率: 干净 vs 对抗')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_epoch_vs_time(csv_path):
    """绘制epoch与时间的关系"""
    df = load_test_data_with_time(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # 只取clean_test的数据来避免重复
    clean_test = df[df['test_name'] == 'clean_test']
    
    plt.plot(clean_test['epoch'], clean_test['elapsed_time'], marker='o', markersize=6, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('累计训练时间 (秒)')
    plt.title('Epoch与训练时间的关系')
    plt.grid(True, alpha=0.3)
    
    # 添加时间标注
    for i, (epoch, time_val) in enumerate(zip(clean_test['epoch'], clean_test['elapsed_time'])):
        if i % 5 == 0:  # 每5个epoch标注一次
            plt.annotate(f'{time_val:.0f}s', (epoch, time_val), textcoords="offset points", xytext=(0,10), ha='center')
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='可视化包含时间数据的测试结果')
    parser.add_argument('--csv_path', type=str, default='model_pth/free_muon_aux_L2muon/test.csv', 
                        help='test.csv文件路径')
    parser.add_argument('--output_dir', type=str, default='results_viz_time', 
                        help='保存可视化图片的目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"错误: {args.csv_path} 不存在!")
        return
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 创建可视化
    print("创建按时间的错误率图...")
    plt1 = plot_test_errors_by_time(args.csv_path)
    plt1.savefig(os.path.join(args.output_dir, 'test_errors_by_time.png'), dpi=300, bbox_inches='tight')
    plt1.show()
    
    print("创建按时间的准确率图...")
    plt2 = plot_accuracies_by_time(args.csv_path)
    plt2.savefig(os.path.join(args.output_dir, 'accuracies_by_time.png'), dpi=300, bbox_inches='tight')
    plt2.show()
    
    print("创建按时间的对比图...")
    plt3 = plot_comparison_by_time(args.csv_path)
    plt3.savefig(os.path.join(args.output_dir, 'comparison_by_time.png'), dpi=300, bbox_inches='tight')
    plt3.show()
    
    print("创建epoch与时间关系图...")
    plt4 = plot_epoch_vs_time(args.csv_path)
    plt4.savefig(os.path.join(args.output_dir, 'epoch_vs_time.png'), dpi=300, bbox_inches='tight')
    plt4.show()
    
    print(f"可视化图片已保存到 {args.output_dir}/")

if __name__ == "__main__":
    main()
