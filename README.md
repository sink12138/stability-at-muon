## 目录结构

```
MUON/
├── main.py                    # 主入口文件，提供统一的项目访问接口
├── README.md                  # 项目说明文档
├── results/                   # 结果输出目录
├── src/                       # 源代码主目录
│   ├── optimizers/            # 优化器相关代码
│   │   └── muon.py           # MUON优化器实现
│   ├── attacks/               # 对抗攻击算法
│   │   └── pgd_attack.py     # PGD攻击算法实现
│   ├── models/                # 神经网络模型定义
│   │   ├── basic_models.py   # 基础模型（ResNet等）
│   │   └── wideresnet.py     # WideResNet模型
│   ├── training/              # 训练相关代码
│   │   ├── train.py          # 标准训练脚本
│   │   └── train_free_trades.py  # FreeTRADES训练脚本
│   ├── testing/               # 测试相关代码
│   │   ├── test_blackbox.py  # 黑盒测试
│   │   └── test_transfer.py  # 迁移攻击测试
│   ├── visualization/         # 可视化和结果分析
│   │   ├── visualize_training.py    # 训练过程可视化
│   │   ├── visualize_test_results.py # 测试结果可视化
│   │   └── compare_test_curves.py   # 测试曲线对比
│   └── utils/                 # 工具函数和脚本
│       └── run_experiments.py # 批量实验运行脚本
└── *.png                      # 各种结果图表文件

```

## 使用方法

### 通过主入口文件运行

```bash
# 运行训练
python main.py train --method standard --training_args --data cifar10 --model res18

# 运行测试
python main.py test --test_type blackbox --testing_args --model_path path/to/model

# 运行可视化
python main.py visualize --viz_type training --viz_args --input path/to/data

# 运行批量实验
python main.py experiments
```

### 直接运行模块

也可以直接运行各个模块的脚本：

```bash
# 训练
python src/training/train.py --data cifar10 --model res18 --method vanilla

# 测试
python src/testing/test_blackbox.py --model_path path/to/model

# 可视化
python src/visualization/visualize_training.py --input path/to/data
```

## 模块说明

### optimizers/

包含 MUON 优化器的实现，这是项目的核心创新点。

### attacks/

包含各种对抗攻击算法的实现，主要用于生成对抗样本进行对抗训练。

### models/

包含各种神经网络模型的定义，包括 ResNet、WideResNet 等。

### training/

包含训练脚本，支持标准训练和 FreeTRADES 等高级训练方法。

### testing/

包含模型测试脚本，支持黑盒攻击测试和迁移攻击测试。

### visualization/

包含结果可视化工具，用于分析训练过程和测试结果。

### utils/

包含实用工具，如批量实验运行脚本等。

# 项目结构文档

```
stability-at-muon/
├── src/                          # 源代码目录
│   ├── optimizers/               # 优化器相关代码
│   │   ├── __init__.py
│   │   └── muon.py              # Muon优化器实现
│   ├── attacks/                  # 攻击算法
│   │   ├── __init__.py
│   │   └── pgd_attack.py        # PGD攻击实现
│   ├── models/                   # 模型定义
│   │   ├── __init__.py
│   │   ├── basic_models.py      # 基础模型
│   │   └── wideresnet.py        # WideResNet模型
│   ├── training/                 # 训练相关
│   │   ├── __init__.py
│   │   ├── train.py             # 主要训练脚本
│   │   └── train_free_trades.py # FreeTRADES训练
│   ├── testing/                  # 测试相关
│   │   ├── __init__.py
│   │   ├── test_blackbox.py     # 黑盒测试
│   │   └── test_transfer.py     # 迁移测试
│   ├── visualization/            # 可视化
│   │   ├── __init__.py
│   │   ├── visualize_training.py       # 训练可视化
│   │   ├── visualize_test_results.py   # 测试结果可视化
│   │   └── compare_test_curves.py      # 测试曲线对比
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       └── run_experiments.py   # 实验运行
├── configs/                      # 配置文件
├── experiments/                  # 实验结果
├── model_pth/                    # 模型权重
├── results/                      # 结果输出
├── main.py                       # 项目主入口
├── PROJECT_STRUCTURE.md          # 本文件
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件
```

## 使用方法

### 通过主入口运行

```bash
# 查看帮助
python3 main.py --help

# 运行训练
python3 main.py train --method vanilla

# 运行测试
python3 main.py test --test-type blackbox

# 运行可视化
python3 main.py visualize --viz-type training
```

### 作为 Python 包导入

```python
import sys
sys.path.append('src')

from optimizers.muon import SingleDeviceMuon
from attacks.pgd_attack import L2PGDAttack
from models.basic_models import ResNet18
```

python main.py visualize --viz_type times -- --csv_path model_pth/fast_muon_l2/test.csv --output_dir model_pth/fast_adam_l2muon
