## Attack: L2 vs L2Muon

### 1.Vanilla SGD

| Method    | Vanilla | Vanilla |
| --------- | ------- | ------- |
| Optimizer | SGD     | SGD     |
| Attack    | L2      | L2Muon  |

![alt text](model_pth/attack/vanilla_sgd_L2muon/train_comparison.png)

### 2.Fast SGD

| Method    | Fast | Fast   |
| --------- | ---- | ------ |
| Optimizer | SGD  | SGD    |
| Attack    | L2   | L2Muon |

![alt text](model_pth/attack/fast_sgd_L2muon/train_comparison.png)

### 3.Fast Muon

| Method    | Fast | Fast   |
| --------- | ---- | ------ |
| Optimizer | Muon | Muon   |
| Attack    | L2   | L2Muon |

![alt text](model_pth/attack/fast_muon_aux_L2muon/train_comparison.png)

### 4.Attack: L2 VS L2Muon VS L2Muon_0

| Method    | Fast | Fast   | Fast     |
| --------- | ---- | ------ | -------- |
| Optimizer | SGD  | SGD    | SGD      |
| Attack    | L2   | L2Muon | L2Muon_0 |

![alt text](model_pth/pic/train_comparison.png)

## Muon

| Method    | Fast | Fast   | Fast | Fast   |
| --------- | ---- | ------ | ---- | ------ |
| Optimizer | SGD  | SGD    | Muon | Muon   |
| Attack    | L2   | L2Muon | L2   | L2Muon |

![alt text](train_comparison.png)
