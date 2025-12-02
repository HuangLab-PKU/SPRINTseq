# CIDRE 参数优化指南

## 概述

CIDRE (Correction of Illumination Distribution by Regularized Energy minimization) 是一个用于校正显微镜图像照明不均的工具。本指南将帮助您根据不同的数据类型优化CIDRE参数。

## 核心参数说明

### 1. lambda_v（空间正则化强度）**最重要**

**默认值**: 6.0  
**范围**: 5.0 - 9.5

**作用**: 控制照明增益表面的平滑程度
- **较高值 (8-9.5)**: 产生更平滑的照明表面，适合:
  - 图像数量少 (<50张)
  - 照明变化缓慢均匀
  - 信噪比较低的数据
  
- **中等值 (6-7)**: 平衡平滑性和细节，适合:
  - 图像数量中等 (50-200张)
  - 大多数常规应用
  
- **较低值 (5-6)**: 保留更多细节，适合:
  - 图像数量多 (>200张)
  - 照明变化复杂
  - 需要校正局部照明不均

### 2. lambda_z（零光正则化强度）

**默认值**: 0.5  
**范围**: 0.3 - 3.0

**作用**: 控制背景（零光）表面的均匀性
- **较高值 (1.5-3.0)**: 强制更均匀的背景，适合:
  - 背景噪声大
  - 暗电流不稳定
  - 荧光显微镜
  
- **中等值 (0.5-1.0)**: 标准设置，适合:
  - 大多数应用
  
- **较低值 (0.3-0.5)**: 允许背景变化，适合:
  - 明场显微镜
  - 背景本身有有意义的变化

### 3. correction_mode（校正模式）

**默认值**: 2

- **模式 0** "保持零光": 保留原始强度范围和零光水平
  - 用于: 需要保持绝对强度值的定量分析
  
- **模式 1** "动态范围校正": 保留原始强度范围
  - 用于: 需要保持强度范围但校正照明的情况
  
- **模式 2** "直接校正"（推荐）: 直接除以照明增益
  - 用于: 大多数应用，最彻底的校正

### 4. q_percent（鲁棒均值比例）

**默认值**: 0.25  
**范围**: 0.1 - 0.5

**作用**: 确定用于估计Q（底层强度分布）的数据比例
- **较小值 (0.1-0.2)**: 更鲁棒，但可能丢失信息
- **中等值 (0.25)**: 平衡
- **较大值 (0.3-0.5)**: 使用更多数据，适合干净数据

### 5. z_limits（零光限制）

**默认值**: 自动检测  
**格式**: [最小值, 最大值]

**作用**: 手动指定背景强度范围
- 可通过拍摄暗帧（关闭光源）图像来确定
- 例如: `[98, 105]` 表示背景在98-105之间

### 6. max_lbfgs_iterations（最大迭代次数）

**默认值**: 500  
**范围**: 300 - 1000

**作用**: L-BFGS优化算法的最大迭代次数
- **增加**: 对于复杂照明模式，可增加到800-1000
- **减少**: 快速测试时可减少到300

## 针对不同应用的推荐参数

### 1. 空间转录组学 / FISH 数据（本项目）

```python
from lib.cidre import cidre_walk

cidre_walk(
    in_dir='input_folder',
    out_dir='output_folder',
    lambda_v=6.5,        # 中等平滑
    lambda_z=0.8,        # 中等背景正则化
    correction_mode=2,   # 直接校正
    q_percent=0.25       # 默认
)
```

**理由**:
- FISH数据通常有50-200张图像，使用中等lambda_v
- 荧光背景较均匀，使用中等lambda_z
- 需要彻底校正，使用模式2

### 2. 荧光显微镜（低信噪比）

```python
cidre_walk(
    in_dir='input_folder',
    out_dir='output_folder',
    lambda_v=7.5,        # 较高平滑（抑制噪声）
    lambda_z=2.0,        # 较高背景正则化
    correction_mode=2,
    q_percent=0.2        # 更鲁棒
)
```

### 3. 明场显微镜

```python
cidre_walk(
    in_dir='input_folder',
    out_dir='output_folder',
    lambda_v=5.5,        # 较低平滑（保留细节）
    lambda_z=0.4,        # 较低背景正则化
    correction_mode=2,
    q_percent=0.3
)
```

### 4. 图像数量很少（<30张）

```python
cidre_walk(
    in_dir='input_folder',
    out_dir='output_folder',
    lambda_v=8.5,        # 高平滑以补偿数据不足
    lambda_z=1.5,        # 高背景正则化
    correction_mode=2,
    max_lbfgs_iterations=800  # 增加迭代
)
```

### 5. 背景极不均匀

```python
cidre_walk(
    in_dir='input_folder',
    out_dir='output_folder',
    lambda_v=6.0,
    lambda_z=2.5,        # 强制背景均匀
    correction_mode=2,
    z_limits=[95, 110]   # 手动设置背景范围
)
```

## 参数调优策略

### 步骤1: 使用默认参数测试

```python
from lib.cidre import cidre_correct

# 先在一个小数据集上测试
cidre_correct('test_input', 'test_output')
```

### 步骤2: 评估结果

检查校正后的图像:
1. **过度平滑**: 如果照明梯度被过度校正，边缘区域看起来不自然
   → 减少 `lambda_v`
   
2. **欠校正**: 如果仍能看到明显的照明不均
   → 增加 `lambda_v`
   
3. **背景不均**: 如果背景有明显的空间变化
   → 增加 `lambda_z`

### 步骤3: 参数微调

每次调整一个参数，观察效果:

```python
# 测试不同的lambda_v
for lv in [5.5, 6.0, 6.5, 7.0, 7.5]:
    cidre_correct(
        'test_input', 
        f'test_output_lv{lv}',
        lambda_v=lv
    )
```

### 步骤4: 应用到完整数据集

```python
# 确定最佳参数后，应用到完整数据
cidre_walk(
    in_dir='full_input',
    out_dir='full_output',
    lambda_v=6.5,  # 你的最佳值
    lambda_z=0.8   # 你的最佳值
)
```

## 常见问题

### Q1: 校正后图像边缘出现伪影？
**A**: 减少 `lambda_v`，或增加图像数量

### Q2: 校正效果不明显？
**A**: 增加 `lambda_v`，检查图像是否真的有照明不均

### Q3: 背景出现条纹或波纹？
**A**: 增加 `lambda_z`，考虑设置 `z_limits`

### Q4: 优化时间过长？
**A**: 减少 `max_lbfgs_iterations` 到 300-400

### Q5: 不同cycle之间校正效果不一致？
**A**: 确保每个cycle有足够的图像（>20张），增加 `lambda_v` 和 `lambda_z`

## 高级技巧

### 1. 使用暗帧确定z_limits

```python
import numpy as np
from skimage.io import imread

# 读取暗帧图像（关闭光源拍摄）
dark_frames = [imread(f'dark_{i}.tif') for i in range(10)]
dark_stack = np.stack(dark_frames)

# 计算背景范围
z_min = float(np.percentile(dark_stack, 1))
z_max = float(np.percentile(dark_stack, 99))

print(f"建议使用 z_limits=[{z_min:.1f}, {z_max:.1f}]")

# 应用到CIDRE
cidre_walk(
    'input', 'output',
    z_limits=[z_min, z_max]
)
```

### 2. 批量测试参数组合

```python
import itertools

# 定义参数范围
lambda_v_range = [6.0, 6.5, 7.0]
lambda_z_range = [0.5, 1.0, 1.5]

# 测试所有组合
for lv, lz in itertools.product(lambda_v_range, lambda_z_range):
    output_dir = f'test_output_lv{lv}_lz{lz}'
    cidre_correct(
        'test_input',
        output_dir,
        lambda_v=lv,
        lambda_z=lz
    )
    print(f"完成: lambda_v={lv}, lambda_z={lz}")
```

### 3. 可视化校正效果

```python
import matplotlib.pyplot as plt
from skimage.io import imread

# 读取原始和校正后的图像
original = imread('input/image.tif')
corrected = imread('output/image.tif')

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(corrected, cmap='gray')
axes[1].set_title('校正后')
axes[1].axis('off')

# 显示强度分布
axes[2].hist(original.ravel(), bins=100, alpha=0.5, label='原始')
axes[2].hist(corrected.ravel(), bins=100, alpha=0.5, label='校正后')
axes[2].set_xlabel('像素强度')
axes[2].set_ylabel('频数')
axes[2].legend()
axes[2].set_title('强度分布对比')

plt.tight_layout()
plt.savefig('cidre_comparison.png', dpi=300)
```

## 参考资料

- CIDRE原文: Smith et al. "CIDRE: an illumination-correction method for optical microscopy" Nature Methods (2015)
- GitHub: https://github.com/smithk/cidre

---

**最后更新**: 2025-10-26  
**适用版本**: SPRINTSeq项目






