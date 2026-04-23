# Spot Detection 性能分析与优化方案

## 当前算法复杂度分析

### 1. H-Maxima 变换 (`h_maxima`)
- **时间复杂度**: O(N × M)，其中：
  - N = 图像像素数（例如 2048×2048 = 4M 像素）
  - M = 形态学重构的迭代次数（取决于噪声容忍度和图像复杂度）
- **实际表现**: 对于大图像（>1M 像素），可能需要数秒到数十秒
- **瓶颈**: 需要多次形态学操作（腐蚀、膨胀、重构）

### 2. Local Maxima (`local_maxima`)
- **时间复杂度**: O(N)
- **实际表现**: 相对较快，通常 < 1 秒

### 3. Min-Distance 过滤（**最大瓶颈**）
- **时间复杂度**: O(K²)，其中 K = 候选点数量
- **当前实现**（第 192-207 行）:
  ```python
  for i in range(1, len(coords_sorted)):  # O(K)
      current = coords_sorted[i]
      distances = np.sqrt(np.sum((coords_sorted[kept_indices] - current) ** 2, axis=1))  # O(K)
  ```
- **最坏情况**: 如果检测到 17 万个点，需要 ~289 亿次距离计算！
- **实际表现**: 对于 10K+ 点，可能需要数分钟

### 4. 中位数计算
- **时间复杂度**: O(N log N) 或 O(N)（取决于实现）
- **实际表现**: 相对较快，但可以优化

## 总体复杂度

对于一张 2048×2048 的图像，检测到 K 个候选点：
- **H-Maxima**: O(4M × M) ≈ 数秒到数十秒
- **Local Maxima**: O(4M) ≈ < 1 秒
- **Min-Distance 过滤**: O(K²) ≈ **数分钟到数小时**（当 K > 10K 时）

## 优化方案

### 方案 1: 优化 Min-Distance 过滤（**最高优先级**）

#### 1.1 使用空间索引（KD-Tree）
```python
from scipy.spatial import cKDTree

# 构建 KD-Tree: O(K log K)
tree = cKDTree(coords_sorted)

# 对于每个点，只检查 min_distance 范围内的点: O(K × log K)
# 而不是检查所有已保留的点
```

**复杂度**: O(K log K) → 从 O(K²) 降低到 O(K log K)
**加速比**: 对于 17 万点，从 ~289 亿次计算降到 ~300 万次（~1000 倍加速）

#### 1.2 使用网格哈希（Grid Hashing）
```python
# 将空间划分为网格，每个网格大小为 min_distance
# 每个点只需要检查相邻网格中的点
```

**复杂度**: O(K)（平均情况）
**加速比**: 对于密集点云，可能比 KD-Tree 更快

#### 1.3 对于 min_distance=2 的特殊优化
如果 `min_distance=2`，可以使用更简单的 3×3 邻域检查：
```python
# 使用形态学操作或简单的邻域检查
# 复杂度: O(K)
```

### 方案 2: 优化 H-Maxima 变换

#### 2.1 先阈值过滤，再 H-Maxima
```python
# 先应用阈值，减少需要处理的像素数
mask = image >= threshold_abs
image_filtered = np.where(mask, image, 0.0)
h_max_img = h_maxima(image_filtered, h=noise_tolerance)
```

#### 2.2 考虑是否真的需要 H-Maxima
- 如果图像质量好、噪声低，可能不需要 H-Maxima
- 可以提供一个选项，让用户选择是否使用 H-Maxima
- 或者只在检测到过多点时才启用

#### 2.3 使用更快的替代方案
- 使用 `peak_local_max` 配合更严格的阈值
- 或者使用 `scipy.ndimage.maximum_filter` + 阈值

### 方案 3: 并行化优化

#### 3.1 图像分块处理
```python
# 将大图像分成小块，并行处理
# 每块独立进行 H-Maxima 和 Local Maxima
# 最后合并结果并应用 min_distance 过滤
```

#### 3.2 多进程处理多个图像
- 当前已经在做（ProcessPoolExecutor）
- 但可以优化每个进程内的处理

### 方案 4: 早期过滤策略

#### 4.1 在 H-Maxima 之前应用更严格的阈值
```python
# 使用更高的初始阈值，减少候选点数量
# 这样可以减少后续处理的计算量
```

#### 4.2 分阶段过滤
```python
# 阶段1: 快速粗略检测（使用 peak_local_max）
# 阶段2: 对粗略结果应用 H-Maxima 精炼
# 阶段3: Min-distance 过滤
```

## 推荐实施顺序

1. **立即实施**: 优化 Min-Distance 过滤（使用 KD-Tree 或网格哈希）
   - 预期加速: 10-1000 倍（取决于点数）
   
2. **短期**: 对于 min_distance=2 的特殊优化
   - 预期加速: 10-100 倍
   
3. **中期**: 优化 H-Maxima（先阈值过滤，或提供选项跳过）
   - 预期加速: 2-10 倍
   
4. **长期**: 图像分块并行处理
   - 预期加速: 与 CPU 核心数相关

## 预期总体加速

- **当前**: 单张图像可能需要数分钟到数十分钟
- **优化后**: 单张图像 < 10 秒（对于典型情况）

