# ImageJ Find Maxima vs 当前 Python 实现对比分析

## ImageJ Find Maxima 算法核心原理

根据 ImageJ 源代码和文档，Find Maxima 算法的关键特性：

### 1. **参数定义**

ImageJ 有两个**独立**的参数：

- **Threshold（阈值）**：最小高度阈值，低于此值的像素将被忽略
- **Noise Tolerance（噪声容忍度）**：**绝对高度差**，只有当一个峰值比其周围区域高出此值时，才被视为有效的最大值

### 2. **算法核心逻辑**

ImageJ 使用**"突起度"（Prominence）**的概念：

1. **洪水淹没算法**：模拟从每个候选峰开始"淹没"的过程
2. **判断标准**：要接受一个最大值，必须满足：
   - 该最大值的高度 > threshold
   - 从该最大值到任何更高的峰之间，必须有一个比当前峰低至少 `noise_tolerance` 的山谷
   - 如果山谷很浅（< noise_tolerance），则该峰被视为"假峰"，被合并到更高的峰

3. **关键点**：
   - Noise Tolerance 是**绝对高度差**，不是相对值
   - 它检查的是**拓扑结构**，而不是简单的空间距离
   - 即使两个峰很近，只要它们之间有足够深的山谷（> noise_tolerance），两个峰都会被保留

### 3. **ImageJ 的具体实现步骤**

根据源代码分析：

```java
// 伪代码
1. 应用 threshold：只考虑 image >= threshold 的像素
2. 对于每个候选最大值：
   a. 计算该峰到周围更高峰之间的最低点（山谷）
   b. 计算 prominence = 峰值高度 - 山谷高度
   c. 如果 prominence >= noise_tolerance，保留该峰
   d. 否则，合并到更高的峰
3. 可选：排除边缘最大值
```

## 当前 Python 实现的问题

### 问题 1: Threshold 和 Noise Tolerance 的关系错误

**当前实现**：
```python
threshold_abs = snr * median
noise_tolerance = threshold_abs * 0.3  # 30% of threshold
```

**问题**：
- ImageJ 中，threshold 和 noise_tolerance 是**独立参数**
- 我们的实现将它们耦合了（noise_tolerance = 0.3 * threshold）
- 这导致当 threshold 很高时，noise_tolerance 也会很高，可能过滤掉真实的峰

**ImageJ 的正确逻辑**：
- threshold 用于**初步过滤**（只考虑高于此值的像素）
- noise_tolerance 用于**判断峰的显著性**（独立于 threshold）

### 问题 2: H-Maxima 的使用可能不正确

**当前实现**：
```python
# 先应用 threshold
image = np.where(image >= threshold_abs, image, 0.0)
# 然后应用 H-Maxima
h_max_img = h_maxima(image, h=noise_tolerance)
```

**潜在问题**：
- H-Maxima 在应用 threshold 后的图像上工作，可能丢失信息
- ImageJ 可能在原始图像上计算 prominence，然后同时应用 threshold 和 noise_tolerance

### 问题 3: 缺少边缘处理

**ImageJ 特性**：
- 可以选择排除边缘最大值（Exclude Edge Maxima）
- 防止边缘效应导致的误检

**当前实现**：
- 没有边缘处理选项

### 问题 4: Local Maxima 检测可能不够精确

**当前实现**：
```python
h_max_img = h_maxima(image, h=noise_tolerance)
maxima_mask = local_maxima(h_max_img)
```

**潜在问题**：
- `local_maxima` 可能使用简单的 3×3 或 5×5 邻域
- ImageJ 可能使用更精确的方法来找到每个峰的精确位置

## 建议的改进方案

### 方案 1: 修正参数关系（**最重要**）

```python
# 正确的逻辑：
# 1. threshold 和 noise_tolerance 应该是独立参数
# 2. threshold 用于初步过滤
# 3. noise_tolerance 用于判断峰的显著性

def find_local_maxima_imagej_style(image, threshold=None, noise_tolerance=None, ...):
    # threshold 和 noise_tolerance 都是独立参数
    # 不应该有 noise_tolerance = threshold * 0.3 这样的关系
```

### 方案 2: 改进 H-Maxima 的使用

```python
# 选项 A: 在原始图像上应用 H-Maxima，然后应用 threshold
h_max_img = h_maxima(image, h=noise_tolerance)
mask = h_max_img >= threshold
maxima_mask = local_maxima(h_max_img)

# 选项 B: 使用更精确的 prominence 计算
# 需要自己实现类似 ImageJ 的洪水淹没算法
```

### 方案 3: 实现真正的 Prominence 计算

```python
def calculate_prominence(image, peak_coords, noise_tolerance):
    """
    计算每个峰的突起度（prominence）
    类似 ImageJ 的算法：检查从峰到更高峰之间的最低点
    """
    # 实现洪水淹没算法
    # 对于每个峰，找到到更高峰之间的最低山谷
    # 如果 prominence >= noise_tolerance，保留该峰
    pass
```

### 方案 4: 使用现有的 Python 库

考虑使用 `pyfindmaxima` 库，它专门设计来复现 ImageJ 的 Find Maxima 功能。

## 关键差异总结

| 特性 | ImageJ Find Maxima | 当前 Python 实现 |
|------|-------------------|------------------|
| **Threshold** | 独立参数，用于初步过滤 | `snr * median`，可能过高 |
| **Noise Tolerance** | 独立参数，绝对高度差 | `threshold * 0.3`，耦合了 |
| **算法** | 洪水淹没 + Prominence 计算 | H-Maxima + Local Maxima |
| **边缘处理** | 可选排除边缘最大值 | 无 |
| **参数独立性** | threshold 和 noise_tolerance 独立 | 耦合了 |

## 立即行动建议

1. **解耦 threshold 和 noise_tolerance**
   - 让它们成为独立参数
   - 不要使用 `noise_tolerance = threshold * 0.3`

2. **调整参数传递**
   - 在调用 `find_local_maxima_imagej_style` 时，分别传入 threshold 和 noise_tolerance
   - 不要从 threshold 计算 noise_tolerance

3. **测试不同的参数组合**
   - 固定 threshold，改变 noise_tolerance
   - 固定 noise_tolerance，改变 threshold
   - 找到与 ImageJ 结果最匹配的组合

4. **考虑使用 pyfindmaxima**
   - 如果时间紧迫，可以考虑使用专门复现 ImageJ 算法的库

