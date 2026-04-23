仔细看了你发的 QC 图表，情况非常清楚。这确实不是简单的“化学衰减”，而是一个典型的**由碱基分布不均（Base Composition Bias）主导的信号波动**，叠加了轻微的衰减。

### 诊断：为什么信号是“锯齿状”的？

正如你所说，Cyc 2、3、5 的峰值是因为这些位置上，你要测的那个 Barcode 序列正好大概率是 Cy3 或 Cy5 对应的碱基，导致点亮的 Spot 特别多，或者说这两个通道在这个 Cycle 信号极强。

  * **现象**：
      * **Cycle 1**：亮度较低（可能是因为 Cy3/Cy5 对应的碱基少，或者很多点是不发光的 Dark bases）。
      * **Cycle 2, 3, 5**：亮度猛增（说明大部分点都在发光，且是 Cy3/Cy5 通道）。
      * **Cycle 8, 9, 10**：亮度回落并趋于平稳（可能是真实的衰减，也可能是碱基分布变了）。
      * **Cy3 vs Cy5（最下图）**：两者趋势高度一致，说明这不是单通道问题，而是整个群体在这些 Cycle 都很亮。

### 这种情况下，传统的“衰减矫正”会失效！

如果你直接用我上一条回答里的 `mean_intensity` 来做归一化（除以均值），你会犯大错：

  * 你会把 **Cycle 2 的真实强信号** 强行压低。
  * 你会把 **Cycle 1 的真实弱信号** 强行拉高（连带把背景噪音也拉爆了）。

### 正确的矫正策略：分位数归一化 + 稳健衰减估计

针对这种“锯齿状”数据，你需要把 **“该不该亮”** 和 **“亮得够不够”** 分开处理。

#### 1\. 策略 A：Quantile Normalization (分位数归一化) —— **最推荐**

不要看均值（Mean），看**分布的形状**。
即使 Cycle 1 亮的点少，亮的那一小部分点的**亮度值（Intensity Value）应该和 Cycle 2 亮点的亮度值是在一个水平线上**的（假设没有衰减）。

  * **原理**：假设每一轮最亮的那前 5% 的点（Top 5%），它们一定是真的信号，且应当具有相同的物理亮度。
  * **操作**：
    1.  对每个 Cycle，计算所有 Spot 亮度的 **95th Percentile (P95)** 或 **90th Percentile (P90)**。
          * *为什么不用 Max？* 容易受坏点/灰尘影响。
          * *为什么不用 Mean？* 受“亮的点有多少”影响极大（就是你现在的图）。
          * *P95 的逻辑*：只要有 5% 的点是真实发光的，P95 就能捕捉到“满血”的信号强度。
    2.  画出 `[P95_cyc1, P95_cyc2, ..., P95_cyc10]` 的曲线。
    3.  **这条曲线才是你真实的化学衰减曲线（Decay Curve）**。你会发现它应该比你现在的 Mean 曲线平滑得多。
    4.  计算归一化因子：`Factor_i = P95_cyc1 / P95_cyci`。
    5.  用这个因子去乘该 Cycle 所有 Spot 的原始信号。

#### 2\. 策略 B：基于码本（Codebook）的预期矫正 (Expectation-Based)

既然你知道 Cyc 2, 3, 5 应该强，说明你手头有码本（Codebook）。

  * **原理**：利用已知信息。如果 Barcode 设计决定了 Cycle 2 有 80% 的点是 Cy3/Cy5，而 Cycle 1 只有 20%。
  * **操作**：
    1.  统计码本中每个 Cycle 预期的 Cy3/Cy5 丰度（比如 `Exp_Ratio = [0.2, 0.8, 0.8, 0.5, 0.8...]`）。
    2.  用实际观测的 Mean Intensity 除以这个 `Exp_Ratio`。
    3.  理论上，除完之后得到的值，就只剩下化学衰减了。
    4.  再对这个残余的衰减进行拟合和补偿。

### 你的 Action Plan (Python 代码调整)

鉴于你的图表显示 Cyc 10 的信号依然不错（Mean Ratio 还在 0.8 左右，没有掉到 20% 以下），说明**化学衰减其实并不严重**。

我建议你采用 **策略 A（分位数法）**，这最简单且稳健。

以下是修改后的代码逻辑：

```python
import numpy as np

def correct_decay_robust(signal_matrix, percentile=95):
    """
    使用分位数(Percentile)来估算和矫正衰减，
    抵抗由碱基分布不均(Base Composition Bias)引起的信号波动。
    """
    n_spots, n_cycles, n_channels = signal_matrix.shape
    
    # 1. 计算每个 Cycle 的“最大有效亮度” (使用 P95 或 P90)
    # 我们把所有通道合起来看，或者只看当前最亮的通道
    # 这里简单起见，取每个 Spot 在当前 Cycle 最亮的那个通道的值作为代表
    max_channel_intensity = np.max(signal_matrix, axis=2) # shape: (n_spots, n_cycles)
    
    # 计算分位数向量 (这就是稳健的衰减曲线)
    # 即使某一轮只有 10% 的点亮，P95 依然能抓到亮点的强度，而不会被暗点拉低
    robust_decay_curve = np.percentile(max_channel_intensity, percentile, axis=0)
    
    # 2. 计算归一化因子 (以 Cycle 1 或 Cycle 2 为基准)
    # 如果 Cycle 1 本身就很弱(不是因为没点亮，而是因为化学反应没开始好)，可以用 Max(Curve) 做基准
    baseline = np.max(robust_decay_curve) 
    # 或者 baseline = robust_decay_curve[0]
    
    scale_factors = baseline / robust_decay_curve
    
    # 3. 应用矫正
    corrected_matrix = np.zeros_like(signal_matrix)
    for t in range(n_cycles):
        corrected_matrix[:, t, :] = signal_matrix[:, t, :] * scale_factors[t]
        
    return corrected_matrix, robust_decay_curve

# 使用:
# corrected_sig, decay_curve = correct_decay_robust(your_data, percentile=90)
```

### 总结建议

1.  **不要担心锯齿**：你的图里 Cyc 2,3,5 高是非常正常的物理现象，说明你的杂交/连接反应效率很高。
2.  **改用 P90/P95 归一化**：用我上面的代码，算出一条新的 Decay 曲线。那条曲线大概率是**平滑下降**的（或者只有轻微波动）。
3.  **先归一化，再解卷积**：
      * Step 1: 用 P95 法拉平不同 Cycle 的亮度差异。
      * Step 2: 此时各 Cycle 能量水平一致了，再跑矩阵求逆（解卷积）去除 Phasing。
      * Step 3: Base Calling。

你可以先试着画一下 `np.percentile(data, 95, axis=0)` 的曲线，看看是不是比 Mean 曲线平滑很多？如果是，那就对了。