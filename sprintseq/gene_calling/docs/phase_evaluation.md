在只有 10 个 Cycle 且有 Codebook（码本）的情况下，估计失相率（Phasing, $p$）和预失相率（Pre-phasing, $q$）其实比无参测序更简单。因为你知道答案“应该”是什么。

这里有三种不同难度的策略，按推荐程度排序：

### 策略一：基于“ON-OFF”跳变的直接观测法（最直观，手动 QC）

这是最符合直觉的方法。你需要寻找一种特定的信号模式：**上一轮极亮，这一轮本该全暗**。

1.  **筛选 Spots**：

      * 在码本中找到一个特定的 Barcode 模式，例如：**Cycle 1 是强信号（比如 Cy3），Cycle 2 是暗信号（无荧光或另一色）**。
      * 或者找 **Cycle 1 (Cy3) -\> Cycle 2 (Cy5)** 的跳变。

2.  **观察“拖尾” (Lagging/Phasing)**：

      * 如果你在 Cycle 2 的 Cy3 通道（本该是黑的）检测到了信号。
      * **计算**：$p \approx \frac{I_{cyc2\_Cy3}}{I_{cyc1\_Cy3}}$
      * *注意*：这个比值通常会包含一点背景噪音，所以要减去底噪。

3.  **观察“抢跑” (Leading/Pre-phasing)**：

      * 反过来，找 **Cycle 1 (暗) -\> Cycle 2 (亮)** 的 Spot。
      * 如果你在 Cycle 1 就看到了本该属于 Cycle 2 的颜色。
      * **计算**：$q \approx \frac{I_{cyc1\_Signal}}{I_{cyc2\_Signal}}$

**缺点**：容易受信号衰减（Decay）影响。如果 Cycle 2 本身就因为衰减变弱了，算出来的 $p$ 会偏小。建议先做完衰减矫正（Envelope Fitting）再算这个。

-----

### 策略二：最大纯度搜索法 (Grid Search for Max Chastity) —— **推荐代码实现**

这是 Illumina RTA 的核心逻辑简化版。
**逻辑**：正确的 $p$ 和 $q$ 值，应该能让矫正后的信号“最纯净”。
**纯净（Chastity）的定义**：一个 Spot 在某个 Cycle，要么很亮（1），要么很暗（0）。如果矫正后变成了 0.3 或 0.6（模棱两可），说明矫正参数不对。

#### 算法流程：

1.  设定一个搜索范围，例如 $p \in [0, 0.05]$，步长 0.005。
2.  对于每一个 $p$ 值，构建矫正矩阵 $M^{-1}$ 并应用到一小部分数据上（比如 1000 个 Spots）。
3.  计算矫正后数据的 **Chastity Score**（纯度分）。
4.  得分最高的那个 $p$，就是最佳估计值。

#### Python 实现代码

我为你写了一个自动寻找最佳 $p$ 值的脚本。它利用\*\*“信号越陡峭越好”\*\*的原理。

```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_chastity(signal_matrix):
    """
    计算信号纯度 (Chastity)。
    Chastity = Max_Int / (Max_Int + Second_Max_Int)
    但在 ISS 只有 10 cycles 单色/双色时，我们可以简化为：
    'Contrast': 信号越接近 0 或 1 (归一化后) 越好。
    或者简单的：L1/L2 norm ratio (稀疏度)。
    
    这里使用一个简单的方差/峰度逻辑：
    如果矫正得好，信号应该两极分化（背景很低，信号很高）。
    如果矫正过度或不足，会有很多“残影”停留在中间值。
    """
    # 简单的 Metric: 变异系数 (CV) 或者 信号值的平方和 (L2 Norm) 
    # 在归一化之后，L2 Norm 越大通常意味着峰越尖锐，背景越干净
    return np.sum(signal_matrix ** 2)

def estimate_phasing_grid_search(signal_matrix, step=0.005, max_p=0.05):
    """
    通过网格搜索找到最佳的 Phasing Rate
    signal_matrix: 已经做过衰减矫正(Decay Corrected)的数据 [n_spots, n_cycles, n_channels]
    """
    best_p = 0
    best_score = -np.inf
    scores = []
    p_values = np.arange(0, max_p + step/2, step)
    
    # 预处理：先简单归一化到 0-1 之间，避免量纲影响
    norm_sig = signal_matrix / (np.max(signal_matrix) + 1e-6)
    
    for p in p_values:
        # 1. 构建简化矩阵 (只考虑 Phasing p, 忽略 q 以简化)
        # M = [[1-p, 0], [p, 1-p]...]
        # 快速构建逆矩阵逻辑: I_true[t] = (I_obs[t] - p * I_true[t-1]) / (1-p)
        
        corrected = np.zeros_like(norm_sig)
        n_cycles = norm_sig.shape[1]
        
        # 递归去卷积 (比矩阵求逆快)
        # s = 1 - p
        # I_corr[t] = (I_obs[t] - p * I_corr[t-1]) / s
        
        s = 1 - p
        for t in range(n_cycles):
            if t == 0:
                corrected[:, t] = norm_sig[:, t] / s # 假设 cycle 0 没有前一轮干扰
            else:
                # 减去上一轮的“拖尾”
                corrected[:, t] = (norm_sig[:, t] - p * corrected[:, t-1]) / s
        
        # 2. 评分
        # 我们希望 corrected 里的值尽可能接近 0 (背景) 或 1 (信号)
        # 也就是“非灰度”的程度。
        # 惩罚中间值: score = sum( (x - 0.5)^2 )
        score = np.sum((corrected - 0.5)**2)
        
        scores.append(score)
        if score > best_score:
            best_score = score
            best_p = p
            
    # 绘图查看搜索结果
    plt.figure(figsize=(6, 4))
    plt.plot(p_values, scores, 'o-')
    plt.title(f'Best Phasing Rate: {best_p*100:.1f}%')
    plt.xlabel('Phasing Rate (p)')
    plt.ylabel('Signal Contrast Score')
    plt.grid(True)
    plt.show()
    
    return best_p

# 使用方法：
# best_p = estimate_phasing_grid_search(decay_corrected_data)
```

-----

### 策略三：利用码本残差 (Residual Minimization with Codebook) —— **最精准**

如果你已经解完码（Base Calling），或者有一个初步的 Gene 列表，你可以反推。

1.  **假设**：你的样品里只有 Gene A, B, C。
2.  **模拟**：用 $p=0.01, 0.02...$ 生成 Gene A, B, C 的“理论模糊指纹”。
3.  **比对**：看哪个 $p$ 生成的指纹，跟你的实际显微镜图像最像（残差最小）。

**公式**：
$$\text{Loss}(p) = \sum || \mathbf{I}_{obs} - \mathbf{M}(p) \cdot \mathbf{Code}_{expected} ||^2$$

这种方法通常用于空间转录组（如 Starfish 流程），因为它可以同时优化 $p$（失相）和 $k$（荧光串扰矩阵）。

### 总结建议

1.  **先做衰减矫正**：务必先用我上一条回答的 `Correct Decay` 把信号拉平。否则衰减会被误判为 Phasing 的反向效果。
2.  **先看图（策略一）**：找一个 `Bright -> Dark` 的 Spot，肉眼估算一下。如果是 1000 掉到 50，那 $p \approx 5\%$。如果掉到 10，那 $p \approx 1\%$。
3.  **再跑代码（策略二）**：用上面的 Grid Search 脚本，精细化确定是 1.5% 还是 2.0%。通常 ISS 的 Phasing 率在 **1% - 4%** 之间是常见的。如果算出 \>10%，通常是生化反应出了大问题（如洗涤不彻底）。