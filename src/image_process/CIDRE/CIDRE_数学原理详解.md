# CIDRE 数学原理详解

## 目录
1. [问题建模](#1-问题建模)
2. [数学模型](#2-数学模型)
3. [优化目标函数](#3-优化目标函数)
4. [优化算法](#4-优化算法)
5. [实现细节](#5-实现细节)

---

## 1. 问题建模

### 1.1 照明不均问题

在光学显微镜成像中，由于光源分布不均、光学系统的渐晕效应等原因，图像中心通常比边缘更亮。我们观测到的图像强度 \(I\) 可以建模为：

$$I(x, y) = v(x, y) \cdot S(x, y) + z(x, y) + \eta(x, y)$$

其中：
- \(S(x, y)\): 真实的样本信号（我们想要恢复的）
- \(v(x, y)\): 照明增益表面（illumination gain surface，描述相对光强分布）
- \(z(x, y)\): 零光表面（zero-light surface，背景/暗电流）
- \(\eta(x, y)\): 噪声

### 1.2 问题的关键挑战

这是一个**欠定问题**（underdetermined problem）：
- 单张图像无法区分"样本暗"还是"照明弱"
- 需要利用**多张图像**的统计信息
- 假设：不同图像中，样本分布不同，但照明模式相同

---

## 2. 数学模型

### 2.1 基本假设

对于同一位置 \((x, y)\) 在 \(N\) 张图像中的观测：

$$q_i(x, y) = v(x, y) \cdot Q_i + b(x, y) \quad i = 1, 2, ..., N$$

其中：
- \(q_i(x, y)\): 第 \(i\) 张图像在位置 \((x, y)\) 的观测强度
- \(Q_i\): 第 \(i\) 个强度水平（底层真实强度的代表）
- \(v(x, y)\): 照明增益（斜率）
- \(b(x, y)\): 截距项（与零光表面相关）

这是一个**线性回归模型**：在 \(Q-q\) 空间中，每个位置的观测值应该落在一条直线上。

### 2.2 Pivot空间变换

为了提高数值稳定性，CIDRE将数据转换到"pivot空间"：

$$Q' = Q - \text{median}(Q)$$
$$q'(x, y) = q(x, y) - \text{median}(q(x, y))$$

这样将数据中心化，避免了大数值导致的数值不稳定。

### 2.3 零光表面

零光表面 \(z(x, y)\) 通过以下关系与 \(b\) 和 \(v\) 联系：

$$z(x, y) = b(x, y) + z_x \cdot v(x, y)$$

其中 \(z_x\) 是一个标量参数，使得零光表面随照明增益变化。

---

## 3. 优化目标函数

CIDRE通过最小化以下能量函数来求解 \(v(x, y)\)、\(b(x, y)\)、\(z_x\) 和 \(z_y\)：

$$E_{\text{total}} = E_{\text{fit}} + \lambda_v E_{\text{vreg}} + \lambda_z E_{\text{zero}} + \lambda_{\text{barr}} E_{\text{barr}}$$

### 3.1 拟合项（Fitting Term）

$$E_{\text{fit}} = \sum_{x,y} \sum_{i=1}^{N} \rho\left(q_i(x,y) - v(x,y) Q_i - b(x,y)\right)$$

其中 \(\rho(\cdot)\) 是M-estimator（稳健估计量）：

**阶段1：最小二乘（Least Squares）**
$$\rho_{\text{LS}}(r) = r^2$$

**阶段2：Cauchy函数（鲁棒回归）**
$$\rho_{\text{Cauchy}}(r) = \frac{w^2}{2} \log\left(1 + \frac{r^2}{w^2}\right)$$

Cauchy函数的特点：
- 对小残差：近似二次函数 \(\approx r^2/2\)
- 对大残差：增长缓慢 \(\approx w^2 \log|r|\)
- 对异常值不敏感（鲁棒性强）

**拟合项的梯度：**

对于Cauchy估计器：
$$\frac{\partial E_{\text{fit}}}{\partial v(x,y)} = \sum_i \frac{Q_i \cdot r_i}{1 + r_i^2/w^2}$$

$$\frac{\partial E_{\text{fit}}}{\partial b(x,y)} = \sum_i \frac{r_i}{1 + r_i^2/w^2}$$

其中 \(r_i = v(x,y) Q_i + b(x,y) - q_i(x,y)\)

### 3.2 空间正则化项（Spatial Regularization）

为了使照明表面平滑，引入正则化项：

$$E_{\text{vreg}} = \sum_{\sigma \in \Sigma} \|\mathcal{L}_{\sigma} * v\|^2$$

其中：
- \(\mathcal{L}_{\sigma}\) 是尺度为 \(\sigma\) 的**拉普拉斯高斯算子**（LoG, Laplacian of Gaussian）
- \(*\) 表示卷积
- \(\Sigma = \{2^{-1}, 2^0, 2^1, ..., 2^k\}\) 是多个尺度

**LoG滤波器定义：**

$$\mathcal{L}_{\sigma}(x, y) = -\frac{1}{\pi\sigma^4}\left(1 - \frac{x^2 + y^2}{2\sigma^2}\right) e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

归一化后：
$$h_{\sigma} = \sigma^2 \cdot \mathcal{L}_{\sigma}$$

**物理意义：**
- LoG算子检测图像的二阶导数（曲率）
- 多尺度保证在不同空间频率上都平滑
- 惩罚 \(v\) 的快速变化，保持照明表面平滑

**正则化项的梯度：**

$$\frac{\partial E_{\text{vreg}}}{\partial v} = \sum_{\sigma} h_{\sigma} * (2 \cdot (h_{\sigma} * v))$$

这是两次LoG卷积的组合。

### 3.3 零光约束项（Zero-light Term）

确保零光表面的一致性：

$$E_{\text{zero}} = \sum_{x,y} \left(v(x,y) \cdot p_x + b(x,y) - p_y(x,y)\right)^2$$

其中：
- \(p_x = z_x - \text{PivotShift}_X\)（pivot空间中的零光x坐标）
- \(p_y(x,y) = z_y(x,y) - \text{PivotShift}_Y(x,y)\)（pivot空间中的零光y坐标）

**物理意义：**
这一项强制所有位置的回归直线都通过或接近同一个零光点 \((z_x, z_y)\)。

**梯度：**

$$\frac{\partial E_{\text{zero}}}{\partial v(x,y)} = 2p_x(b(x,y) + v(x,y) p_x - p_y(x,y))$$

$$\frac{\partial E_{\text{zero}}}{\partial b(x,y)} = 2(b(x,y) + v(x,y) p_x - p_y(x,y))$$

$$\frac{\partial E_{\text{zero}}}{\partial z_x} = 2\sum_{x,y} v(x,y)(b(x,y) + v(x,y) p_x - p_y(x,y))$$

$$\frac{\partial E_{\text{zero}}}{\partial z_y} = -2\sum_{x,y} (b(x,y) + v(x,y) p_x - p_y(x,y))$$

### 3.4 障碍项（Barrier Term）

限制 \(z_x\) 和 \(z_y\) 在合理范围内：

$$E_{\text{barr}} = \phi(z_x; z_{\min}, z_{\max}) + \phi(z_y; z_{\min}, z_{\max})$$

其中障碍函数 \(\phi(x; x_{\min}, x_{\max})\) 定义为：

$$
\phi(x) = \begin{cases}
\left(\frac{x - x_2}{x_2 - x_1}\right)^2 & \text{if } x < x_2 \\
0 & \text{if } x_2 \leq x \leq x_3 \\
\left(\frac{x - x_3}{x_4 - x_3}\right)^2 & \text{if } x > x_3
\end{cases}
$$

其中 \(x_1 = x_{\min}\), \(x_2 = x_{\min} + w\), \(x_3 = x_{\max} - w\), \(x_4 = x_{\max}\)，\(w\) 是过渡宽度。

**物理意义：**
这是一个"势阱"，防止零光参数取不合理的值。

---

## 4. 优化算法

### 4.1 两阶段优化策略

CIDRE采用**两阶段优化**：

**阶段1：最小二乘初始化**
- 使用 \(\rho_{\text{LS}}\)（最小二乘）
- 仅优化拟合项：\(E = E_{\text{fit}}\)
- 快速获得初始解

**阶段2：鲁棒正则化优化**
- 使用 \(\rho_{\text{Cauchy}}\)（Cauchy鲁棒估计）
- 优化完整目标：\(E = E_{\text{fit}} + \lambda_v E_{\text{vreg}} + \lambda_z E_{\text{zero}} + \lambda_{\text{barr}} E_{\text{barr}}\)
- 使用阶段1的结果作为初始值
- Cauchy宽度 \(w\) 设为阶段1的均方误差

### 4.2 L-BFGS优化器

CIDRE使用**L-BFGS**（Limited-memory BFGS）算法：

**L-BFGS算法简介：**

BFGS是拟牛顿方法，通过迭代近似Hessian矩阵的逆：

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \mathbf{H}_k \nabla E(\mathbf{x}_k)$$

其中：
- \(\mathbf{H}_k\): Hessian逆的近似
- \(\alpha_k\): 线搜索确定的步长
- \(\nabla E\): 目标函数的梯度

L-BFGS只存储最近 \(m\) 次迭代的信息（默认 \(m = 100\)），大大节省内存。

**更新公式：**

$$\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$$
$$\mathbf{y}_k = \nabla E_{k+1} - \nabla E_k$$

$$\mathbf{H}_{k+1} = \mathbf{V}_k^T \mathbf{H}_k \mathbf{V}_k + \rho_k \mathbf{s}_k \mathbf{s}_k^T$$

其中 \(\mathbf{V}_k = \mathbf{I} - \rho_k \mathbf{y}_k \mathbf{s}_k^T\)，\(\rho_k = 1/(\mathbf{y}_k^T \mathbf{s}_k)\)

**为什么使用L-BFGS：**
- 参数数量巨大（每个像素有 \(v\) 和 \(b\)，例如 512×512 图像有 524,288 个变量）
- L-BFGS内存效率高（\(O(nm)\) vs \(O(n^2)\)）
- 收敛速度快（超线性收敛）
- 不需要显式计算Hessian矩阵

### 4.3 优化变量

优化变量向量：

$$\mathbf{x} = [v_1, v_2, ..., v_{R \times C}, b_1, b_2, ..., b_{R \times C}, z_x, z_y]^T$$

其中 \(R \times C\) 是图像尺寸（实际在降采样后的尺寸上优化，如 40×40）。

---

## 5. 实现细节

### 5.1 数据预处理

**步骤1：图像排序**
对每个像素位置，将所有图像的该位置强度排序：

$$S(x, y, :) = \text{sort}(\{I_1(x,y), I_2(x,y), ..., I_N(x,y)\})$$

**步骤2：降维压缩**
将 \(N\) 张图像压缩到 200 个分位数（quantiles）：
- 如果 \(N \leq 200\)：保持不变
- 如果 \(N > 200\)：将排序后的强度分成 200 组，每组取平均

这样将数据量标准化，便于参数设置。

**步骤3：尺度空间重采样**（Scale-space Resampling）
当图像数量不足时，使用多尺度方法增强鲁棒性：
- 创建图像金字塔：\(\{S, S_{1/2}, S_{1/4}, ...\}\)
- 将不同尺度的数据拼接，增加有效数据量
- 根据熵（entropy）判断是否需要

### 5.2 Q的估计（鲁棒均值）

从中心分位数选取25%的像素：

$$M = \left\{(x, y) : \text{rank}(\bar{I}(x, y)) \in \left[\frac{3R \times C}{8}, \frac{5R \times C}{8}\right]\right\}$$

其中 \(\bar{I}(x, y) = \text{mean}_i(I_i(x, y))\)

然后对这些位置求平均：

$$Q_k = \frac{1}{|M|} \sum_{(x,y) \in M} S(x, y, k)$$

**为什么选择中心分位数：**
- 避免极暗区域（可能是背景，信息少）
- 避免极亮区域（可能饱和或异常值）
- 鲁棒性强

### 5.3 参数的自适应设置

**lambda_v的自适应：**

$$\lambda_v = \begin{cases}
9.5 + \frac{6 - 9.5}{200} \cdot N & \text{if } N < 200 \\
6.0 & \text{if } N \geq 200
\end{cases}$$

图像少时增加正则化强度。

**Cauchy宽度的自适应：**

$$w = \text{MSE}_{\text{LS}} = \sqrt{\frac{1}{N} \sum_i (q_i - v Q_i - b)^2}$$

使用最小二乘拟合的均方误差作为Cauchy函数的宽度。

### 5.4 图像校正

优化完成后，对原始图像进行校正：

**模式0（保持零光）：**
$$I_{\text{corrected}} = I_{\text{raw}} - z + \bar{z}$$
$$I_{\text{corrected}} = \frac{I_{\text{corrected}} - \bar{I}_{\text{corrected}}}{\bar{v}/v} + \bar{I}_{\text{corrected}}$$

**模式1（动态范围校正）：**
$$I_{\text{corrected}} = (I_{\text{raw}} - z) \cdot \frac{1}{v} \cdot \frac{\bar{v}}{\bar{I}_{\text{raw}} - \bar{z}} + \bar{z}$$

**模式2（直接校正）：** ⭐推荐
$$I_{\text{corrected}} = \frac{I_{\text{raw}} - z}{v}$$

其中 \(\bar{\cdot}\) 表示平均值。

---

## 6. 数学直觉与几何解释

### 6.1 Q-q空间的几何意义

对于每个像素位置 \((x, y)\)：
- 横坐标：估计的底层强度 \(Q\)
- 纵坐标：观测到的强度 \(q\)
- 期望：数据点落在直线 \(q = vQ + b\) 上

**不同位置的差异：**
- 照明强的位置：斜率 \(v\) 大（陡峭）
- 照明弱的位置：斜率 \(v\) 小（平缓）
- 所有直线应该相交于零光点 \((z_x, z_y)\)

### 6.2 正则化的作用

**空间正则化 \(E_{\text{vreg}}\)：**
- 相邻像素的 \(v\) 值应该相似
- 照明变化是平滑的，不会突变
- 类比：拉伸一张橡胶膜，能量最小时表面最平

**零光正则化 \(E_{\text{zero}}\)：**
- 所有回归直线应汇聚到一个公共点
- 物理上：零光（背景）在空间上变化小
- 约束了模型的自由度，提高稳定性

### 6.3 为什么需要两阶段优化

**阶段1（最小二乘）：**
- 快速找到一个"大致正确"的解
- 为阶段2提供好的初始值
- 避免陷入局部最优

**阶段2（Cauchy + 正则化）：**
- 处理异常值（如灰尘、死像素）
- 加入平滑约束
- 精细调整得到最终解

---

## 7. 数学公式总结

### 完整的优化问题

$$
\min_{v, b, z_x, z_y} \left\{
\sum_{x,y} \sum_{i} \frac{w^2}{2} \log\left(1 + \frac{(q_i - vQ_i - b)^2}{w^2}\right)
+ \lambda_v \sum_{\sigma} \|h_{\sigma} * v\|^2
+ \lambda_z \sum_{x,y} (vp_x + b - p_y)^2
+ \lambda_{\text{barr}} (\phi(z_x) + \phi(z_y))
\right\}
$$

受约束于：
- \(v(x, y) > 0\) （照明增益为正）
- \(z_{\min} \leq z_x, z_y \leq z_{\max}\)

### 关键参数的物理意义

| 参数 | 物理意义 | 典型值 |
|------|----------|--------|
| \(\lambda_v\) | 照明平滑程度（空间正则化） | \(10^6 - 10^7\) |
| \(\lambda_z\) | 零光一致性（背景均匀性） | \(10^{0.5} \approx 3\) |
| \(w\) | 异常值容忍度（Cauchy宽度） | MSE of LS fit |
| \(Q\) | 底层强度分布 | 从数据估计 |

---

## 8. 算法复杂度分析

### 时间复杂度

**预处理：**
- 图像排序：\(O(N \log N \cdot R \cdot C)\)
- 降维压缩：\(O(N \cdot R \cdot C)\)

**优化：**
- 每次迭代：
  - 拟合项计算：\(O(R \cdot C \cdot Z)\)，其中 \(Z = 200\)
  - LoG卷积：\(O(R \cdot C \cdot k \cdot m^2)\)，\(k\) 个尺度，\(m\) 是核大小
  - 总计：\(O(R \cdot C \cdot Z)\)
- 迭代次数：\(T \approx 500\)
- 总时间：\(O(T \cdot R \cdot C \cdot Z)\)

**实际运行时间：**
对于 512×512 图像，100张图，约 2-5 分钟（标准PC）。

### 空间复杂度

- 图像栈：\(O(R \cdot C \cdot N)\)
- 优化变量：\(O(2 \cdot R_s \cdot C_s)\)，工作尺寸通常 40×40
- L-BFGS存储：\(O(m \cdot n)\)，\(m = 100\) 次迭代，\(n\) 是变量数

---

## 9. 理论保证与限制

### 9.1 模型假设

1. **线性模型假设**：照明效应是乘性的
2. **照明不变性**：所有图像共享相同的照明模式
3. **样本多样性**：不同图像的样本分布足够多样
4. **平滑照明**：照明变化是平滑的

### 9.2 失效情况

- **图像数量太少**（\(N < 20\)）：欠定，解不唯一
- **样本单一**：所有图像内容相似，无法区分样本和照明
- **照明非平滑**：如激光斑点，违反平滑假设
- **非线性效应**：如光漂白、饱和

### 9.3 收敛性

L-BFGS保证：
- 在凸问题上：全局收敛
- 在非凸问题上：收敛到局部最优

CIDRE的目标函数是**非凸的**，但通过两阶段策略和良好初始化，实践中通常得到满意的解。

---

## 10. 与其他方法的比较

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **BaSiC** | 无监督低秩+稀疏分解 | 无需多张图像 | 计算慢，参数多 |
| **CIDRE** | 鲁棒回归+正则化 | 快速，参数少，理论清晰 | 需要多张图像 |
| **Flat-field** | 使用标准平场图像 | 简单直接 | 需要专门采集平场 |
| **Retrospective** | 多项式拟合 | 单张图像即可 | 对复杂照明效果差 |

---

## 11. 实际应用建议

### 11.1 何时使用CIDRE

✅ **适合：**
- 有足够数量的图像（\(N \geq 30\)）
- 图像内容多样（不是所有图都拍同一个区域）
- 照明变化平滑
- 需要快速自动化处理

❌ **不适合：**
- 图像很少（\(< 20\) 张）
- 所有图像几乎相同
- 照明有突变（如激光斑点）
- 样本本身有大面积均匀区域且强度接近照明梯度

### 11.2 参数调优策略

1. **从默认参数开始**：\(\lambda_v = 6.0\), \(\lambda_z = 0.5\)
2. **观察校正后的图像**：
   - 过度平滑（边缘不自然）→ 减小 \(\lambda_v\)
   - 欠校正（仍有照明梯度）→ 增大 \(\lambda_v\)
   - 背景不均匀 → 增大 \(\lambda_z\)
3. **使用小数据集测试**：快速迭代找最佳参数
4. **批量应用到完整数据**

---

## 参考文献

1. **Smith, K., Li, Y., Piccinini, F., Kohler, G., Watkins, S., & Horvath, P. (2015).**  
   *CIDRE: an illumination-correction method for optical microscopy.*  
   **Nature Methods**, 12(5), 404-406.

2. **Nocedal, J., & Wright, S. (2006).**  
   *Numerical optimization.*  
   Springer Science & Business Media.

3. **Huber, P. J. (1964).**  
   *Robust estimation of a location parameter.*  
   **The Annals of Mathematical Statistics**, 35(1), 73-101.

4. **Lindeberg, T. (1998).**  
   *Feature detection with automatic scale selection.*  
   **International Journal of Computer Vision**, 30(2), 79-116.

---

**最后更新**：2025-10-26  
**作者**：SPRINTSeq项目组

