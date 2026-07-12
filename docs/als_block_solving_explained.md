# ALS 分块求解原理详解

## 1. 问题的起点：我们想解什么？

考虑 lm_head 输出层。权重矩阵 $W \in \mathbb{R}^{V \times d}$，其中 $V = 151936$（词表大小），$d = 896$（隐藏维度）。

输入激活 $X \in \mathbb{R}^{N \times d}$ 来自最后一个 Transformer 层（$N$ 是这批数据的 token 数）。

输出 logits：$Z = X W^\top \in \mathbb{R}^{N \times V}$。

ALS 的目标是：**给定 $X$ 和正确的标签，直接求出最优的 $W$**。

标准最小二乘问题：

$$\min_W \|X W^\top - Y_{\text{target}}\|_F^2$$

闭式解是解一个 $d \times d$ 的线性系统：

$$W^\top = (X^\top X)^{-1} X^\top Y_{\text{target}}$$

**但是** $Y_{\text{target}} \in \mathbb{R}^{N \times V}$ 是一个 $N \times 151936$ 的矩阵。如果直接做 $X^\top Y_{\text{target}}$，这是一个 $d \times V = 896 \times 151936$ 的矩阵乘法。然后还要把结果存下来（~500MB float32），再去解线性系统。这在显存和计算上都很重。

**分块求解的思想**：把 $V = 151936$ 按行切成小块（block），每次只处理一个小块。块大小 $b = 1024$，共 $\lceil 151936 / 1024 \rceil \approx 149$ 块。

---

## 2. 分块的数学原理

把 $W$ 按行切分：

$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_m \end{bmatrix}, \quad W_i \in \mathbb{R}^{b \times d}$$

对应的，输出 logits 也按列切分：

$$Z = X W^\top = [X W_1^\top \;\; X W_2^\top \;\; \cdots \;\; X W_m^\top]$$

**关键观察**：$X$ 对所有块都是**同一个**！这意味着 $(X^\top X)^{-1}$ 也只需要算**一次**。

对于第 $i$ 块，最小二乘问题是：

$$\min_{W_i} \|X W_i^\top - Y_i\|_F^2$$

解：

$$\boxed{W_i^\top = \underbrace{(X^\top X)^{-1}}_{\text{所有块共享}} \cdot \underbrace{X^\top Y_i}_{\text{每块独立计算}}}$$

这就把 $896 \times 151936$ 的大矩阵运算拆成了 149 个 $896 \times 1024$ 的小矩阵运算。

---

## 3. 用代码对照理解

以中间层为例（[`_solve_linear_layer`](altopt/als.py:370)，比输出层更简单，因为没有标签处理）：

```python
# 所有块共享的准备工作（只算一次）
X_f32 = X.detach().float()           # [N, d_in]
XtX = X_f32.T @ X_f32                # [d_in, d_in]  ← 所有块共享！
XtX_reg = XtX + λI                   # 加正则化
L = torch.linalg.cholesky(XtX_reg)   # Cholesky 分解 ← 所有块共享！

n_blocks = (d_out + block_size - 1) // block_size

for i in range(n_blocks):
    start = i * block_size
    end = min(start + block_size, d_out)

    # ── 每块独立的部分 ──
    W_block = W[start:end, :]                # [b, d_in]  ← 取当前块的旧权重
    Y_block = X @ W_block.T                  # [N, b]     ← 旧权重的输出
    XtY = X.T @ Y_block                      # [d_in, b]  ← 正规方程右侧

    # ── 利用共享的 Cholesky 分解快速求解 ──
    W_new_block = torch.cholesky_solve(XtY, L).T  # [b, d_in]

    # ── EMA 阻尼更新 ──
    damped = (1 - α) * W_block + α * W_new_block
    weight[start:end, :] = damped
```

---

## 4. 逐行拆解：每一步在做什么

### 第 1 步：$X^\top X$ — 信息的"压缩"

$$X^\top X \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

这是一个 $896 \times 896$ 的矩阵，**与输出维度 $d_{\text{out}}$ 无关**。它编码了"输入数据的几何结构"——你的输入 token 在 $d_{\text{in}}$ 维空间中分布成什么形状。

**直觉**：$X^\top X$ 类似于协方差矩阵。它告诉你输入 token 在 896 维空间中的各个方向上的"信息量"有多大。信息量大的方向，权重估计就准确；信息量小的方向（接近零的特征值），需要正则化 $\lambda I$ 来稳定。

**关键**：这一步对所有 149 个块只做一次。

### 第 2 步：Cholesky 分解 — 把矩阵变成"三角形"

$$X^\top X + \lambda I = L L^\top$$

Cholesky 把一个**正定矩阵**分解成下三角矩阵 $L$ 及其转置 $L^\top$ 的乘积。这有什么用？

求解 $A x = b$ 需要 $\mathcal{O}(n^3)$。但如果 $A = L L^\top$，可以用**两次三角回代**在 $\mathcal{O}(n^2)$ 内求解：

$$\begin{cases} L z = b & \text{前向代入 — 快} \\ L^\top x = z & \text{后向代入 — 快} \end{cases}$$

这是**所有块都受益的计算**——一次分解，149 次复用。

**为什么矩阵一定是正定的？** $X^\top X$ 是半正定的（任意非零向量 $v$，$v^\top X^\top X v = \|Xv\|^2 \geq 0$）。加上 $\lambda I$（$\lambda > 0$）后，所有特征值至少为 $\lambda$，因此严格正定。Cholesky 分解必然存在且唯一。

### 第 3 步：$Y_{\text{block}} = X \cdot W_{\text{block}}^\top$ — 用旧权重生成 target

对中间层，没有 ground-truth 标签。ALS 的 target 是**当前权重自己生成的输出**：

```python
Y_block = X @ W_block.T   # [N, b]
```

这看起来荒谬——"用旧权重生成输出，然后用这个输出去求新权重"。但如果 $X^\top X$ 是满秩的，那么：

$$(X^\top X)^{-1} X^\top (X W_{\text{old}}^\top) = I \cdot W_{\text{old}}^\top = W_{\text{old}}^\top$$

**解出来的就是原来的权重！** ALS 之所以还能改进权重，是因为：

1. **其他层的 ALS 更新改变了 $X$**——当上一层被 ALS 修改后，这一层的输入 $X$ 变了，原 $W_{\text{old}}$ 不再是 $X$ 下的最优。
2. **数值精度**——float32 运算有微小误差，Cholesky + 三角求解引入微小变化（在这个场景下实际上不是优势，而是噪音）。

所以对中间层 ALS 的语义实际上是：**"在输入 $X$ 下重新校准权重，使其输出更接近线性映射的最优解"**。

### 第 4 步：$X^\top Y_{\text{block}}$ — 正规方程右侧

```python
XtY = X.T @ Y_block   # [d_in, b]
```

$X^\top Y_{\text{block}}$ 是一个 $d_{\text{in}} \times b$ 的矩阵。$b = 1024$，所以 $X^\top Y_{\text{block}}$ 是 $896 \times 1024$ 的。

这一步的直觉：$X^\top Y$ 是"输入 $X$ 和目标输出 $Y$ 之间的协方差"——它编码了"每个输入维度与每个输出维度的相关性"。正规方程 $X^\top X \cdot W^\top = X^\top Y$ 就是"找到一个 $W$ 使得这种相关性被最优建模"。

### 第 5 步：Cholesky 求解 — 一步到位

```python
W_new_T = torch.cholesky_solve(XtY, L)  # 等价于 (XᵀX+λI)⁻¹ XᵀY
```

不显式求逆矩阵（数值不稳定），而是利用 $L$ 做两次三角求解。结果是 $W_{\text{new}}^\top \in \mathbb{R}^{d_{\text{in}} \times b}$，转置后得到 $W_{\text{new}} \in \mathbb{R}^{b \times d_{\text{in}}}$。

### 第 6 步：EMA 阻尼 — 不要跳太远

```python
damped = (1 - α) * W_block + α * W_new_block
```

直接写 $W_{\text{new}}$ 到权重上相当于"一步到最优"。但在 Transformer 的上下文中这很危险（残差连接的扰动放大效应）。EMA 阻尼让权重慢慢移动到最优方向：

$$W \leftarrow 0.99 \cdot W_{\text{old}} + 0.01 \cdot W_{\text{new}}$$

对于靠近输入的层，$\alpha(\ell)$ 被进一步缩小到 0.005。

---

## 5. 输出层 (lm_head) 和中间层的核心区别

| | 中间层 | 输出层 |
|---|---|---|
| **Target $Y$** | $X W_{\text{old}}^\top$（重构） | One-hot 标签（真实信号） |
| **$X$ 的来源** | 同一 batch 的激活 | 同一 batch 的激活 |
| **$X$ 的筛选** | 全部使用 | 只使用标签落在当前块范围内的 token（`mask`） |
| **语义** | 在输入 $X$ 下重新校准 | 真正的最小二乘分类器训练 |

输出层的 target 构造逻辑（[`_solve_head_layer`](altopt/als.py:322-335)）：

```python
# 块 i 覆盖词表位置 [start, end)
mask = (labels >= start) & (labels < end)  # 哪些 token 的标签落在这个块里

X_masked = X[mask]                          # 只取这些 token 的输入
target_tokens = labels[mask] - start        # 映射到块内坐标
Y_target = one_hot(target_tokens)           # 构建 one-hot target

# 解: W_i = argmin ||X_masked · W_i^T - Y_target||²
```

**直觉**：lm_head 的每一行 $w_v \in \mathbb{R}^d$ 都是"token $v$ 的分类向量"。如果 token $v$ 的标签出现了，那么输入 $X$ 应该被 $w_v$ 映射到高概率。ALS 直接求解这个映射——不需要迭代梯度下降。

**注意**：输出层 ALS 对每个块使用了**不同的** $X_{\text{masked}}$（只包含标签落在该块内的 token），因此每个块的 $X^\top X$ 也不同。这意味着输出层每块都需要独立做 Cholesky 分解，不像中间层可以共享。但由于 $b = 1024$，词表被均匀覆盖时每个块只有约 $\frac{800}{149} \approx 5.4$ 个 token 的激活参与——实际上大多数块的 $X_{\text{masked}}$ 很小甚至为空，Cholesky 的成本极低。

---

## 6. 一张图总结整个流程

```
  输入: X [N × 896], 权重 W [151936 × 896]
  ┌─────────────────────────────────────────────┐
  │ ① 算 XᵀX [896×896], Cholesky 分解 → L       │ ← 全局，做一次
  │    用时: O(N·896² + 896³/3)                   │
  ├─────────────────────────────────────────────┤
  │  for 块 i = 0..148:                          │
  │    ② 取 W[1024i : 1024(i+1), :]              │ ← 当前块 [1024 × 896]
  │    ③ 算 Y = X · W_blockᵀ                     │ ← 旧输出 [N × 1024]
  │    ④ 算 XᵀY [896 × 1024]                     │ ← 正规方程右侧
  │    ⑤ Cholesky_solve(L, XᵀY) → W_newᵀ         │ ← 闭式解 [896 × 1024]
  │    ⑥ W_block ← 0.99·W_block + 0.01·W_new     │ ← EMA 阻尼
  │    用时每块: O(N·896·1024 + 896²·1024)        │
  └─────────────────────────────────────────────┘
```

**效率对比**：如果直接解 $151936 \times 896$ 的全系统：

$$\text{FLOPs}_{\text{direct}} = \mathcal{O}(V \cdot d^2) = \mathcal{O}(151936 \cdot 896^2) \approx 1.2 \times 10^{11}$$

分块后，所有块共享步骤①的成本，每块只需：

$$\text{FLOPs}_{\text{per-block}} = \mathcal{O}(b \cdot d^2) = \mathcal{O}(1024 \cdot 896^2) \approx 8.2 \times 10^8$$

总成本大约相同（因为 $\sum b_i = V$），但**显存占用从 $\mathcal{O}(V \cdot d)$ 降到了 $\mathcal{O}(b \cdot d)$**，差了 148 倍（$\sim$500MB → $\sim$3.5MB per block）。这就是分块的根本意义——**用时间换空间**。

---

## 7. 常见疑问

### Q1: 分块求解的结果和一次性求整个 $W$ 的结果一样吗？

**完全一样。** 因为：

$$\min_W \|X W^\top - Y\|^2 = \sum_{i=1}^{m} \min_{W_i} \|X W_i^\top - Y_i\|^2$$

行之间没有耦合——第 $i$ 块的解不影响第 $j$ 块的解。目标函数关于每行是独立的。这是线性最小二乘的分离性（separability）。

**唯一的差别**：如果对不同块使用了**不同的 $X$**（如输出层对每个块只用标签落在该块内的 token），则解会略有不同——因为每个块看到的 $X$ 子集不同。但每个块的解在该块自身的 $X$ 下仍然是最优的。

### Q2: 为什么中间层的 ALS 看起来是在"自己重建自己"？

对，这正是 ALS 的局限。中间层没有 ground-truth label，只能做重构。这等价于在输入 $X$ 下让权重的输出尽量接近线性映射的最优表示。它能"拉直"权重（让 $X W^\top$ 的秩尽量满），但不能给权重注入新的语义信息。

这也就是为什么 ALS 当前只对**输出层**（有真实标签）用处最大，而对中间层收益有限。`_solve_linear_layer` 方法目前存在但未被主循环大范围调用。

### Q3: 块大小 $b$ 怎么选？

$b$ 是显存和计算效率的平衡：

- $b$ 太小 → 块太多 → 循环开销大（149 次 vs 15 次的 Python 循环开销）
- $b$ 太大 → $X^\top Y_{\text{block}}$ 矩阵太大 → 显存压力

当前 $b = 1024$ 是经验值。对 lm_head（$V = 151936$），1024 产生 149 块，每块的 $X^\top Y$ 是 $896 \times 1024$，约 3.5 MB float32——非常舒适的大小。

### Q4: 如果某块没有对应标签怎么办？

跳过。代码中的处理：

```python
mask = (labels_flat >= start) & (labels_flat < end)
if not mask.any():
    continue  # 没有标签落在这个块，直接跳过
```

对于被跳过的块，权重保持 ALS 前的值不变。

### Q5: 为什么用 Cholesky 而不是直接求逆？

1. **数值稳定性**：$X^\top X$ 可能接近奇异，直接求逆会放大误差。Cholesky 分解 + 三角求解对条件数不敏感。
2. **速度**：Cholesky 分解是 $\frac{1}{3}n^3$，求逆是 $n^3$。三角求解是 $n^2$，直接矩阵乘法是 $n^3$。总体快约 3 倍。
3. **无需显式逆矩阵**：`torch.cholesky_solve(XtY, L)` 内部做两次三角回代，等价于 $(L L^\top)^{-1} X^\top Y$，但从不显式计算逆矩阵。

### Q6: 正则化参数 $\lambda$ 的作用？

$$\lambda = 10^{-3}$$

- $\lambda$ 太小时：$X^\top X$ 接近奇异 → Cholesky 失败 → 回退到 `torch.linalg.lstsq`（更慢但更稳）
- $\lambda$ 太大时：解被过度缩向零 → 权重几乎没有变化 → ALS 失去意义

$10^{-3}$ 是经验平衡值——既保证 Cholesky 稳定，又不显著扭曲解。
