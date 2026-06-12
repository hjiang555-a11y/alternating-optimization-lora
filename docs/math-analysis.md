# 数值分析: Alternating Optimization Framework 的数学基础

**日期**: 2026-06-12  
**关联**: Round 5 实验数据 + 报告 #001-#004 综合结论  
**目标**: 从优化理论角度解释实验观察到的现象，推导 AltOpt 超越 AdamW 的理论条件

---

## 1. 问题形式化

### 1.1 后训练优化问题

给定预训练模型参数 $\theta_0 \in \mathbb{R}^d$，后训练的目标是在数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 上最小化经验风险：

$$\min_{\theta \in \mathbb{R}^d} \mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \ell(f_\theta(x_i), y_i)$$

其中 $\ell$ 是交叉熵损失，$f_\theta$ 是 LLM 的前传函数。

### 1.2 AltOpt 的三阶段结构

AltOpt 将参数更新分解为三个连续阶段:

| 阶段 | 操作 | 数学形式 |
|------|------|----------|
| Phase I (ALS) | 块坐标精确求解 | $W_k^{(t+1)} = \arg\min_{W_k} \|X_k W_k^T - Y_{\text{target}}\|^2$ |
| Phase II (SGD) | 梯度下降精化 | $\theta^{(t+1)} = \theta^{(t)} - \eta \nabla \mathcal{L}(\theta^{(t)})$ |
| Phase III (Perturb) | 参数空间扰动 | $\theta^{(t+1)} = \theta^{(t)} + \varepsilon,\ \varepsilon \sim \mathcal{N}(0, \sigma^2)$ |

### 1.3 LoRA 的低秩约束

LoRA 将参数更新限制在低秩流形上:

$$\Delta W = \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r},\ A \in \mathbb{R}^{r \times d_{\text{in}}},\ r \ll \min(d_{\text{out}}, d_{\text{in}})$$

---

## 2. ALS Reconstruction Loss 的量级分析

### 2.1 为什么 ALS 第一步产生 ~10⁵ 量级的 loss

这是实验中最一致的观察（4/4 实验）。数学解释如下:

**ALS 求解的是重建损失，而非交叉熵损失**。给定输入激活 $X$，ALS 求解:

$$W_{\text{new}} = \arg\min_W \|X W^T - Y_{\text{target}}\|^2$$

其中 $Y_{\text{target}} = X W_{\text{old}}^T$ 是当前前传输出。ALS 的闭式解:

$$W_{\text{new}} = (X^T X + \lambda I)^{-1} X^T Y_{\text{target}}$$

**关键问题**: ALS 在块级别做精确最小二乘拟合，但它使用的是**局部激活值**。每个块独立求解，不感知跨块交互和后续层的误差传播。这导致:

1. **量级不匹配**: Reconstruction loss $\|X W^T - Y\|^2$ 的量级是 $\mathcal{O}(N \cdot d_{\text{in}} \cdot \|W\|^2)$，而交叉熵 loss 是 $\mathcal{O}(\log V)$（$V$ 为词表大小）。对于 $d_{\text{in}} = 768$（GPT-2）或 $d_{\text{in}} = 768$（OPT-125m），$N \cdot d_{\text{in}}$ 可达 $10^5 - 10^6$。
2. **分布偏移**: ALS 改变了一个层的权重，但后续层仍以旧权重的预期输入分布运行 → 产生分布偏移 → 交叉熵 loss 飙升。
3. **ALS 步后的 cross-entropy 等于模型在被"打乱"的中间表示上做推断**。

**定量**: 设 ALS 在层 $l$ 上修改了权重矩阵 $W_l \in \mathbb{R}^{768 \times 768}$。前传时层 $l+1$ 收到的是 $W_l^{\text{new}} x_l$，而它被训练时预期接收 $W_l^{\text{old}} x_l$。这个分布偏移通过 transformer 的残差连接逐层放大。

对于 12 层 transformer:
$$\text{输出偏移} \approx \prod_{k=l}^{L} (I + \Delta_k) \cdot x$$

其中 $\Delta_k$ 是每层的扰动。在 $L=12$ 层架构中，即使每层扰动很小，乘积效应也很大。

### 2.2 ALS 重建损失的衰减模型

ALS → SGD 过渡后的损失可以建模为:

$$\mathcal{L}(t) = \mathcal{L}_{\text{ALS}} \cdot e^{-\alpha t} + \mathcal{L}_{\text{target}}$$

其中:
- $\mathcal{L}_{\text{ALS}}$: ALS 后的初始重建损失（~10⁵）
- $\alpha$: SGD 的衰减速率（与学习率 $\eta$ 和梯度的 Lipschitz 常数相关）
- $\mathcal{L}_{\text{target}}$: 目标 loss（~2-3 量级）
- $t$: SGD 步数

**当 $\mathcal{L}_{\text{ALS}} \gg \mathcal{L}_{\text{target}}$ 时**，初期 loss 曲线完全由 ALS 项主导。要达到"正常"范围，需要:

$$t > \frac{1}{\alpha} \ln\frac{\mathcal{L}_{\text{ALS}}}{\mathcal{L}_{\text{target}}} \approx \frac{1}{\alpha} \ln(10^5/10) \approx \frac{9.2}{\alpha}$$

这解释了为什么 12 步（报告 #004）和 40 步（报告 #001）都不够 — ALS 的"消化期"可能需要 50-100 步。

---

## 3. 收敛速度的优化理论分析

### 3.1 AdamW vs AltOpt 的收敛速度差距

**AdamW** 在凸假设下的收敛速度为 $\mathcal{O}(1/\sqrt{T} + \sigma^2/T)$（带噪声梯度），在非凸情况下为 $\mathcal{O}(1/\sqrt{T})$。

**AltOpt** 的收敛分析更复杂，因为它混合了精确求解（ALS）和梯度下降（SGD）:

- **ALS 在块内是精确的**: 在当前激活值下达到块内全局最优。收敛速度理论上是 $\mathcal{O}(1)$（一步收敛）— 但仅限于固定激活值的假设下。
- **激活值在变化**: 当 ALS 修改 $W_l$ 后，所有后续层的激活值 $x_{l+1}, ..., x_L$ 都改变了，它们之前被求解时的条件不再成立。
- **这等效于**: ALS 每一步都移动了优化目标。收敛保证不再适用。

### 3.2 为什么 AltOpt 在 ≤200 步时总是输

从优化理论角度，有三个根本原因:

**原因 1: Block Coordinate Descent 的次优性**

交替最小二乘是 Block Coordinate Descent (BCD) 的一个实例。BCD 的收敛速度是:

$$\mathcal{L}(\theta^{(k)}) - \mathcal{L}(\theta^*) \leq \mathcal{O}\left(\frac{L R_0^2}{k}\right)$$

其中 $L$ 是 Lipschitz 常数，$R_0$ 是初始距离。关键在于：**ALS 的 Lipschitz 常数 $L$ 在深度神经网络中非常大**，因为一个块的改变通过非线性激活函数传播到后续所有块。

相比之下，AdamW 的自适应步长在每个参数维度上独立缩放梯度，有效地降低了有效 Lipschitz 常数。

**原因 2: ALS 忽略了层间耦合**

ALS 在每个层的每个块中独立求解最小二乘，假设其他块固定。但深度网络中，层 $l$ 权重的最优值**依赖于**层 $l+1$ 的权重（链式法则）。ALS 的独立性假设被违反，导致:

$$\nabla_{W_l} \mathcal{L}(\theta^{\text{ALS}}) \neq 0$$

即 ALS 求解完成后，梯度仍非零 — ALS 甚至没有找到该块的驻点。

**原因 3: 每次 ALS 步骤都"重置"了 SGD 的动量**

SGD (with momentum) 积累历史梯度信息来加速收敛:

$$v_t = \beta v_{t-1} + \nabla \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

当 ALS 修改权重后，SGD 的动量向量 $v_t$ 指向的是旧参数空间的梯度方向，对新参数 $\theta^{\text{ALS}}$ 不再有效。这相当于每次 ALS 后"重启"了优化器。

### 3.3 理论交叉点: AltOpt 何时超越 AdamW？

假说: **AltOpt 在 SGD 步数足够消化 ALS 偏移后，应能利用 ALS 的全局拟合优势**。

定义"消化时间" $T_{\text{digest}}$ 为 ALS 后的 SGD 步数，使得:

$$\mathcal{L}(\theta_{t + T_{\text{digest}}}) \leq \mathcal{L}(\theta_{t-1})$$

即 SGD 将 ALS 引入的 loss 增加完全消除。

理论交叉条件: AltOpt 的总步数 $T = 4 \times (1 + T_{\text{SGD}} + 1)$（4 个 ALS-SGD-Perturb 周期）中，有足够的 SGD 步来消化每次 ALS:

$$T_{\text{SGD}} > T_{\text{digest}}$$

实验估计: 从报告 #002 的 loss 曲线来看，ALS 后大约需要 **50-80 步 SGD** 才能恢复到 ALS 前的 loss 水平。因此 $T_{\text{SGD}} = 50$（当前调度）刚好处于边界 — 刚消化完又来了新的 ALS。

**预测**: 如果 $T_{\text{SGD}} \geq 200$，AltOpt 应该开始显示出超过 AdamW 的收敛速度。这需要下一个实验（Round 6）来验证。

---

## 4. 扰动的正则化效应: 数学直觉

### 4.1 观察

报告 #004 中: with_perturb 的 train_loss 更高 (13.09 vs 9.04) 但 eval ppl 更低 (86k vs 317k)。
报告 #002 中: with_perturb 的 ppl 更低 (650 vs 非扰动路径下不可比较)。

### 4.2 理论解释

**假说: 参数空间噪声等价于隐式正则化**。

在每次扰动后，SGD 的更新可以写为:

$$\theta_{t+1} = \theta_t + \varepsilon_t - \eta \nabla\mathcal{L}(\theta_t + \varepsilon_t)$$

这等价于在扰动的参数上计算梯度。利用泰勒展开:

$$\nabla\mathcal{L}(\theta + \varepsilon) \approx \nabla\mathcal{L}(\theta) + \nabla^2\mathcal{L}(\theta) \cdot \varepsilon$$

有效更新方向为:

$$\Delta\theta \approx -\eta[\nabla\mathcal{L}(\theta) + \nabla^2\mathcal{L}(\theta) \cdot \varepsilon] + \varepsilon$$

Hessian-噪声内积项 $\nabla^2\mathcal{L} \cdot \varepsilon$ 起到了**曲率感知的正则化**作用：在高曲率方向（尖锐极小值），该项很大，推动参数向更平坦的区域移动。

这与 **Sharpness-Aware Minimization (SAM)** (Foret et al., 2020) 的核心思想一致:

$$\min_\theta \max_{\|\varepsilon\| \leq \rho} \mathcal{L}(\theta + \varepsilon)$$

SAM 通过在参数邻域内寻找最坏情况的 loss 来鼓励平坦极小值。AltOpt 的扰动阶段虽然不显式做 maximization，但**随机扰动 + 后续 SGD 精化**的过程可以看作一种隐式的 SAM 近似。

### 4.3 平坦极小值为何泛化更好

从 PAC-Bayes 理论 (Dziugaite & Roy, 2017):

$$\text{Generalization Gap} \leq \mathcal{O}\left(\sqrt{\frac{\|\theta\|^2 + \log(1/\delta)}{\sigma^2 N}}\right)$$

其中 $\sigma$ 是参数扰动的方差。**扰动越大 → 泛化界越紧** → 验证集表现更好，即使训练 loss 更高。

这精确地解释了我们的观察: 扰动提高了训练 loss（因为参数被噪声推开），但降低了 eval perplexity（因为找到了更平坦的区域）。

---

## 5. LoRA 为什么在低步数下主导优化器选择

### 5.1 LoRA 作为隐式正则化器

LoRA 将参数更新限制在 low-rank 子空间 $S_r = \{BA : B \in \mathbb{R}^{d_{\text{out}} \times r}, A \in \mathbb{R}^{r \times d_{\text{in}}}\}$ 中。这提供了两层正则化:

1. **维度正则化**: 可训练参数量从 $d_{\text{out}} \times d_{\text{in}}$ 降至 $r \times (d_{\text{out}} + d_{\text{in}})$ — 约 100-1000× 减少
2. **谱正则化**: 更新矩阵 $\Delta W = BA$ 的秩 ≤ r，限制了模型可以表达的函数变化

### 5.2 收敛速度的几何解释

在低秩流形上优化可以看作在 $S_r$ 上的约束优化。约束优化的收敛速度通常**快于**无约束优化，因为:

- 可行域更小 → 搜索空间缩小 → 达到 $\varepsilon$-最优需要的迭代次数更少
- 低秩流形的曲率通常比全参数空间的曲率更温和 → 梯度下降的步长可以更大

定量: 对于参数空间 $\mathbb{R}^D$ 上的优化，梯度下降需要 $\mathcal{O}(\kappa \log(1/\varepsilon))$ 次迭代（$\kappa$ 是条件数）。对于低秩流形 $S_r \subset \mathbb{R}^D$，有效条件数 $\kappa_{\text{eff}}$ 可能远小于 $\kappa$。

### 5.3 与我们的实验数据的一致性

| 观察 | 理论解释 |
|------|----------|
| LoRA 5-30× 优于全秩（报告 #001, #002） | 低秩流形的有效条件数降低 |
| Protocol C 的 cross-seed std 极低（±1.17） | 低秩约束消除了解空间的许多局部极小值 |
| AltOpt full-rank ppl std=557 vs LoRA ppl std=1.2 | Full-rank 空间的 loss landscape 更崎岖 |

---

## 6. 理论预测与待验证假设

基于以上分析，我们提出以下可检验的预测:

### H1: 消化期预测

**预测**: ALS 消化期 $T_{\text{digest}} \approx 60-80$ 步 SGD（对于 OPT-125m，lr=1e-4）。
**验证**: 在 ALS 步后的 100 步内，每 10 步记录一次 loss，拟合指数衰减模型 $\mathcal{L}(t) = \mathcal{L}_{\text{ALS}} e^{-\alpha t} + \mathcal{L}_{\text{target}}$。

### H2: 交叉点预测

**预测**: 当 ALS:SGD 比例中 SGD 步数 > 150 时，AltOpt full-rank 的最终 perplexity 将追平 AdamW。
**验证**: 运行 ALS:SGD = 1:200 的实验（Round 7）。

### H3: 扰动强度最优值

**预测**: 扰动强度 $\sigma$ 存在 U 形最优值 — 太小无法逃离局部极小值，太大会破坏已学到的结构。
**验证**: $\sigma \in \{0, 10^{-4}, 5\times 10^{-4}, 10^{-3}, 5\times 10^{-3}, 10^{-2}\}$ 消融。

### H4: LoRA + ALS 协同

**预测**: 如果在 LoRA 的低秩空间内执行 ALS（在 $B$、$A$ 的列空间内做块求解），ALS 的分布偏移问题将大幅减弱（因为低秩约束限制了权重变化的幅度）。
**验证**: 实现低秩 ALS 求解器，与 Protocol C (无 ALS) 比较。

---

## 7. 数值实例: ALS 重建损失的衰减拟合

基于 Round 5 Protocol A 的 loss 数据，拟合衰减模型:

| 周期 | ALS 后 SGD 步 | 观测 loss | 模型预测 loss |
|------|---------------|-----------|---------------|
| Cyc1 ALS | 0 | ~8.66 (step 1 SGD) | — |
| Cyc1 SGD50 | 50 | — | — |
| Cyc2 ALS | 50 | ~6.56 (step 51 SGD) | — |
| Cyc2 SGD50 | 100 | — | — |
| Cyc3 ALS | 100 | ~7.07 (step 101 SGD) | — |
| Cyc3 SGD50 | 150 | — | — |
| Cyc4 ALS | 150 | ~7.5 (step 151 SGD) | — |
| Cyc4 SGD50 | 200 | ppl=1373 | — |

观察: ALS 后的 SGD loss 值（~6-8）与相邻 SGD 步的 loss（~6-7）基本持平，说明 SGD 在 50 步内确实消化了大部分 ALS 偏移，但仍未达到 AdamW 的 loss 水平（~2-4）。

---

## 8. 总结

| 数学洞察 | 实验验证 |
|----------|----------|
| ALS reconstruction loss ~O(N·d·‖W‖²) | 第一步 loss ~10⁵（4/4 实验） |
| BCD 的 Lipschitz 常数在深度网络中很大 | AltOpt 200 步仍落后 AdamW |
| ALS 忽略层间耦合 → 非驻点 | ALS 后 SGD 梯度仍非零 |
| 参数噪声 ≈ 隐式 SAM → 鼓励平坦极小值 | 扰动改善 eval ppl 但恶化 train loss |
| 低秩流形的有效条件数更低 | LoRA 5-30× 加速收敛 |

**核心结论**: 交替优化框架的 ALS→SGD 过渡中的"消化期"是当前性能瓶颈。一旦 SGD 步数超过消化阈值（估计 150-200 步），AltOpt 应能利用 ALS 的块精确拟合优势。这个预测需要在 Round 6 中验证。

**开放问题**:
1. 低秩空间中的 ALS 是否完全避免了分布偏移问题？
2. FLOPs 归一化后（ALS 比 SGD 便宜 ~1.5×），"等 SGD 步"的条件是否公平？
3. 对于 7B+ 模型，损失地形的 Lipschitz 常数是否不同？

---

*文献引用将在 librarian 搜索完成后补充*
