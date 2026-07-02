# 四种算法 (Protocol A/B/C/D) 的实现原理与代码详解

本文档面向初学者，用最简单的方式解释四种 Post-Training 协议是如何在代码层面实现的。

---

## 零、先理解一个普通的 PyTorch 训练循环

在理解四种 Protocol 之前，先看一个**最简单的训练循环**长什么样：

```python
# 这是任何深度学习训练的"基本模板"
model = load_model("Qwen2.5-0.5B")           # 加载模型
optimizer = AdamW(model.parameters(), lr=1e-4) # 创建一个优化器

for step in range(100):                       # 训练 100 步
    batch = get_next_batch()                  # ① 取一批数据
    outputs = model(batch)                    # ② 前向传播：模型预测
    loss = outputs.loss                       # ③ 计算损失（预测错了多少）
    loss.backward()                           # ④ 反向传播：计算梯度
    optimizer.step()                          # ⑤ 更新参数（沿着梯度方向走一步）
    optimizer.zero_grad()                     # ⑥ 清空梯度（为下一步做准备）
```

这 6 步是所有 Protocol 的共同基础。四种 Protocol 的区别，就在于**第 ②-⑥ 步的具体实现方式不同**。

---

## 一、2×2 因子设计：四种协议从哪里来

我们想比较两件事：**用什么优化器**（ASP 还是 AdamW）和**用什么参数形态**（全秩还是 LoRA）。

所以交叉出四种组合：

|               | 全秩参数 (全部 494M 参数都可更新) | LoRA 参数 (只训练 ~3M 的低秩适配器) |
|---------------|----------------------------------|-------------------------------------|
| **ASP 优化器**  | Protocol A                       | Protocol C                          |
| **AdamW 优化器** | Protocol B                       | Protocol D                          |

现在逐个看这四种 Protocol 是怎么实现的。

---

## 二、Protocol B：AdamW + 全秩 (最简基准)

这是**最直接、最标准的微调方式**。你把整个模型的所有参数都交给 AdamW 优化器去调整。

### 2.1 完整训练代码

```python
# ===== Protocol B: AdamW + 全秩 =====
# 这就是工业界最常用的"全参数微调"

from transformers import AutoModelForCausalLM
from torch.optim import AdamW

# ① 加载模型（全部 494M 参数都可以被训练）
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# model.parameters() 返回所有 4.94 亿个参数

# ② 创建 AdamW 优化器，管理所有参数
optimizer = AdamW(
    model.parameters(),    # ← 全部参数！494M 个
    lr=1e-4,              # 学习率：每一步调整多大幅度
    weight_decay=0.01,    # 权重衰减：防止参数过大（正则化）
    betas=(0.9, 0.999)    # Adam 的两个动量参数（历史梯度的"记忆"）
)

# ③ 训练循环
model.train()  # 切换到训练模式（开启 dropout、batch norm 等）
for step in range(100):
    for batch in train_dataloader:
        batch = {k: v.to("cuda") for k, v in batch.items()}

        optimizer.zero_grad()                # 清空上一步的梯度
        outputs = model(**batch)             # 前向传播 → 得到 loss
        loss = outputs.loss
        loss.backward()                      # 反向传播 → 计算所有参数梯度

        # 梯度裁剪：防止某一步梯度太大导致训练崩溃
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()                     # AdamW 更新所有参数

        step += 1
        if step >= 100: break
```

### 2.2 AdamW 到底做了什么？

AdamW 是 Adam 的一个变体，每一步做以下计算：

```
对于每个参数 θ：

  m_t = β₁·m_{t-1} + (1-β₁)·g_t     ← 更新一阶矩（梯度方向的移动平均）
  v_t = β₂·v_{t-1} + (1-β₂)·g_t²    ← 更新二阶矩（梯度大小的移动平均）
  θ_t = θ_{t-1} - lr·(m̂_t / (√v̂_t + ε) + λ·θ_{t-1})
                                               ↑
                                          权重衰减项（这就是 AdamW 比 Adam 多的地方）
```

用通俗的话说：
- **m_t**："最近梯度的平均方向"——像一个有惯性的球，不会突然转向
- **v_t**："最近梯度的大小"——如果某个参数一直波动很大，就减小步长
- **权重衰减**："参数不要太大"——惩罚过大的参数值

### 2.3 Protocol B 的关键特征

| 特征 | Protocol B |
|------|-----------|
| 训练参数数量 | 全部 494M |
| 优化器 | AdamW |
| 更新方式 | 每步都是标准的梯度下降 |
| 显存占用（0.5B） | ~2GB |
| 显存占用（7B） | ~42GB（需要 DeepSpeed ZeRO-2 分到两张卡） |

---

## 三、Protocol D：AdamW + LoRA (实用首选)

这是**最实用的微调方式**。你只训练极少量参数（约 0.1%），其余参数全部冻结。

### 3.1 完整训练代码

```python
# ===== Protocol D: AdamW + LoRA =====
# 只训练 ~3M 参数，而非全部 494M

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

# ① 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ② 在模型上"贴上" LoRA 适配器
lora_config = LoraConfig(
    r=8,                # 秩：把参数更新限制在 8 维子空间内
    lora_alpha=16,      # α：缩放因子，α=2r 是标准设置
    lora_dropout=0.05,  # LoRA 层的 dropout 比率
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    #                ↑ 只在 Q/K/V/O 四个 attention 矩阵上加 LoRA
)
model = get_peft_model(base_model, lora_config)
# 现在 model 有两类参数：
#   - 冻结的基模型参数 (base_model.xxx.weight)  ← 不训练
#   - 可训练的 LoRA 参数 (lora_A, lora_B)        ← 只训练这些！

# ③ 创建优化器：只优化 LoRA 参数
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    #      ↑ 只取 requires_grad=True 的参数 → 只有 LoRA 的 ~3M 参数
    lr=1e-4,
    weight_decay=0.01
)

# ④ 训练循环：和 Protocol B 一模一样！
model.train()
for step in range(100):
    for batch in train_dataloader:
        batch = {k: v.to("cuda") for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()                         # 只计算 LoRA 参数的梯度
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=1.0
        )
        optimizer.step()                        # 只更新 LoRA 参数

        step += 1
        if step >= 100: break
```

### 3.2 LoRA 在每一层做了什么？

假设某一层的 Q 矩阵是 $W_Q \in \mathbb{R}^{4096 \times 4096}$（约 1600 万参数）。

**没有 LoRA 时**：
- 输入 x 经过 W_Q： $h = W_Q \cdot x$
- 训练时更新整个 W_Q（1600 万参数）

**有 LoRA 时**：
- 原始 W_Q 被**冻结**（不再更新）
- 在旁边加两个小矩阵 A 和 B：
  - $A \in \mathbb{R}^{8 \times 4096}$（约 3.2 万参数）
  - $B \in \mathbb{R}^{4096 \times 8}$（约 3.2 万参数）
- 输出变成：$h = W_Q \cdot x + (\alpha/r) \cdot B \cdot A \cdot x$
- 训练时**只更新 A 和 B**（共 6.4 万参数，而非 1600 万）

```python
# LoRA 在代码层面的"等价伪代码"
class LoRALayer:
    def __init__(self, W_base):
        self.W = W_base     # 冻结的原始权重（不可训练）
        self.A = nn.Linear(in=4096, out=8, bias=False)   # 可训练
        self.B = nn.Linear(in=8, out=4096, bias=False)   # 可训练
        self.scaling = alpha / r  # = 16/8 = 2.0

    def forward(self, x):
        # h_base = W·x （冻结部分）
        h_base = F.linear(x, self.W)

        # h_lora = (α/r)·B·A·x （可训练部分）
        h_lora = self.scaling * self.B(self.A(x))

        # 最终输出 = 原始 + LoRA 修正
        return h_base + h_lora
```

### 3.3 Protocol D 的关键特征

| 特征 | Protocol D |
|------|-----------|
| 训练参数数量 | ~3M（仅 LoRA A/B 矩阵） |
| 优化器 | AdamW |
| 更新方式 | 标准梯度下降，但只调 LoRA 参数 |
| 显存占用（0.5B） | ~1.5GB |
| 显存占用（7B） | ~10GB |
| **核心优势** | 省参数、省显存、不易过拟合 |

---

## 四、Protocol A：ASP + 全秩 (最复杂)

这是 ASP 的核心——把 ALS、SGD、Perturb 三个阶段交替执行。

### 4.1 整体结构：Phase 循环

ASP 不是"每步都一样"的训练，而是**三个阶段轮流执行**：

```
Cycle 1: ALS (1步) → SGD (50步) → Perturb (1步)
Cycle 2: ALS (1步) → SGD (50步) → Perturb (1步)
Cycle 3: ALS (1步) → SGD (50步) → Perturb (1步)
```

### 4.2 完整训练代码

```python
# ===== Protocol A: ASP + 全秩 =====

from transformers import AutoModelForCausalLM
from altopt.als import ALSBlockSolver

# ① 加载模型（float32 → ALS 的矩阵运算需要高精度）
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float32,  # ← 注意：float32，不是 bfloat16！
    device_map="auto"
)

# ② 初始化 ALS 求解器
als_solver = ALSBlockSolver(model)  # 内部存了模型的引用

# ③ 配置 ASP 调度
# ASP 采用 3 个 Cycle，每个 Cycle 包含 ALS→SGD→Perturb
sgd_per_cycle = 50  # 每个 Cycle 内做 50 步 SGD
n_cycles = 3        # 总共 3 个 Cycle

global_step = 0
for cycle in range(n_cycles):
    # ═══════════════════ Phase I: ALS ═══════════════════
    model.eval()  # ALS 在 eval 模式下计算
    with torch.no_grad():
        # ALS 求解：对于每一层，闭式求解最优 W
        # 内部流程：
        #   1. 把这一层的 W 按行分成 1024 行的小块
        #   2. 对每块解正规方程：(XᵀX + λI)·W_newᵀ = XᵀY
        #   3. 把求得的 W_new 直接写回模型参数
        als_solver.solve_block(batch, block_size=1024)
    global_step += 1

    # ═══════════════════ Phase II: SGD ═══════════════════
    # ALS 的精确求解改变了模型参数，但没考虑层间协调。
    # SGD 的作用：让各层参数互相适应。
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,
        momentum=0.9,       # 动量：让优化有"惯性"
        weight_decay=0.01
    )
    model.train()
    for sgd_step in range(sgd_per_cycle):
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            if sgd_step >= sgd_per_cycle: break

    # ═══════════════════ Phase III: Perturb ═══════════════════
    # 加扰动：帮模型跳出局部最优，找到更平坦的最优点
    with torch.no_grad():
        sigma = 1e-3  # 扰动强度（初始值，后续会余弦衰减）
        for param in model.parameters():
            noise = sigma * torch.randn_like(param)  # 高斯噪声
            param.add_(noise)  # in-place 加噪声
    global_step += 1

# 最终评估
ppl = evaluate(model, eval_dataloader)
```

### 4.3 ALS 到底怎么做？（逐行解释）

```python
# ALS 的核心代码（简化自 altopt/als.py）
# 目标：给定输入 X，找到最优的 W 使得 ‖X·Wᵀ - Y‖² 最小

def solve_block(layer_name, X, Y_old, block_size=1024):
    W = layer_name.weight.data          # [d_out, d_in]
    d_out, d_in = W.shape

    # Step 1: 建矩阵 (XᵀX + λI)
    # 这是一个 d_in × d_in 的正定矩阵（保证可逆）
    XtX = X.T @ X                       # [d_in, d_in]
    reg = 1e-4 * torch.eye(d_in)        # λI 正则项
    XtX_reg = XtX + reg                 # (XᵀX + λI)

    # Step 2: 按块求解
    # 把 d_out（=4096）分成 block_size（=1024）的小块
    n_blocks = d_out // block_size       # = 4 块

    for i in range(n_blocks):
        start = i * block_size           # = 0, 1024, 2048, 3072
        end = start + block_size         # = 1024, 2048, 3072, 4096

        # 当前块的"目标输出"
        Y_block = X @ W[start:end, :].T  # [N, block_size]

        # 解: (XᵀX + λI) · W_new_T = XᵀY
        XtY = X.T @ Y_block              # [d_in, block_size]
        W_new_T = torch.linalg.solve(XtX_reg, XtY)  # ← 这就是那个复杂的矩阵求逆

        W_new = W_new_T.T                # [block_size, d_in]

        # 直接写入新权重
        W[start:end, :] = W_new

    return W
```

### 4.4 Perturb 在做什么？

```python
# 扰动就是给所有参数加一点随机噪声
# 为什么？→ 帮助跳出窄的局部最优

def perturb(model, sigma=1e-3):
    with torch.no_grad():
        for name, param in model.named_parameters():
            # 生成和参数形状一样的高斯噪声
            noise = sigma * torch.randn_like(param)
            # 直接加到参数上
            param.add_(noise)

# 训练前期：sigma 大（更多探索）
# 训练后期：sigma 小（更多精调）
# 用余弦衰减控制：σ(t) = σ₀ * 0.5 * (1 + cos(π*t/T))
```

### 4.5 Protocol A 的关键特征

| 特征 | Protocol A |
|------|-----------|
| 训练参数数量 | 全部 494M |
| 优化器 | ALS + SGD + Perturb（三阶段交替） |
| 更新方式 | ALS：闭式解 → SGD：梯度 → Perturb：噪声 |
| 数值精度 | float32（ALS 需要高精度） |
| 显存占用（0.5B） | ~2GB |
| 显存占用（7B） | ❌ 被深度边界阻断（≥28 层不收敛） |
| **核心风险** | ALS 扰动可能太大，SGD 无法恢复 |

---

## 五、Protocol C：ASP + LoRA (不对称组合)

Protocol C 是 ASP 优化器用于 LoRA 参数形态——但有一个重要 caveat：**当前 ALS 求解器不能直接用于 LoRA 参数**。

### 5.1 为什么 ALS 不能直接用于 LoRA？

ALS 求解器的设计是针对 `nn.Linear` 层的——它直接读取和写入 `module.weight.data`。

但 LoRA 层的参数形态是：
```
W_eff = W_base + (α/r) · B · A
```

这里 `W_base` 是冻结的，可训练的是 `A` 和 `B`。ALS 求解器不知道如何"写入" `A` 和 `B`。

因此 Protocol C **实际上是 "SGD + Perturb" 两个阶段的交替**，不包含 ALS。论文把这种不对称性作为一个已知限制写在了 §3.2。

### 5.2 Protocol C 的训练代码

```python
# ===== Protocol C: ASP on LoRA =====

from peft import LoraConfig, get_peft_model

# ① 加载模型 + LoRA
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = get_peft_model(base, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
))

# ② ASP 调度：SGD → Perturb 循环（没有 ALS！）
sgd_per_cycle = 50
n_cycles = 3
global_step = 0

for cycle in range(n_cycles):
    # ═══════════════════ Phase II: SGD ═══════════════════
    # 和 Protocol A 的 SGD 阶段一样，但只优化 LoRA 参数
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        #      ↑ 只取 LoRA 参数（A, B 矩阵）
        lr=1e-4,
        momentum=0.9,
        weight_decay=0.01
    )
    model.train()
    for sgd_step in range(sgd_per_cycle):
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            global_step += 1
            if sgd_step >= sgd_per_cycle: break

    # ═══════════════════ Phase III: Perturb ═══════════════════
    with torch.no_grad():
        sigma = 5e-4  # LoRA 模式下用更小的扰动
        for param in model.parameters():
            if param.requires_grad:  # 只扰动 LoRA 参数
                param.add_(sigma * torch.randn_like(param))
    global_step += 1
```

### 5.3 低秩 ALS 求解器（X1 扩展，2026-06-23 实现）

我们的 X1 扩展成功实现了**能用于 LoRA 参数的 ALS 求解器**。核心思想是：

1. 在满秩空间求解完整的 $W_{\text{new}}$
2. 计算差异 $\Delta W = W_{\text{new}} - W_{\text{eff}}$
3. 把 $\Delta W$ **投影回 LoRA 的 B 矩阵**：$\Delta B = \Delta W \cdot A^T \cdot (AA^T + \lambda I)^{-1} / \alpha$

```python
# X1 低秩 ALS 的核心投影步骤（altopt/als.py 第 554-569 行）
delta_W = W_new_block - effective_W[start:end, :]

# 关键公式：把满秩的 ΔW 映射回低秩的 B 空间
# delta_B = ΔW · A^T · (AA^T + λI)^{-1} / scaling
delta_B = delta_W @ A_pinv.T / scaling

# 直接在 B 矩阵上做 in-place 更新
lora_B[start:end, :] += delta_B.to(lora_B.dtype)
```

这消除了 Protocol C 的 ALS 不对称性——论文最重大的方法论限制已被解决。

### 5.4 Protocol C 的关键特征

| 特征 | Protocol C |
|------|-----------|
| 训练参数数量 | ~3M（LoRA A/B） |
| 优化器 | SGD + Perturb（不含 ALS） |
| 更新方式 | 梯度 + 噪声，只影响 LoRA 参数 |
| 核心限制 | ALS 相位缺失（当前论文版本） |
| X1 修复 | 低秩 ALS 已实现，7B 可运行 |

---

## 六、四种 Protocol 的完整对比

### 6.1 训练循环对比

```python
# ─── Protocol A: ASP + 全秩 ───
for cycle in range(n_cycles):
    als_solver.solve_block(batch)      # ALS：闭式求解，改所有参数
    for _ in range(50):
        sgd_step(model)                 # SGD：梯度下降，协调各层
    perturb(model, sigma=1e-3)          # 扰动：加噪声，探索更优解

# ─── Protocol B: AdamW + 全秩 ───
for step in range(100):
    adamw_step(model)                  # AdamW：自适应梯度，改所有参数

# ─── Protocol C: ASP + LoRA ───
for cycle in range(n_cycles):
    for _ in range(50):
        sgd_step(loRA_only)            # SGD：梯度下降，只改 LoRA
    perturb(loRA_only, sigma=5e-4)     # 扰动：更小的噪声

# ─── Protocol D: AdamW + LoRA ───
for step in range(100):
    adamw_step(loRA_only)             # AdamW：自适应梯度，只改 LoRA
```

### 6.2 全面特征矩阵

| 维度 | Protocol A | Protocol B | Protocol C | Protocol D |
|------|-----------|-----------|-----------|-----------|
| **优化器** | ASP（三阶段） | AdamW | ASP（两阶段） | AdamW |
| **参数形态** | 全秩 | 全秩 | LoRA r=8 | LoRA r=8 |
| **训练参数数** | 494M | 494M | ~3M | ~3M |
| **每步计算** | ALS: O(b³) → SGD: O(d²) → Perturb: O(d) | O(d²) + 2m | SGD: O(d²) → Perturb: O(d) | O(d²) + 2m |
| **数值精度** | float32 | bfloat16 | bfloat16 | bfloat16 |
| **更新频率** | ALS 稀疏（每 52 步 1 次） | 每步 | SGD 密集（每步） | 每步 |
| **训练速度** | 最慢（ALS 开销大） | 快 | 较快 | 最快 |
| **过拟合风险** | 低（隐式正则化） | 高 | 低 | 低 |
| **深度限制** | ≤24 层 | 无 | ≤24 层 | 无 |
| **实用性** | 特殊场景 | 有大量数据时 | 理论价值 | **日常首选** |

### 6.3 FLOPs 预算：如何公平比较

因为不同操作的代价不同，我们必须按**总计算量**而非**总步数**来公平比较：

| 操作 | FLOPs | 相对代价 |
|------|-------|---------|
| 一次前向传播 | $2 \times N_{\text{params}}$ | 1× |
| 一次反向传播 | $4 \times N_{\text{params}}$ | 2× |
| 一次 SGD 更新 | $N_{\text{params}}$ | 0.5× |
| 一次 AdamW 更新 | $3 \times N_{\text{params}}$ | 1.5× |
| 一次 ALS 块求解 | $N \times b^2 + b^3$ | ~100× per block |

```
Protocol 运行至 Σ FLOPs ≥ BUDGET，而非 Σ steps ≥ BUDGET
```

### 6.4 实际使用中选哪个 Protocol？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 日常微调（≤800步） | **Protocol D** | 最快、最省显存、不易过拟合 |
| 有大量数据（>1万条） | Protocol B | 全秩可能更好，但注意过拟合 |
| 极低数据（≤400条） | Protocol A | ASP 的隐式正则化防止过拟合 |
| 模型≤24层 + 长训练 | Protocol A | ASP 在 800+ 步时可能追上 AdamW |
| 模型≥28层 | ❌ Protocol A | 深度边界——必定发散 |

---

## 七、一个具体的数值例子

假设我们要在 Qwen2.5-0.5B 上做 100 步的 WikiText-2 后训练。

**Protocol D (LoRA r=8)**：
```
Load model        → 1s
Train 100 steps   → 25s (每步约 0.25s, 只更新 LoRA 参数)
Evaluate          → 2s
Total: ~30s
PPL: 1.62
Peak memory: ~1.5GB
```

**Protocol B (全秩)**：
```
Load model        → 1s
Train 100 steps   → 30s (每步约 0.3s, 更新全部参数)
Evaluate          → 2s
Total: ~33s
PPL: 44.4 (严重过拟合!)
Peak memory: ~2GB
```

**Protocol A (ASP + 全秩)**：
```
Load model (fp32) → 2s
ALS (1步)         → 2s (矩阵求逆)
SGD (50步)        → 15s
Perturb (1步)     → 1s
ALS (1步)         → 2s
SGD (50步)        → 15s
Perturb (1步)     → 1s
Evaluate          → 2s
Total: ~40s
PPL: 3,766 (开始收敛但还没追上 AdamW)
Peak memory: ~2GB
```

这个例子清楚地说明了为什么**在日常使用中 Protocol D 是首选**——它最快、最省显存、效果最好。
