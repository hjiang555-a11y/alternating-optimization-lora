# Protocol A 为什么在 Qwen2.5-7B 上无法运行

**一句话结论**: Protocol A 只改 lm_head 这一个输出层，但 Qwen2.5-7B 有 28 层残差连接。lm_head 的任何微小变化（仅 ~1%）被残差连接逐层放大约 8.7 倍 → 浅层梯度被裁剪削平 → SGD 永远追不上 → PPL 在 120 万处震荡，永不收敛。

---

## 1. Protocol A 到底改了模型的哪些参数

看 `altopt/als.py:212-258`，`solve_block()` 方法的逻辑：

```python
# altopt/als.py:243
for name, module in self.model.named_modules():
    if isinstance(module, nn.Linear) and ("lm_head" in name):
        self._solve_head_layer(name, module, batch, labels, block_size)
```

**整个 ALS 阶段只遍历并修改一个层：lm_head。** 其他 27 个 transformer 层的参数完全不动。

具体来说，`_solve_head_layer` (als.py:262-363) 做的事：

```
lm_head 权重矩阵: [50257, 768]  →  分 ~50 个 block，每 block 1024 行

对每个 block:
  1. 找出标签落在当前 block 范围内的所有 token
  2. 构造 one-hot 目标矩阵 Y_target
  3. 用 Cholesky 分解解: W_new = (XᵀX + λI)⁻¹ Xᵀ Y_target
  4. EMA 阻尼写入: W ← 0.99 × W_old + 0.01 × W_new
```

**ALS 只采纳了 1%（`step_size=0.01`）的新解**，旧权重的 99% 被保留。这个保守的设计是为了防止灾难性遗忘。但对 28 层的 Qwen2.5-7B 来说，甚至连这 1% 都太多了。

lm_head 修改完毕后，ALS 阶段结束。之后进入 SGD 阶段。

---

## 2. 为什么区区 1% 的 lm_head 变化会致命

### 2.1 残差连接的结构

Transformer 的每一层都是残差结构：

```
h_{l+1} = h_l + f_l(h_l)
```

其中 `f_l` 是 Self-Attention + FFN。

在反向传播中，lm_head 处计算的损失要通过这个残差链一层层往回传：

```
∂L/∂h_0 = ∂L/∂h_28  ×  ∂h_28/∂h_27  ×  ∂h_27/∂h_26  ×  ...  ×  ∂h_1/∂h_0
            ↑              ↑              ↑                        ↑
         lm_head梯度    层27残差Jacobian  层26残差Jacobian          层0残差Jacobian

每层残差Jacobian = I + ∂f_l/∂h_l，范数 ≈ 1.08（论文 §6.2 拟合值）

总放大倍数 = 1.08^27 ≈ 8.7
```

### 2.2 梯度裁剪的致命后果

SGD 阶段每步做（`altopt/sgd.py:63-99`）：

```python
loss.backward()                          # 反向传播，计算全部参数的梯度
clip_grad_norm_(parameters, max_norm=1.0) # 全局梯度裁剪到 1.0
optimizer.step()                          # θ -= lr × g
```

梯度裁剪会**等比缩放所有参数的梯度**，使得总体积范数不超过 1.0。

但问题来了：lm_head 附近层的梯度是正常的（因为他们离 lm_head 近，没有被残差放大），而**浅层的梯度被放大了 8.7 倍**。裁剪操作的结果是：

```
裁剪前:
  Layer 27 梯度范数: ~0.1     ← 正常
  Layer 0  梯度范数: ~0.87    ← 被残差放大 8.7×

裁剪后（总范数归一化到 1.0）:
  Layer 27 有效更新: 0.1 / 总范数 ≈ 正常
  Layer 0  有效更新: 0.87 / 总范数 → 被严重削平

→ Layer 0 的有效更新量远小于实际需要的量
→ 浅层参数无法配合 lm_head 的变化
→ 参数不对齐
→ 下次前向传播时输出更大的 loss
→ 梯度更大
→ 裁剪更严重
→ 恶性循环
```

---

## 3. 实测数据

### 3.1 Qwen2.5-7B 协议 A 实测结果 (`trainer.py:275-279`)

```
11 次独立尝试，2 个分布式后端（DeepSpeed ZeRO-2 和 PyTorch FSDP），全部失败:

  step 100:  PPL = 1,169,679
  step 200:  PPL = 1,033,027
  step 300:  PPL = 1,120,941
  step 704:  终止

收敛趋势: 不存在。PPL 在 ~120 万处上下震荡，无任何改善。
基线 PPL: 133.16
```

### 3.2 跨深度对比 (`runs/p1.2_depth/results.json`)

| 模型 | 层数 | 协议 A PPL | 基线 PPL | 状态 |
|------|------|-----------|---------|------|
| OPT-125m | 12 | 106.9 | 231 | ✓ 收敛（改善~2×） |
| TinyLlama-1.1B | 22 | 15.5 | 146 | ✓ 收敛（改善~9×） |
| Qwen2.5-0.5B | 24 | 18.0 | 411 | ✓ 收敛（改善~23×） |
| **Qwen2.5-7B** | **28** | **~1,200,000** | **133** | **✗ 发散** |

临界点出现在 **24 层和 28 层之间**。24 层及以下能收敛，28 层及以上不行。这不是巧合——8 个不同架构都遵循这个规律（论文 §5.6）。

### 3.3 八架构验证 (`runs/cross_arch/`)

| 模型 | 层数 | 协议 A | 状态 |
|------|------|--------|------|
| GPT-2 | 12 | 185 | ✓ |
| OPT-125m | 12 | 651 | ✓ |
| TinyLlama-1.1B | 22 | 7,323 | ✓ |
| Qwen2.5-0.5B | 24 | 3,766 | ✓ |
| DeepSeek-1.5B | 28 | NaN | ✗ |
| SmolLM2-135M | 30 | 69,748 | ✗ (非 NaN 但极度恶化) |
| Mistral-7B | 32 | NaN | ✗ |
| Qwen2.5-7B | 28 | ~1.2M | ✗ |

---

## 4. 为什么不通过调参绕过

`als.py` 已经实现了三层保护机制（als.py:68-88）：

```python
depth_decay_beta = 2.0       # 按层深度指数衰减 ALS 更新量
skip_early_ratio = 0.5       # 跳过前 50% 的层（不碰它们）
clip_threshold   = 0.05      # ‖ΔW‖/‖W‖ > 5% 时裁剪
clip_catastrophic = 0.5      # ‖ΔW‖/‖W‖ > 50% 时回滚整个 ALS 周期
```

但这些保护**对 lm_head 无效**。原因在于 `solve_block()` 方法内部的分发逻辑（als.py:195-208）：

```python
def _should_skip_layer(self, name, is_head):
    if is_head:               # ← lm_head 永远不跳过
        return False           # ← ALS 必须解 lm_head，这是它的核心目标
    entry = self._layer_depth_map.get(name)
    if entry is None:
        return True
    return layer_idx < self._sensitive_zone_end  # ← 非 head 层才可能被跳过
```

**lm_head 被硬编码为不可跳过**。因为 ALS 的设计目的就是在 lm_head 上做闭式最小二乘求解——跳过 lm_head 就等于 ALS 阶段什么都不做。

其他曾尝试的缓解策略及结果：

| 策略 | 预期效果 | 实际结果 |
|------|---------|---------|
| EMA 深度衰减 (β=2.0) | 浅层接受更温和的更新 | 无效：ALS 只改 lm_head，不改其他层 |
| 层跳过 (skip_early_ratio=0.5) | 避免长残差链的浅层被波及 | 无效：同上 |
| 范数裁剪 (clip_threshold=0.05) | 限制单层变化幅度 | 对 lm_head 本身的 1% 变化已经够小，但残差放大后依然崩溃 |
| 增加 SGD 步数到 350/周期 | 给 SGD 更多时间消化 | 无效：梯度裁剪和残差放大的矛盾是结构性的，步数多少都解不了 |
| 降低 learning rate | 更温和的更新 | 收敛更慢，但裁剪问题依然存在 |
| Cholesky 换 lstsq | 数值更稳定 | 无效：不是数值精度问题，是梯度流问题 |

---

## 5. 数学本质

```
问题可以形式化描述为：

给定 L 层 transformer:
  h_0 = embedding(x)
  h_{l+1} = h_l + f_l(h_l; W_l)  for l = 0, ..., L-1
  logits = lm_head(h_L)           ← ALS 只改这里的权重

理论上的 ALS 目标:
  找到全部 (W_0, W_1, ..., W_{L-1}, W_lm) 使得 PPL 最小

实际的 ALS 做法:
  固定 (W_0, ..., W_{L-1})，只解 W_lm 的局部最小二乘

不匹配的本质:
  W_lm 的任何改变都会使 (W_0, ..., W_{L-1}) 不再是最优的
  → 需要重新调整前面 27 层的所有参数
  → 但前面 L-1 层被 ALS 冻结了
  → 只能靠 SGD 逐层调整
  → 梯度从 lm_head 逆向传播回 Layer 0 时被残差放大
  → 裁剪削平有效更新
  → 无法收敛

当 L ≤ 24 时：
  放大倍数 ≈ 1.08^24 ≈ 6.3 倍 → SGD 在 125-250 步内可以勉强化解
  实验结果：收敛（但 12L 仍然非单调，24L 有宽方差）

当 L ≥ 28 时：
  放大倍数 ≈ 1.08^27 ≈ 8.7 倍 → 超出了 SGD 的恢复能力上限
  实验结果：全部发散
```

---

## 6. 怎么修

有三个方向（均超出当前 ASP 框架的范围）：

1. **对所有层做 ALS（而不只是 lm_head）**：每层都独立求解最小二乘 → 需要逐层向前传播并缓存所有激活 → 内存开销极大（28 层 × 7B 参数 → 不可行），且层间耦合问题没有根本解决

2. **用对层深度不敏感的优化器替代 SGD**：例如 LARS（Layer-wise Adaptive Rate Scaling）或 LAMB，这些优化器对每层独立计算学习率，可以抵消残差放大。但论文只比较了 AdamW 和 SGD

3. **完全放弃 ALS，只用 SGD + Perturb**：论文 §6.9.2 验证了这个方向的可行性——GPT-2 上 SGD+Perturb 在 800 步达到 PPL 2.00（vs AdamW 2.78），而且没有深度限制。这是目前最有希望的方向

---

## 7. 相关代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `altopt/als.py:212-258` | `solve_block()` | ALS 阶段入口——只遍历 lm_head |
| `altopt/als.py:262-358` | `_solve_head_layer()` | 核心代数——Cholesky 分解 + EMA 阻尼 |
| `altopt/als.py:195-208` | `_should_skip_layer()` | 层跳过逻辑——lm_head 硬编码不可跳过 |
| `altopt/als.py:141-158` | `_depth_aware_step_size()` | 层深度衰减——但 ALS 不走这条路（只影响非 head 层） |
| `altopt/sgd.py:63-99` | `step()` | SGD 一步——含 `clip_grad_norm_(max_norm=1.0)` |
| `altopt/trainer.py:384-465` | `_setup_fsdp()` | 7B 协议 A 的 FSDP 配置 |
| `altopt/trainer.py:678-755` | `_train_fsdp()` | 7B 协议 A 的训练循环 |

## 8. 实验数据位置

| 文件 | 内容 |
|------|------|
| `runs/p1.2_depth/results.json` | 4 模型跨深度 ASP 结果 |
| `runs/cross_arch/summary_cross_arch.json` | 8 架构协议 A vs B 对比 |
| `runs/qwen25_7b_800s/Qwen25-7B_PA_800s_s42.json` | Qwen2.5-7B 协议 A 种子 42 的完整损失曲线 |
| `runs/qwen25_7b_800s/Qwen25-7B_PA_800s_s123.json` | 种子 123 |
| `runs/qwen25_7b_800s/Qwen25-7B_PA_800s_s456.json` | 种子 456 |
