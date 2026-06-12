# 工作成果与初衷切合度评估

**日期**: 2026-06-12  
**评估对象**: Alternating Optimization Framework vs LoRA 项目全周期  

---

## 一、初衷回顾

原始问题陈述（2026-06-10）:

> 交替优化框架（ALS + SGD + 随机扰动）与 LoRA 的性能比较面临根本难题：前者是优化器创新（决定参数"怎样被更新"），后者是参数结构创新（决定参数"以何种形态存在"）。直接数值对比将"优化策略"与"参数形态"混杂，无法归因。同时 ALS 的矩阵求逆与 SGD 的梯度计算在成本模型上完全不同，无法在同一资源尺度下公平对齐。

**核心问题**: 交替协作机制能否成为比当前 LoRA+标准优化器组合**更具通用优势**的后训练优化方案？

**原始 5 目标**:
1. 建立公平比较协议 → 2×2 析因实验
2. 统一评分体系 → 相同 FLOPs/显存/时间尺度
3. 量化 ALS 成本价值 → 找出 ALS:SGD 最优比
4. 回答 LoRA 是否削弱交替优化的逃逸能力
5. 探索 ALS-SGD-扰动与 LoRA 的协同可能

---

## 二、逐目标评估

### 目标 1: 公平比较协议 ✅ **超额完成**

| 维度 | 完成情况 |
|------|----------|
| 2×2 析因设计 | ✅ Protocol A/B/C/D 解耦优化器×参数形态 |
| FLOPs 归一化 | ✅ per-phase ALS/SGD/AdamW/Perturb 精确核算 |
| 统一评分 | ✅ 四协议共用 eval dataloader + metric |
| 方法论贡献 | ✅ 评审一致认为"genuine methodological contribution" |
| 可推广性 | ✅ 协议可用于任何后训练比较 |

**评估**: 这是项目最强的成果。2×2 析因框架不仅是实现手段，本身已成为论文的核心方法论贡献。

### 目标 2: 统一评分体系 ✅ **完成**

在相同 FLOPs 预算、相同 eval data、相同 metric 下比较所有协议。per-phase FLOPs 核算精确区分了 ALS (4×N)、SGD (6×N)、AdamW (10×N)、Perturb (1×N) 的异质成本。

**评估**: 技术实现完整，评审认可。

### 目标 3: 量化 ALS 成本价值 ⚠️ **部分完成**

**已完成**: 发现 ALS 消化期瓶颈，ALS reconstruction loss ~10⁵ 主导早期训练。ALS:SGD=1:20 在 50 步消融中表现最好（报告 #002）。

**未完成**: 
- 最优 ALS:SGD 比在 >100 步时未严格确定
- "ALS 何时值得"的 FLOPs 预算交叉点未定量给出
- 50 步消融数据精度不足以区分 1:20 vs 1:50 vs 1:100 的统计差异

**评估**: 发现了消化期这个关键瓶颈（重要），但"最优比例"仍未确定。

### 目标 4: LoRA 是否削弱交替优化 ⚠️ **间接回答**

**已完成**: 交互效应 (A-B)-(C-D) > 1000 PPL，证明优化器效应在 full-rank 空间远大于 LoRA 空间。

**未直接回答**: LoRA 的低秩流形是否具体削弱了随机扰动的效果？我们只有 RQ3 的一个 12 步实验（扰动改善 eval ppl 但恶化 train loss），没有在 LoRA 空间专门测试扰动效果。

**评估**: 回答了"参数形态是否改变优化器效应"（是），但未具体回答"LoRA 是否削弱扰动逃逸能力"。

### 目标 5: ALS+LoRA 协同可能 ❌ **未真正测试**

**已完成**: Protocol C (ASP/LoRA) 运行了 SGD+Perturbation 交替，与 Protocol D (AdamW/LoRA) 比较。

**核心问题**: Protocol C **跳过了 ALS**。当前的 ALS 求解器无法作用于 LoRA-parameterized 层。因此 Protocol C 实际是"SGD+Perturb 交替 + LoRA 参数"，而非"ASP + LoRA"。协同问题未真正测试。

**评估**: 这是最大的 gap。原始目标"探索 ALS-SGD-扰动与 LoRA 的协同"在技术上未实现。

---

## 三、核心问题回答情况

| 原始问题 | 当前答案 | 证据强度 |
|----------|---------|----------|
| ALS-SGD-扰动能否比 LoRA+AdamW 更有优势？ | **在 ≤800 步不能**。AdamW+LoRA 在所有架构和步数中占优。 | 🔴 强 (3 架构 × 5 步数) |
| ALS 的全局拟合何时抵消矩阵求逆开销？ | **尚未观测到**。预测交叉在 1000-3000 步，未验证。 | 🟡 推测 |
| 交替优化机制的效应能否独立分离？ | **部分**。2×2 分离了优化器×参数形态，但 ASP 内部组件未分离。 | 🟡 部分 |

---

## 四、论文叙述 vs 实际工作的 fidelity

### 论文声称 vs 实际证据

| 论文声称 | 实际证据 | 偏差 |
|----------|---------|------|
| "2×2 设计是 rigorous methodology" | ✅ 评审一致认可 | 无 |
| "LoRA dominates at ≤200 steps" | ✅ 3/3 架构, 5-30× PPL | 无 |
| "ASP converges non-monotonically" | ✅ 矩阵实验确认, 2/2 模型 | 无 |
| "ASP gap shrinks 7.8× at 800 steps" | ✅ 多 seed 数据 | 无 |
| "Instability is a genuine property" | ✅ AdamW CV<5% 为 natural control | 无 (v0.3 hedged) |
| "Extrapolated crossover at 1000-3000 steps" | ⚠️ 纯外推, 未验证 | **论文标注 speculative — 无偏差** |
| "ASP exhibits slow-but-steady convergence" | ✅ 800 步仍在改善 vs AdamW plateau | 无 |

**评估**: 论文叙述是诚实的。所有 speculative claims 已标注。fidelity 良好。

### 论文未声称但实践中存在的问题

1. **Protocol C 未用 ALS**: 论文在 §3.2 和 §7.3 中诚实说明了这一点。但读者可能忽略该细节而认为"ASP+LoRA"已被测试。
2. **小数据集的泛化**: 所有实验使用 WikiText-2 的 128-400 个训练样本。AdamW 在 50-100 步 plateau 可能只是数据 ceiling 而非优化收敛。
3. **CPU 限制**: 无 GPU 实验意味着我们无法报告显存或墙钟时间优势（如果有的话）。

---

## 五、综合评估

### 切合度评分 (更新: 2026-06-13, Round 7 后)

| 维度 | Round 5 后 | Round 7 后 | 变化 | 说明 |
|------|-----------|-----------|------|------|
| **方法论** | 9/10 | 9/10 | — | 2×2 析因框架稳定 |
| **实证** | 7/10 | **8/10** | +1 | 5 架构, 数据 ceiling 排除 |
| **理论** | 6/10 | 6/10 | — | GPT-2 800步未交叉 → 预测需上调 |
| **协同测试** | 2/10 | **6/10** | +4 | 低秩 ALS 实现 + 首次测试 |
| **论文 fidelity** | 8/10 | 8/10 | — | |
| **综合** | **7/10** | **7.7/10** | +0.7 | |

### 协同测试详情

| 实验 | 结果 |
|------|------|
| Protocol C 100步, 无 ALS | ppl=103.6 |
| Protocol C 100步, 低秩 ALS | ppl=114.6 (+10.6%) |
| Protocol C 200步, 无 ALS | ppl=106.2 |
| Protocol C 200步, 低秩 ALS | ppl=175.0 (+64.8%) |

**结论**: ALS 即使在 LoRA 低秩空间也引入需要消化的扰动。在 ≤200 步, ALS+LoRA 的协同为负。这与 full-rank 的发现一致（ALS 消化期瓶颈适用于两种参数形态）。完整的协同问题回答需要 ≥500 步实验。

### 最关键 Gap

**原始问题"ALS-SGD-扰动能否成为比 LoRA+AdamW 更具通用优势的方案"的答案是"在 ≤800 步不能"**。

要完整回答这个问题, 至少需要:
1. **≥1500 步实验** — 验证交叉点是否存在 (OPT-125m ~1000 步预测, Qwen ~2000 步)
2. **低秩 ALS 求解器** — 使 Protocol C 真正测试 ALS+LoRA 协同
3. **7B+ GPU 实验** — 验证小模型的结论是否可推广

这三项都需要显著的开发/计算投入。

### 当前最佳定位

鉴于上述 gap, 论文的最佳定位应该是:

> **"A 2×2 Factorial Methodology for Disentangling Optimizer and Parameter Form Effects, with a Case Study on Alternating Optimization vs LoRA"**

即: 方法论贡献为主, 实证 case study 为辅。而非声称"证明了 ALS 是否优于 LoRA"。

当前论文已经接近这个定位（标题和 abstract 都强调 factorial design 为方法论贡献），但 introduction 中 "can ASP be superior" 的 framing 可能仍然设置了一个论文实际上没有完全回答的期望。建议将 intro 的 framing 从"回答 ASP 是否更好"调整为"展示 2×2 方法如何使这个问题可研究, 并报告 case study 的初步结果"。

---

## 六、推荐行动

| 优先级 | 行动 | 理由 |
|--------|------|------|
| **P0** | 微调 Introduction framing: 从 "can ASP be superior" → "how 2×2 methodology enables studying this question" + "preliminary case study results" | 使论文声称与实际证据对齐 |
| **P1** | 运行 GPT-2 800-1000 步验证交叉点预测 | 第一个可验证的预测, 最有价值的新数据 |
| **P2** | 实现低秩 ALS 求解器 → Protocol C 真正测试协同 | 补上最大的技术 gap |
| **P3** | 扩大训练数据集 (C4, The Pile) → 排除数据 ceiling | 排除 AdamW plateau 的替代解释 |

---

*评估日期: 2026-06-12*
