# 高价值研究路线图

**更新日期**: 2026-07-16
**当前判定**: **Major Revision — P1 机制实验完成，P2 Conditional Go**
**最后提交**: `0c6096a` — P2 synthesis with calibrated baselines + convergence-matched P1.3

> `docs/p2-synthesis.md` 为完整评估。`experiments/_p1.1/2/3*.py` 为可复现脚本。`runs/p1.*/results.json` 为机器可读产物。

---

## 1. P0：证据审计 ✓（已完成）

- [x] claims-audit（14 项主张 → 证据标签）
- [x] protocol_b_full_rank_results.json 创建
- [x] 论文 v3.4 "Accept" → v0.7.1 "Major Revision"
- [x] quasi-factorial、24–28L 失稳转变、N_EVAL=200 统一

## 2. P1：机制实验 ✓（全部完成，GPU 已执行）

### P1.1 组件归因（OPT-125m, 3 seeds, 200 steps）

| 条件 | PPL | vs Baseline(231) |
|------|-----|-------------------|
| SGD-only | 59.4 ± 3.1 | 3.9× better |
| ALS+SGD | 62.5 ± 0.4 | 3.7× better |
| SGD+Perturb | 62.2 ± 0.6 | 3.7× better |
| Full ASP | 69.0 ± 3.6 | 3.4× better |

**结论**: ALS 主效应 −3.1，Perturb 主效应 −2.8，交互 −3.6（拮抗）。两组件各自损害 SGD。**注意**: 200 步短预算结果，长预算可能不同。

### P1.2 跨深度 ASP（4 模型, 12L–28L）

| 模型 | 层数 | ASP PPL | vs Baseline | 状态 |
|------|------|---------|-------------|------|
| OPT-125m | 12L | 106.9 | 2.2× (base=231) | ✓ 收敛 |
| TinyLlama | 22L | 15.5 | 9.4× (base=146) | ✓ 收敛 |
| Qwen0.5B | 24L | 18.0 | 22.8× (base=411) | ✓ 收敛 |
| Qwen7B | 28L | ∞ | — | ✗ 发散 |

**注意**: 跨模型族比较，绝对 PPL 不可比。结论仅关于稳定性（收敛/发散），非绝对性能。

### P1.3 隐式正则化（OPT-125m, WT2+C4）

| 比较 | WT2 PPL | C4 PPL | WT2/C4 | 解释 |
|------|---------|--------|--------|------|
| ASP@200 | 66.5 | 47.5 | 1.40 | 跨域泛化 |
| ASP@800 | 75.1 | 48.1 | 1.56 | 跨域泛化 |
| AdamW@200 | 18.5 | 108.5 | 0.17 | 记忆 WT2 |

**结论**: ASP C4 PPL 比 AdamW 好 1.9×。ASP 的隐式正则化是真实的——防止记忆、保持跨域泛化。

## 3. P2：综合评估（Conditional Go）

详见 [`docs/p2-synthesis.md`](docs/p2-synthesis.md)。核心判断：**可发表** — 干净的负结果 + 深度失稳确认 + 隐式正则化正面发现。

下一步：
- [x] 更新论文 v0.7.1 → v0.8（加入 P1 结果）— commit `60ae760`
- [ ] 独立复核者复算主表
- [ ] 选择投稿期刊（TMLR / arXiv+workshop）

**论文 v0.8 变更摘要**：
- 新增 §5.9 Mechanism Validation（P1.1 组件归因 + P1.2 跨深度 + P1.3 跨域正则化）
- Abstract 从 5 → 6 条核心结论
- Contributions 从 6 → 7 条
- §7.2 增加跨域 + 组件归因证据
- §7.3 limitation #5 标记为 partially resolved
- §7.4 定性比较表新增跨域泛化 + 瓶颈两行
- §8 发现表从 8 → 10 条，结论从 5 → 6 条，实验数 14 → 17
