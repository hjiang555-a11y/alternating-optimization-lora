# ASP vs LoRA: A Quasi-Factorial Comparison for LLM Post-Training

> **状态**: 论文 v0.8 — Round 6 **Major Revision**. P1 机制实验已完成，P2 Conditional Go.
> **核心**: 准析因 2×2 比较框架，8 架构实测，深度 24–28 层失稳转变，组件归因 + 跨域正则化验证
> **路线图**: [todo.md](todo.md) — P0+P1 已完成，待独立复核 + 选刊

---

## 文档导航

| 文档 | 说明 |
|------|------|
| **[论文 v0.8](paper/paper_draft_v0.7.md)** | Canonical draft（1022 行，10 项发现，17 实验，6 核心结论） |
| **[实验注册表](docs/experiment-registry.md)** | 8 架构 × 4 协议 × 50–800 步矩阵，含证据标签 |
| **[主张→证据映射](docs/claims-audit.md)** | 14 项核心主张，含 observed/transcribed/inferred/predicted 标签 |
| **[当前路线图](todo.md)** | P0+P1 已完成 / P2 Conditional Go / 待独立复核 |
| **[综合评估](docs/p2-synthesis.md)** | P1 校准 + 收敛匹配 + Conditional Go 决策 |
| **[全部发现](docs/all-findings.md)** | 21 项发现，证据强度 + 论文状态标记 |
| **[公平比较方法论](docs/fair_comparison_methodology.md)** | 准析因设计的核心方法论 |
| **[数学分析](docs/math-analysis.md)** | ALS 重建损失、收敛理论、PAC-Bayes 分析 |
| **[深度失稳因果理论](docs/causal_depth_boundary.md)** | SCM 框架下的残差干预传播模型 |
| **[机制笔记](docs/mechanism-notes.md)** | 组件归因设计、隐式正则化验证缺口 |
| **[P0.2 可行性报告](docs/p0.2-feasibility.md)** | HellaSwag + 参数量匹配实验评估 |
| **[v3.4 LaTeX](paper/paper_v3.4.tex)** / **[PDF](paper/paper_v3.4.pdf)** | 前一版本 LaTeX 源码及编译 PDF |
| **[修订计划](paper/revision_plan.md)** | Round 1 评审应对方案 |

### 子目录

| 目录 | 内容 | 文件数 |
|------|------|--------|
| [`docs/archive/`](docs/archive/) | 历史评分、早期实验报告、已被取代的评估 | 24 |
| [`docs/reference/`](docs/reference/) | 算法详解、协议实现、评估标准（教育性文档） | 9 |
| [`paper/reviews/`](paper/reviews/) | Round 1–6 同行评审记录 | 6 |
| [`runs/p1.*/`](runs/) | P1 实验产物（组件归因、跨深度、正则化） | 3 实验 |

---

## 核心发现

| # | 发现 | 证据强度 | 来源 |
|---|------|---------|------|
| 1 | LoRA r=8 在 $L/d_h \leq 0.035$ 架构上达到充分性平台 | 5/5 模型族，中英跨语言，100–1600 步 | §5.2, §5.7 |
| 2 | 全秩微调在小数据上灾难性过拟合（PPL=1.25 但 HellaSwag −3.2pp） | N=3, 7B 规模, 3 下游任务 | §5.6.2–5.6.4 |
| 3 | **秩充分性定律** $r_{\min} = \eta \cdot L/d_h$ ($\eta \approx 230$) | SmolLM2 10 点精细校准，η±8% | §6.6–6.8 |
| 4 | ASP 在 ≤24 层收敛，≥28 层失稳 | 8/8 架构确认，11 次 7B 尝试 | §5.6 |
| 5 | ASP 隐式正则化：C4 跨域验证（1.9× 优于 AdamW） | 收敛匹配，WT2+C4 双数据集 | §5.4, §5.9.3 |
| 6 | ASP 组件归因：ALS 为主要瓶颈（+3.1 PPL） | 4 条件嵌套消融，N=3 seeds | §5.9.1 |
| 7 | ASP 收敛质量由预训练质量主导，非深度 | 4 模型 12–28L | §5.9.2 |
| 8 | 低秩 ALS 始终负收益（7/7 比较） | 3 模型，100–800 步 | §5.8 |
| 9 | M-index ($M<1$) 为轻量记忆化诊断 | 跨尺度 0.5B→7B | §6.7 |
| 10 | 7B B/D 8.3× PPL 差异由过拟合驱动，非秩不足 | 参数匹配基线确认 | §5.7 |

---

## 快速开始

```bash
pip install -e ".[dev]"
python experiments/runner.py experiments/configs/base.yaml
python experiments/analysis.py logs/
pytest tests/  # 122 passed, 2 pre-existing failures (bitsandbytes GPU dependency)
```

---

## 仓库结构

```
├── paper/paper_draft_v0.7.md    # 唯一论文草稿 (v0.8, 1022 行)
├── paper/paper_v3.4.tex         # 前一版本 LaTeX 源码
├── paper/paper_v3.4.pdf         # 前一版本编译 PDF
├── paper/revision_plan.md       # Round 1 评审应对方案
├── docs/
│   ├── claims-audit.md          # 主张→产物可追溯性
│   ├── experiment-registry.md   # 全实验矩阵
│   ├── p2-synthesis.md          # P2 综合评估
│   ├── all-findings.md          # 21 项发现汇总
│   ├── archive/                 # 历史快照（24 文件）
│   └── reference/               # 教育性文档（9 文件）
├── altopt/                      # 核心框架（ALS, SGD, Perturb, LoRA, trainer）
├── experiments/                 # 实验脚本（含 _p1.1/_p1.2/_p1.3）
├── tests/                       # 122 测试
└── runs/                        # 数据产物（p1.1/p1.2/p1.3 结果已入库）
```

## License

MIT
