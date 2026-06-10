# 实验报告 #003: RQ1-RQ6 系统性消融实验框架 + 7B 规模化基础设施

**日期**: 2026-06-10  
**实验类型**: 框架建设 + 消融实验设计  
**新增模块**: `model_utils.py`, `deepspeed_engine.py`, `ablation.py`, `visualization.py`  
**模型**: GPT-2 (124M) — 快速验证; Llama-2-7B 配置就绪  

---

## 1. 本阶段目标

在完成 OPT-125m 2×2 析因实验（报告 #002）后，本阶段实现两个核心目标：

1. **Phase 2 — 规模化基础设施**: 使框架支持 7B+ 模型与 DeepSpeed ZeRO 分布式训练
2. **Phase 3 — RQ1-RQ6 系统性消融**: 设计并实现覆盖全部 6 个研究问题的消融实验框架
3. **辅助工具**: 统一可视化工具包，直接对接消融结果 JSON

---

## 2. 新增模块总览

| 模块 | 文件 | 功能 | 状态 |
|------|------|------|------|
| 模型加载工具 | `altopt/model_utils.py` | 7B+ 模型加载（bf16/fp16/int4/int8）、gradient checkpointing、多 GPU 设备映射、显存估算 | ✅ |
| DeepSpeed 引擎 | `altopt/deepspeed_engine.py` | ZeRO-1/2/3 集成、bf16 混合精度、checkpoint、显存分析 | ✅ |
| 消融实验框架 | `experiments/ablation.py` | RQ2-RQ6 独立消融函数 + `run_all_ablation()` 一键运行 | ✅ |
| 可视化工具包 | `experiments/visualization.py` | 6 种图表类型（训练曲线、Pareto 前沿、热力图、消融柱状图等） | ✅ |
| 7B 实验配置 | `experiments/configs/llama2_7b.yaml` | Llama-2-7B 2×2 析因 + DeepSpeed 参数 | ✅ |

---

## 3. Phase 2: 规模化基础设施

### 3.1 模型加载 (`altopt/model_utils.py`)

```
load_model_and_tokenizer(ModelLoadConfig(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    dtype="bf16",
    device_map="auto",
    gradient_checkpointing=True,
    use_flash_attention=True,
))
```

关键能力:
- **多种 dtype**: bf16 / fp16 / fp32 / int8 / int4
- **自动设备映射**: `accelerate` 的 `device_map="auto"` 跨 2× RTX 5090
- **Gradient checkpointing**: 激活值重计算，降低 40-60% 显存
- **FlashAttention-2**: PyTorch 2.0+ 原生支持
- **显存估算**: `estimate_training_memory_gb()` 给出各组件（weights/optimizer/gradients/activations）显存明细

### 3.2 DeepSpeed 集成 (`altopt/deepspeed_engine.py`)

```
ZeRO 内存对比 (Llama-2-7B, bf16, 2×32GB GPU):

┌─────────────┬──────────┬──────────┬──────────┬──────────┐
│             │ No ZeRO  │ ZeRO-1   │ ZeRO-2   │ ZeRO-3   │
├─────────────┼──────────┼──────────┼──────────┼──────────┤
│ Weights     │ 14 GB    │ 14 GB    │ 14 GB    │  7 GB    │
│ Gradients   │ 14 GB    │ 14 GB    │  7 GB    │  0.7 GB  │
│ Opt States  │ 28 GB    │ 14 GB    │ 14 GB    │  1.4 GB  │
│ Activations │  8 GB    │  8 GB    │  8 GB    │  8 GB    │
│ **Total**   │**64 GB** │**50 GB** │**43 GB** │**17 GB** │
└─────────────┴──────────┴──────────┴──────────┴──────────┘
```

- **默认 ZeRO-2**: 对 7B + 2 GPU 是 sweet spot
- **bf16 混合精度**: RTX 5090 原生支持
- **显存估算 API**: `DeepSpeedEngine.estimate_memory()`

### 3.3 Trainer 集成

`AltOptTrainer` 新增 `_train_deepspeed()` 方法：在 `use_deepspeed=True` 时自动切换到 DeepSpeed 引擎，保持与标准训练路径相同的 hook 接口。

---

## 4. Phase 3: RQ2-RQ6 消融实验设计

### RQ2: 效率边界 (Efficiency Frontier)

```
run: rq2_efficiency_frontier(model_name="gpt2")
```

| 方案 | 内容 |
|------|------|
| 协议 | Protocol A (AltOpt full-rank) vs Protocol B (AdamW full-rank) |
| 变量 | 总 FLOPs 预算（通过多个 checkpoint 采样） |
| 指标 | 相同 FLOPs 下的 final loss、perplexity |
| 产出 | Pareto 前沿曲线 |

### RQ3: 损失地形交互 (Perturbation Effect)

```
run: rq3_perturbation_effect(model_name="gpt2")
```

| 方案 | 内容 |
|------|------|
| 协议 | AltOpt with perturbation vs AltOpt without perturbation |
| 变量 | 扰动阶段的有无 |
| 指标 | 扰动前后的 loss drop、最终 perplexity 差值 |
| 产出 | 扰动事件标注的 loss 曲线 |

### RQ4: 泛化能力 (Generalization Gap)

```
run: rq4_generalization(model_name="gpt2")
```

| 方案 | 内容 |
|------|------|
| 协议 | 四协议全部运行 |
| 变量 | 优化器 × 参数形态 |
| 指标 | eval_loss - train_loss（泛化差距） |
| 产出 | 泛化差距柱状图 |

### RQ5: 协同可能 (Synergy)

```
run: rq5_synergy(model_name="gpt2")
```

| 方案 | 内容 |
|------|------|
| 协议 | Protocol C (LoRA+AltOpt) vs Protocol D (LoRA+AdamW) |
| 变量 | 优化器类型（均在 LoRA 条件下） |
| 指标 | perplexity 差值（C_delta = ppl_D - ppl_C） |
| 产出 | 协同优势量化 |

### RQ6: ALS:SGD 最优比

```
run: rq6_als_sgd_ratio(model_name="gpt2")
```

| 方案 | 内容 |
|------|------|
| 协议 | Protocol A，扫描 4 个比例 |
| 比例 | 1:10, 1:20, 1:50, 1:100 |
| 指标 | 各比例下的 final perplexity + FLOPs |
| 产出 | 双 Y 轴柱状图（perplexity + FLOPs） |

### 一键运行

```bash
python experiments/ablation.py gpt2 runs/ablation/
# 输出: runs/ablation/ablation_results.json
```

---

## 5. 可视化工具包

`experiments/visualization.py` 提供 6 种图表：

| 图表 | 函数 |
|------|------|
| 训练 loss 曲线 | `plot_training_curves()` |
| FLOPs-Perplexity Pareto | `plot_pareto_frontier()` |
| 2×2 析因热力图 | `plot_factorial_heatmap()` |
| ALS:SGD 消融柱状图 | `plot_ratio_ablation()` |
| 泛化差距对比 | `plot_generalization_gap()` |
| 扰动效应标注 | `plot_perturbation_effect()` |

一键生成所有图表：
```bash
python experiments/visualization.py runs/ablation/ablation_results.json figures/
```

---

## 6. 测试结果

```
67 passed in 3.38s
```

所有现有测试无回归。新增模块通过 import 验证和端到端 smoke test。

---

## 7. 下一步计划

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 运行消融实验（GPT-2） | `python experiments/ablation.py gpt2` 产生实际数据 |
| P0 | 下载 Llama-2-7B | 需要 HF token 和 ~14GB 磁盘 |
| P1 | 7B 规模 DeepSpeed 实验 | `python experiments/runner.py experiments/configs/llama2_7b.yaml` |
| P1 | 基于消融数据生成图表 | 验证可视化工具包在所有 RQ 上的输出 |
| P2 | 多数据集验证 | C4、The Pile 等更大规模数据集 |

---

*Last updated: 2026-06-10*
