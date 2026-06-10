# alternating-optimization-lora 项目推进路线

## 当前基线 (Phase 2+3+4 完成)
- 核心框架: AltOptFramework, LoRABaseline ✅
- 基础设施: Trainer, Profiling, PEFT, Checkpoint, Evaluator ✅
- 规模化: ModelLoader (7B+), DeepSpeed ZeRO-1/2/3 ✅
- 消融: RQ1-RQ6 系统性消融实验框架 ✅
- 可视化: 6 种图表类型, 一键生成 ✅
- 文档: 3 份实验报告 + 数学框架文档 ✅
- 测试: 67 tests passing ✅

---

## 计划执行清单

### Step 1: 端到端集成验证 (GPT-2 最小可行)
**状态**: ✅ — 报告 #001

### Step 2: 修复 Protocol B/C/D 集成路径
**状态**: ✅ — 报告 #001

### Step 3: 统一评分协议实战验证
**状态**: ✅ — 报告 #001

### Step 4: FLOPs 精确计数接入 (fvcore)
**状态**: ✅ — 报告 #001

### Step 5: 小规模对比实验 (GPT-2, 50 steps)
**状态**: ✅ — 报告 #001

### Step 6: 实验报告撰写 (报告 #001)
**状态**: ✅

### Step 7: Round 2 基础设施完善
**状态**: ✅

### Step 8: Round 3 GPT-2 Conv1D + 复现性
**状态**: ✅ — 报告 #002

### Step 9: Round 4 OPT-125m 干净 2×2 + ALS:SGD 消融
**状态**: ✅ — 报告 #002

### Step 10: Phase 2 — 规模化基础设施
**状态**: ✅
- model_utils.py: 7B+ 模型加载 (bf16/int4/gradient checkpointing)
- deepspeed_engine.py: ZeRO-1/2/3 集成, 显存分析
- llama2_7b.yaml: 7B 实验配置
- trainer.py: DeepSpeed 训练循环

### Step 11: Phase 3 — 消融实验框架
**状态**: ✅
- ablation.py: RQ2-RQ6 独立消融 + run_all_ablation()
- analysis.py: RQ1 2×2 析因分析

### Step 12: Phase 4 — 可视化工具包
**状态**: ✅
- visualization.py: 6 种图表 + generate_all_plots()

### Step 13: 实验报告 #003
**状态**: ✅

---

## 后续规划

### Phase 5: 数据产出
- [ ] 运行 ablation.py 产生消融数据 (`python experiments/ablation.py gpt2`)
- [ ] 基于消融结果生成可视化图表
- [ ] 下载 Llama-2-7B 模型

### Phase 6: 7B 规模化实验
- [ ] Llama-2-7B 2×2 析因 (DeepSpeed ZeRO-2)
- [ ] 7B 规模 ALS:SGD 最优比验证
- [ ] 与 GPT-2/OPT 结果的一致性分析

### Phase 7: 扩展到更大规模
- [ ] 13B/70B 模型支持 (ZeRO-3)
- [ ] 多节点 / DeepSpeed 优化
- [ ] 多数据集验证

---

*Last updated: 2026-06-10*
