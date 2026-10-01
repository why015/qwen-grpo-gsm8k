# Small LLM 数学推理后训练实验记录

## Quickstart

从 clone 到复现完整评测流程：

```bash
# 1. 环境
git clone https://github.com/why015/qwen-grpo-gsm8k.git
cd qwen-grpo-gsm8k
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 2a. 下载 GSM8K 原始数据（国内可先 export HF_ENDPOINT=https://hf-mirror.com）
python prepare_gsm8k.py

# 2b. 构建训练/验证切分
python deep_project/build_data.py

# 3. 运行 verifier 单元测试
python tests/test_verifier.py

# 4. 训练 SFT
python train_sft.py

# 5. 训练 GRPO
GRPO_SEED=42 python train_grpo.py

# 6. 评测
python deep_project/evaluate.py --data deep_project/data/processed/development500.jsonl --model runs/<run_id> --output runs/<run_id>/eval_dev500.jsonl

# 7. 查看产物
cat runs/<run_id>/manifest.json
cat results/runs.csv
```

## 环境变量

| 变量 | 默认值 | 作用 |
| :--- | :--- | :--- |
| `GRPO_SEED` | `42` | 随机种子 |
| `LOSS_TYPE` | `dapo` | `grpo` / `dapo` / `dr_grpo` |
| `SCALE_REWARDS` | `group` | `group` / `none` |
| `EPSILON_HIGH` | `None` | DAPO 上界 |
| `REWARD_VERSION` | `r1` | R1 / R2 / R3 |
| `REWARD_MODE` | `strict` | `strict`（默认）/ `lenient`（复现历史 v3） |
| `STOP_AT_STEP` | 不设 | 提前停止 |

## 产物结构

```text
runs/
└── <run_id>/
    ├── manifest.json
    ├── audit_groups.jsonl
    └── adapter_*.safetensors

results/
├── runs.csv
├── seed_42/
├── seed_1234/
├── seed_2026/
├── baseline_sft/
└── baseline_base/
```

## 已知限制

- 仓库不含模型权重和 GSM8K 原始数据。
- 需要 8GB 以上显存。
- TRL 1.14 默认 `loss_type='dapo'`。

**PowerShell 环境变量写法**（Windows 用户）：

```powershell
$env:GRPO_SEED = "42"
$env:LOSS_TYPE = "dr_grpo"
$env:SCALE_REWARDS = "none"
$env:REWARD_MODE = "strict"
python train_grpo.py
```

## 一、项目概述

本项目研究小规模语言模型在数学推理任务上的后训练行为。关注点不只是最终准确率，还包括以下问题：

- SFT 能带来多少整体提升？
- SFT 是否在提升整体准确率的同时引入局部能力退化？
- 这些退化是否可以通过 Prompting 恢复，还是代表能力真正丢失？
- Prompt 干预在恢复部分样本的同时，是否引入新的错误？
- GRPO 是否能在 SFT 基础上进一步改善，并修复部分 Prompting 无法修复的退化？
- Reward 设计、Group Size、KL 正则化等如何影响 GRPO 的行为？

项目主线是 **SFT → GRPO 两阶段后训练**。Prompting 实验作为诊断工具，用来刻画 SFT 退化的性质，而不是主要优化路线。

## 二、实验设置

### 2.1 模型与硬件

| 项目 | 值 |
| :--- | :--- |
| Base Model | `Qwen2.5-0.5B-Instruct` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8188 MiB |
| 驱动版本 | `596.49` |
| Python | `3.12.14` |
| PyTorch | `2.13.0+cu130` |
| CUDA | `13.0` |
| cuDNN | `92000` |
| Transformers | `5.17.0` |
| PEFT | `0.21.0` |
| TRL | `1.14.0` |
| Datasets | `5.0.1` |

### 2.2 数据集配置

| 项目 | 值 |
| :--- | :--- |
| 原始 GSM8K train | 7473 条 |
| 训练集 | 6973 条 |
| 验证集 | 500 条 |
| 训练与验证 ID 交集 | 0 条 |
| 评测规模 | 200 条（验证集前 200 条） |
| 数据字段 | `id` / `prompt` / `answer` / `sft_response` |

### 2.3 Prompt 模板

以下模板后接题目原文：

```text
请解答下面的数学应用题，给出简短计算过程。最后一行以「答案：」开头，冒号后只写结果数字，不写单位。
```

### 2.4 Verifier 解析规则

| 解析模式 | 匹配格式或规则 |
| :--- | :--- |
| 模式 1 | `#### X`（GSM8K 官方格式） |
| 模式 2 | `答案：X` / `The answer is X` / `final answer is X` |
| 模式 3 | 原文显示为独立的 `X`，具体格式待确认 |
| 兜底 | 在最后一行搜索数字，不在全文搜索 |

**数值归一化规则**：

- 去除 `$` 符号。
- 去除逗号。
- 去除末尾句号。
- 转换为 `float`；如果为整数，则转换为 `int`。

### 2.5 解码配置

| 参数 | 值 |
| :--- | :--- |
| `do_sample` | `False`（Greedy） |
| `max_new_tokens` | `1024` |
| `pad_token_id` | `eos_token_id` |

### 2.6 LoRA / SFT 配置

| 参数 | 值 |
| :--- | :--- |
| LoRA rank | `8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.0` |
| Target Modules | 全部线性层：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` |
| SFT 训练 Epoch | `1` |
| SFT 学习率 | `1e-4` |
| SFT Batch Size | `1` |
| SFT 梯度累积 | `4` |
| SFT `max_length` | `512` |
| SFT 训练时长 | 2052 秒（约 34 分钟） |
| SFT 训练步数 | 约 1743 步 |

> **配置说明（重要）**：本项目的 "GRPO v3" 基于 TRL 1.14.0 的 `GRPOConfig`。经审计确认，该版本的 `loss_type` 默认值为 `dapo`（token-level aggregation），`scale_rewards` 默认为 `group`（除以 group 标准差），但 `epsilon_high` 默认为 `None`（未启用 DAPO 的 asymmetric clipping）。
>
> 因此，本文中的 "GRPO v3" 更准确地说是 **GRPO 与 DAPO 的混合配置**：采用 DAPO 的 token-level loss aggregation，但保留 GRPO 的对称裁剪和 group std normalization。
>
> - **TRL 1.14 默认值**：`configs/trl_1_14_defaults_probe.json`
> - **v3 实际使用配置**：`configs/grpo_v3_actual.json`（含 Git SHA 与重构说明）
>
> 后续 GRPO / DAPO / Dr.GRPO 的对比消融将显式指定 `loss_type`、`scale_rewards`、`epsilon_high` 等参数以区分三种算法。

## 三、Base 模型基线

Base 模型在 200 条验证集样本上的结果如下：

| 指标 | 值 |
| :--- | :---: |
| 准确率 | **39.0%** |
| 正确题数 | 78 / 200 |
| 平均输出 Token 数 | 247.5 |
| 解析失败数 | 6 |

该结果作为后续 SFT 和 GRPO 的统一 Baseline，用于比较各阶段表现。

## 四、监督微调

### 4.1 Base 与 SFT 整体对比

| 阶段 | 准确率 | 平均输出 Token 数 | 解析失败数 |
| :--- | :---: | :---: | :---: |
| Base | 39.0%（78 / 200） | 247.5 | 6 |
| SFT | **48.5%（97 / 200）** | 82.6 | 0 |
| 变化 | **+9.5 个百分点** | −164.9 | — |

### 4.2 SFT 训练日志

| 指标 | 值 |
| :--- | :---: |
| 训练时长 | 2052 秒 |
| 最终训练 Loss | 0.5256 |
| 最终 `mean_token_accuracy` | 0.8451 |
| 最终 Epoch | 1.0 |

从整体准确率看，SFT 明显提升了数学推理任务表现。模型学到了更贴近 GSM8K 标准解法的推理格式，平均输出长度从 Base 的 247.5 Token 压缩到约 82.6 Token。

但是，总体准确率掩盖了样本级变化。逐题比较 Base 和 SFT 的输出后发现，部分原本做对的题目在 SFT 后做错了。这说明 **整体提升与局部退化可以同时存在**。

## 五、SFT 退化分析

定义 **退化样本（Regression Case）** 为：Base 正确、SFT 错误的样本。

### 5.1 Base 与 SFT 的逐题转移矩阵

| Base | SFT | 数量 |
| :---: | :---: | :---: |
| 对 | 对 | 50 |
| 错 | 错 | 75 |
| 错 | 对 | 47 |
| 对 | 错 | **28** |
| **合计** | — | **200** |

其中，Base 正确、SFT 错误的 28 道题定义为 **SFT Regression Case**。

### 5.2 退化样本错误类型

| 错误类型 | 数量 |
| :--- | :---: |
| 纯算术错误 | 4 |
| 推理链不完整 | 5 |
| 数量关系理解错误 | 4 |
| 单位 / 比例错误 | 4 |
| 遗漏约束条件 | 4 |
| 完全误读题目 | 7 |
| 合计（原文记录） | 28 |

**核心观察**：SFT 退化不只是算术错误。相当一部分错误来自题意理解、数量关系、比例与单位，以及约束建模。

这说明 SFT 学到的不只是计算能力，也包含一套特定的推理策略。当这套策略与具体题目不匹配时，可能出现样本级退化。

这一节回答的问题是：**SFT 改善了什么，又打破了什么？**

- 改善的是整体推理格式和输出风格。
- 打破的是部分原本答对题目的推理路径。

## 六、SFT 退化是否可通过 Prompting 恢复

研究问题是：这 28 道退化样本是能力真正丢失，还是默认推理策略没有调动出原本的能力？

### 6.1 Prompt 干预设计

在 28 道退化样本上测试四组 Prompt：

| Prompt | 干预内容 |
| :--- | :--- |
| Original | 无额外强调，作为基准 |
| Prompt A | 强调检查计算、完成完整推理过程 |
| Prompt B | 强调注意数量关系、注意约束条件 |
| Prompt C | 同时包含 A 与 B 的全部四点 |

### 6.2 退化样本恢复结果

| Prompt | 恢复数 | 恢复率 |
| :--- | :---: | :---: |
| Original | 0 / 28 | 0% |
| Prompt A | 2 / 28 | 7.1% |
| Prompt B | 3 / 28 | 10.7% |
| **Prompt C** | **9 / 28** | **32.1%** |

### 6.3 分析用工作定义

- **Soft Regression**：Prompt C 可以恢复的 9 道题。
- **Hard Regression**：Prompt C 无法恢复的 19 道题。

> 这只是分析用的工作定义。Hard Regression 不应理解为“能力永久丢失”，也不代表相关能力在模型参数中一定不存在。它仅表示在当前 Prompt 干预范围内无法恢复。

## 七、Prompt 恢复与新退化

### 7.1 完整评测集结果

将 Prompt C 应用到完整 200 条评测样本上：

| 阶段 | 准确率 | 平均输出 Token 数 |
| :--- | :---: | :---: |
| SFT 原 Prompt | **48.5%（97 / 200）** | 82.6 |
| Prompt C | 47.5%（95 / 200） | 85.4 |
| 变化 | **−1.0 个百分点** | +2.8 |

### 7.2 逐题转移矩阵

| SFT 原 Prompt | Prompt C | 数量 |
| :---: | :---: | :---: |
| 对 | 对 | 80 |
| 错 | 错 | 88 |
| 错 | 对 | 15 |
| 对 | 错 | 17 |
| **合计** | — | **200** |

净变化为减少 2 道正确题，即准确率下降 **1.0 个百分点**。

该结果不应简单描述为“Prompt C 无效”。更准确的表述是：**Prompt C 改变了模型的推理行为，但没有稳定提升总体能力**。

它同时产生恢复和新退化，两者在总体上大致抵消，略微偏负。

## 八、错误类型恢复分析

Prompt C 在 28 道退化样本中恢复了 9 道。按错误类型统计如下：

| 错误类型 | 总数 | 恢复数 | 恢复率 |
| :--- | :---: | :---: | :---: |
| 纯算术错误 | 4 | 3 | 75% |
| 推理链不完整 | 5 | 2 | 40% |
| 数量关系 | 4 | 2 | 50% |
| 比例 / 单位 | 4 | 1 | 25% |
| 遗漏约束 | 4 | 0 | 0% |
| 完全误读 | 7 | 1 | 14% |
| 合计（原文记录） | 28 | 9 | 32.1% |

这组数据表明，Prompt C 对算术计算和部分数量关系类错误有较高的恢复能力，对推理不完整类错误有一定帮助，但对遗漏约束和完全误读类错误的恢复能力较弱。

需要强调的是，这是基于当前 28 道样本的经验观察，不应写成普适规律。样本量较小，各类型恢复率差异也可能受具体题目难度影响。

## 九、Prompt C 引入的新错误

SFT 正确、Prompt C 错误的新退化共 17 道，分类如下：

| 错误类型 | 数量 |
| :--- | :---: |
| 纯计算错误 | 4 |
| 题意理解错误 | 4 |
| 推理完全崩塌 | 5 |
| 过度推理 | 1 |
| 逻辑关系错误 | 2 |
| 巧合 / 不稳定 | 1 |
| **合计** | **17** |

其中，推理崩塌与过度推理共 6 道，占 17 道中的约三分之一。

需要注意的是，不应简单得出“Prompt C 因为输出变长而导致过度思考”的结论，因为后面的推理长度分析不支持这一解释。

**核心观察**：Prompt C 不仅能救回错误，也可能把原本正确的推理轨迹推向错误。它改变了模型处理题目的方式，这种改变在部分题目上带来恢复，在另一些题目上带来新的失败。

## 十、推理长度分析

对四组转移分别统计原 Prompt 和 Prompt C 的平均输出 Token 数：

| 转移组 | 数量 | 原 Prompt Token 数 | Prompt C Token 数 | 差值 |
| :--- | :---: | :---: | :---: | :---: |
| SFT 错 → C 对 | 15 | 77.4 | 89.5 | **+12.1** |
| SFT 对 → C 错 | 17 | 96.7 | 99.6 | +2.9 |
| SFT 对 → C 对 | 80 | 72.8 | 73.2 | +0.4 |
| SFT 错 → C 错 | 88 | 89.7 | 93.1 | +3.4 |
| **总体** | **200** | **82.6** | **85.4** | **+2.8** |

**关键观察**：最明显的推理长度增长出现在成功救回的样本中，即“SFT 错 → C 对”组，增加 12.1 Token；而被 Prompt C 导致退化的样本仅增加 2.9 Token。

因此，当前数据不支持“Prompt C 的副作用主要来自输出变长或过度思考”这一解释。

更合理的表述是：**显式推理指令可能改变推理轨迹**。

- 对部分原本推理不足的样本，这种改变可能有帮助。
- 对另一些样本，它可能改变原本正确的题意表征或推理路径。

这里采用“提示”“表明”“可能”等保守表述，不把相关性写成已证明的因果关系。

## 十一、阶段性发现

1. **SFT 提升整体准确率**：从 39.0% 提升到 48.5%，增加 **9.5 个百分点**。

2. **整体提升伴随局部退化**：SFT 引入了 **28 个样本级退化**。

3. **部分退化可通过 Prompting 恢复**：28 个退化样本中，9 个可以通过显式推理 Prompting 恢复，其余 19 个在当前干预范围内无法恢复。

4. **Prompt C 同时带来恢复与新退化**：在完整 200 条评测样本上救回 15 道，同时导致 17 道原本正确的题目退化，净减少 2 道正确题。

5. **副作用不能简单归因于输出变长**：救回样本的 Token 增长明显高于退化样本。

6. **SFT 退化涉及多个因素**：包括推理轨迹、题意表征和推理充分性。这为下一步 GRPO 提供了动机：通过训练信号而非 Prompt 干预，探索是否能更稳定地修复 SFT 退化。

# 十二、GRPO 后训练（完整修订版）

在 SFT 基础上，本项目进一步使用 GRPO 进行强化学习后训练。

当前已经完成两组主要 GRPO 实验：GRPO v2 与 GRPO v3。两组实验在训练步数、num_generations、学习率、KL 系数以及最大生成长度等多个参数上均存在差异，因此二者主要用于比较不同训练配置下的整体行为，而不能视为严格的单变量消融实验。

后续将通过控制变量实验进一步分析 Reward、Group Size、KL 系数与 Sampling Temperature 的独立影响。

## 12.1 GRPO 训练配置

| 参数 | GRPO v2 | GRPO v3 |
| --- | --- | --- |
| max_steps | 300 | 1000 |
| num_generations | 4 | 8 |
| learning_rate | 1e-5 | 5e-6 |
| beta（KL 系数） | 0.01 | 0.04 |
| max_completion_length | 384 | 256 |
| gradient_accumulation_steps | 4 | 8 |
| 训练时长 | 2422 秒（约 40 分钟） | 10270 秒（约 2.85 小时） |
| 训练期间最终 Reward | 0.35 | 0.30 |
| 训练期间最终 train_loss | — | 0.008313 |
| KL（前 10 步均值） | 4.7e-04 | 5.8e-05 |
| KL（后 10 步均值） | 1.9e-03 | 2.2e-03 |
| KL（全期均值） | 1.6e-03 | 1.3e-03 |
| KL 中位数 | 1.4e-03 | 1.3e-03 |
| 最后一步 KL | 1.5e-03 | 1.4e-03 |

> **术语与配置说明**：本项目的 "GRPO v3" 基于 TRL 1.14 的 `GRPOConfig`。该版本 TRL 的 `loss_type` 默认为 `dapo`（token-level aggregation），`scale_rewards` 默认为 `group`（除以 group 标准差），但 `epsilon_high` 默认为 `None`（未启用 DAPO 的 asymmetric clipping）。因此本文中的 "GRPO v3" 更准确地说是 GRPO 与 DAPO 的混合配置。后续的 GRPO / DAPO / Dr.GRPO 消融实验将显式指定 `loss_type`、`scale_rewards`、`epsilon_high` 等参数以区分三种算法。

## 12.2 GRPO 基线结果（Development Set 上）

| 阶段 | 准确率 | 平均输出 Token 数 | 解析失败数 |
| --- | --- | --- | --- |
| Base | 39.0%（78 / 200） | 247.5 | 6 |
| SFT | 48.5%（97 / 200） | 82.6 | 0 |
| SFT + Prompt C | 47.5%（95 / 200） | 85.4 | 0 |
| GRPO v2 | 46.5%（93 / 200） | 86.1 | 0 |
| GRPO v3 | 51.0%（102 / 200） | 91.5 | 1 |

目前结果显示：

- GRPO v2 相比 SFT 下降 2.0 个百分点；
- GRPO v3 相比 SFT 提升 2.5 个百分点；
- GRPO v3 达到当前最高准确率 51.0%；
- 从 Base 到 SFT 再到 GRPO v3，准确率由 39.0% 到 48.5% 再到 51.0%。

这表明 GRPO 的效果对训练配置较为敏感。并不是进行 RL 后训练就一定能够提升性能：v2 出现退化，而 v3 则获得进一步增益。

需要特别说明的是，上述数字基于 200 条 development set。后续在独立 held-out 300 条上的评测显示 GRPO v3 的优势未能复现（详见 12.3）。

## 12.3 GRPO 泛化检查（Held-out 300 条）

在完成 200 条 development set 上的初步评测后，为检验 GRPO v3 的提升是否具有泛化能力，我们从验证集中保留了一个从未参与任何配置选择的 held-out 集（300 条），对 Base、SFT、GRPO v3 进行了独立评测。

### 12.3.1 三个数据集上的对比

| 数据集 | Base | SFT | GRPO v3 |
| --- | --- | --- | --- |
| Dev-200 | 39.0% | 48.5% | 51.0% |
| Held-out-300 | 35.3% | 45.7% | 43.7% |
| Combined-500 | 36.8% | 46.8% | 46.6% |

#### 关键观察

- Base 与 SFT 在 dev 和 held-out 上的相对提升方向一致（SFT 稳健提升约 10 个百分点）；
- GRPO v3 在 dev 上相比 SFT 提升 +2.5pp，但在 held-out 上下降 −2.0pp；
- 合并 500 条后，GRPO v3 相比 SFT 净变化为 −0.2pp。

### 12.3.2 逐题 transition 分析

| Transition | 数据集 | 修复 | 搞坏 | 净变化 | 修复/搞坏比 |
| --- | --- | --- | --- | --- | --- |
| Base → SFT | Dev-200 | 47 | 28 | +19 | 1.68 |
| Base → SFT | Held-out-300 | 68 | 37 | +31 | 1.84 |
| Base → SFT | Combined-500 | 115 | 65 | +50 | 1.77 |
| SFT → GRPO v3 | Dev-200 | 16 | 11 | +5 | 1.45 |
| SFT → GRPO v3 | Held-out-300 | 12 | 18 | −6 | 0.67 |
| SFT → GRPO v3 | Combined-500 | 28 | 29 | −1 | 0.97 |

#### 关键观察

- Base → SFT 在 dev 和 held-out 上修复/搞坏比均接近 1.7~1.8，非常一致；
- SFT → GRPO v3 在 dev 上修复多于搞坏（1.45），但在 held-out 上完全反转（0.67）；
- 合并 500 条后，修复与搞坏基本相等（28 vs 29），净变化为 −1。

### 12.3.3 统计显著性（McNemar test）

| Transition | 数据集 | p 值（双侧 exact McNemar） | 结论 |
| --- | --- | --- | --- |
| Base → SFT | Combined-500 | ≈ 2.39e-4 | 显著提升 |
| SFT → GRPO v3 | Combined-500 | = 1.0 | 无显著差异 |
| SFT → GRPO v3 | Dev-200 | ≈ 0.442 | 无显著差异 |
| SFT → GRPO v3 | Held-out-300 | ≈ 0.362 | 无显著差异 |

#### 核心结论

- SFT 在合并 500 条上呈现明确的 paired improvement；
- GRPO v3 在任一数据集上均未显示统计显著的超越 SFT 的证据。

### 12.3.4 GRPO 泛化检查结论

GRPO v3 最初在 200 条 development set 上表现出正向趋势，将 SFT 准确率从 48.5% 提升到 51.0%。然而，这一提升未能在独立 held-out 300 条样本上复现：

| 数据集 | SFT | GRPO v3 | Delta |
| --- | --- | --- | --- |
| Dev-200 | 48.5% | 51.0% | +2.5pp |
| Held-out-300 | 45.7% | 43.7% | −2.0pp |
| Combined-500 | 46.8% | 46.6% | −0.2pp |

样本级 transition 显示相同模式：

| 数据集 | GRPO 修复 | GRPO 搞坏 | 净变化 |
| --- | --- | --- | --- |
| Dev-200 | 16 | 11 | +5 |
| Held-out-300 | 12 | 18 | −6 |
| Combined-500 | 28 | 29 | −1 |

因此，Dev-200 上观察到的 GRPO 提升不应被解释为已确认的泛化增益。

由于 Dev-200 曾被用于 GRPO 配置选择（v2 vs v3），configuration-selection bias 是一个合理且重要的解释。其他因素，包括有限样本波动、题目组成差异、GRPO 训练本身的不稳定性，以及单次训练 seed 带来的差异，目前尚不能排除。

相比之下，SFT 在两个子集上均产生一致的提升，是目前该设置下更稳健的性能改善来源。

## 12.4 SFT Regression Recovery

### 12.4.1 实验设置

前面的 SFT 分析中，共发现 28 个 Base 正确、SFT 错误的 regression cases（基于 Dev-200）。

本节在这 28 道题上分别测试：

- Prompt C 干预（在 SFT 模型上）；
- GRPO v2 训练；
- GRPO v3 训练。

统一使用相同的 verifier 与解码配置，评测样本完全相同。

### 12.4.2 总体恢复结果

| 方法 | 恢复数 | 恢复率 |
| --- | --- | --- |
| Base | 28 / 28 | 100%（按定义） |
| SFT | 0 / 28 | 0%（按定义） |
| SFT + Prompt C | 9 / 28 | 32.1% |
| GRPO v2 | 3 / 28 | 10.7% |
| GRPO v3 | 9 / 28 | 32.1% |

#### 初步观察

- GRPO v2 仅恢复 3 题，与它在完整 200 条评测集上相比 SFT 下降 2.0 个百分点一致；
- GRPO v3 恢复 9 题，与 Prompt C 的恢复数相同；
- 虽然恢复总数相同，但两者恢复的题目集合并不相同（见 12.4.3）。

### 12.4.3 恢复集合的重叠分析

| 分组 | 数量 | 题目 ID |
| --- | --- | --- |
| Prompt C 与 GRPO v3 共同恢复 | 4 | 1046, 3462, 4287, 6797 |
| 仅 Prompt C 恢复 | 5 | 22, 4160, 4339, 476, 4874 |
| 仅 GRPO v3 恢复 | 5 | 1023, 3832, 4226, 5575, 5774 |
| 两者都未恢复 | 14 | — |

从集合角度看，这 28 道 regression 被划分为：

| 类型 | 数量 |
| --- | --- |
| Prompt C 与 GRPO 都能恢复 | 4 |
| 仅 Prompt C 能恢复 | 5 |
| 仅 GRPO 能恢复 | 5 |
| 两者都无法恢复 | 14 |

#### 关键观察

Prompt C 与 GRPO v3 的恢复集合重叠仅 4 题，两者各自独立恢复 5 题。repair set 的低重叠提示二者可能影响不同的 failure modes，而不是在修复完全相同的一批错误。

### 12.4.4 按错误类型的恢复差异

将 28 道 regression case 按错误类型拆开，对比 Prompt C 与 GRPO v3 的独立恢复能力（只统计各自独有的恢复）：

| 错误类型 | 总数 | Prompt C 独有恢复 | GRPO v3 独有恢复 |
| --- | --- | --- | --- |
| 纯算术 | 4 | 1 | 1 |
| 推理不完整 | 5 | 1 | 1 |
| 数量关系 | 4 | 2 | 0 |
| 比例 / 单位 | 4 | 1 | 0 |
| 遗漏约束 | 4 | 0 | 1 |
| 完全误读 | 7 | 0 | 2 |

#### 观察

- 在当前样本中，Prompt C 的独有恢复更多出现在数量关系、比例 / 单位类；
- GRPO v3 的独有恢复更多出现在完全误读、遗漏约束类；
- 纯算术与推理不完整两类，两者各自独有恢复 1 题，未观察到明显差异。

这一分布提示两种干预可能影响不同类型的 failure mode，但当前样本量不足以据此判断具体机制。各类型样本数仅 4~7 题，恢复率差异可能受具体题目难度影响，不应视为普适规律。

### 12.4.5 GRPO v2 与 v3 的恢复差异

| 方法 | 恢复数 | 恢复集合 |
| --- | --- | --- |
| GRPO v2 | 3 | 2216, 4226, 4287 |
| GRPO v3 | 9 | 1023, 1046, 3462, 3832, 4226, 4287, 5575, 5774, 6797 |
| v2 与 v3 共同恢复 | 2 | 4226, 4287 |
| 仅 v3 恢复 | 7 | 1023, 1046, 3462, 3832, 5575, 5774, 6797 |

#### 观察

- v2 恢复的 3 题中有 1 题（2216）在 v3 中反而没被恢复；
- v2 与 v3 的恢复集合高度不对称，v3 的 9 题中 7 题是 v2 无法恢复的。

这与 12.2 节的结论一致：GRPO 效果对训练配置较为敏感。由于 v2 与 v3 并非严格控制变量实验，本节不进一步归因于单一参数。

### 12.4.6 SFT → GRPO v3 与 SFT → Prompt C 的对比

在 Dev-200 上：

| 对比 | 修复（前错→后对） | 新增退化（前对→后错） | 净变化 | 准确率 |
| --- | --- | --- | --- | --- |
| SFT → Prompt C | 15 | 17 | −2 | 47.5% |
| SFT → GRPO v3 | 16 | 11 | +5 | 51.0% |

#### 关键观察（仅限 Dev-200）

- 修复数接近（15 vs 16），说明两种方法都有能力挽回一部分错误；
- 新增退化数差异明显（17 vs 11），GRPO v3 的新增 regression 数量相比 Prompt C 少约 35%；
- 净效果相反：Prompt C 净降 2 题，GRPO v3 净升 5 题。

需要特别说明：上述结论仅基于 Dev-200。在 held-out 300 条上，GRPO v3 相比 SFT 净变化为 −6，方向完全反转。因此，Dev-200 上"GRPO v3 比 Prompt C 有更好的 recovery-regression trade-off"这一观察未能在 held-out 上复现。

### 12.4.7 Prompt 与 GRPO 的样本级重叠分析

基于 12.4.3 和 12.5 的数据（Dev-200），进一步分析 Prompt C 与 GRPO v3 在样本级上的重叠。

#### 修复集合重叠

| 指标 | 值 |
| --- | --- |
| Prompt C 修复数 | 9 |
| GRPO v3 修复数 | 9 |
| 交集 | 4 |
| 并集 | 14 |
| Jaccard overlap | 4 / 14 = 28.6% |

在 Prompt C 无法恢复的 19 个 Hard Regression 中，GRPO v3 恢复了 5 个（5 / 19 ≈ 26.3%）。

在 Prompt C 可以恢复的 9 个 Soft Regression 中，GRPO v3 恢复了 4 个（4 / 9 ≈ 44.4%）。

#### 新增退化集合重叠

| 指标 | 值 |
| --- | --- |
| Prompt C 新增退化 | 17 |
| GRPO v3 新增退化 | 11 |
| 交集 | 8 |
| 并集 | 20 |
| Jaccard overlap | 8 / 20 = 40.0% |

GRPO v3 的 11 个新增退化中有 8 个（72.7%）同时也是 Prompt C 的失败样本。

#### 样本级结论

修复集合的 Jaccard overlap 为 28.6%，而新增退化集合为 40.0%。此外，GRPO v3 的 11 个新增退化中有 8 个（72.7%）同时也是 Prompt C 的失败样本，提示存在一批对不同干预都较敏感的样本。

样本级重叠分析表明：

- 修复集合的重叠度较低，说明 Prompt 干预与 GRPO 训练可能作用于不同的失败模式；
- 新增退化集合的重叠度较高，提示存在一批 intervention-sensitive samples——原始 SFT 可以做对，但一旦改变推理策略（无论通过 Prompt 还是 RL）就容易失败；
- 两类干预并非简单的替代关系。

需要说明的是，以上分析基于 Dev-200。由于 Dev-200 曾被用于 GRPO 配置选择，其样本级结论在 held-out 上未得到独立验证。

## 12.5 GRPO v3 新增退化分析

基于 Dev-200，GRPO v3 引入的新退化（SFT 对 → GRPO 错）共 11 题。按错误机制分类如下：

| 错误类型 | 数量 | 题目 ID |
| --- | --- | --- |
| 纯算术错误 | 3 | 1435, 5930, 2693 |
| 题意理解错误 | 4 | 5695, 2802, 2910, 826 |
| 比例 / 百分比错误 | 1 | 2539 |
| 推理完全崩塌 | 1 | 2762 |
| 逻辑关系错误 | 2 | 3376, 1033 |
| 合计 | 11 | — |

### 代表性错误模式

#### 模式 1：数字或单位算错（1435, 5930, 2693）

- 1435：5 pennies 的 5 × $0.01 算成 $0.50（差 10 倍）；
- 5930：36 + 12 + 1 算成 59；
- 2693：20000 × 0.10 算成 20000。

#### 模式 2：题意表征偏移（5695, 2802, 2910, 826）

- 5695：题目问百分比，GRPO 先算数量再转百分比；
- 2802：20% discount 被理解为「每件商品 × 20%」；
- 2910：60 ÷ 5 = 12 车次被算成 60 × 520 = 31200；
- 826：20 × $50 被算成 20 × 2。

#### 模式 3：推理路径偏移（3376, 1033）

- 3376：deadlift 掉 200 磅当成加 200 磅；
- 1033：最后一步多减了一个 300。

#### 模式 4：比例关系算错（2539）

2539：9/15 × 100% 算成 66%（应 60%）。

#### 模式 5：推理完全崩塌（2762）

2762：从正确的 refund 计算出发，最终输出 55 瓶（应 6 瓶）。

### 与 Prompt C 新增退化对比

| 对比 | Prompt C | GRPO v3 |
| --- | --- | --- |
| 新增退化总数 | 17 | 11 |
| 纯算术 | 4 | 3 |
| 题意理解 | 4 | 4 |
| 推理崩塌 | 5 | 1 |
| 过度推理 | 1 | 0 |
| 逻辑关系 | 2 | 2 |
| 巧合 / 不稳定 | 1 | 0 |
| 比例 / 百分比 | 0 | 1 |
| 共同搞坏（交集） | — | 8 |

### 关键差异

- 推理崩塌类型：Prompt C 有 5 例，GRPO v3 只有 1 例；
- 过度推理类型：Prompt C 有 1 例，GRPO v3 为 0；
- 题意理解错误：两者数量相同（4 题），说明这一类问题尚未被当前 Prompt intervention 或 GRPO 配置稳定解决；
- 共同搞坏的 8 题：占 GRPO v3 新增退化的 8 / 11，也占 Prompt C 新增退化的 8 / 17，提示存在一批对策略干预敏感的样本。

需要特别说明：以上分析基于 Dev-200，held-out 上的新增退化模式尚未做同等粒度的分析。由于 GRPO v3 在 held-out 上整体表现为 −6，其新增退化的分布可能与 Dev-200 有系统差异。

### 阶段性结论

GRPO v3 在 Dev-200 上的净提升 +5 题，来源是 16 题修复减 11 题新增退化。但在 held-out 300 条上，修复 12 题、搞坏 18 题，净变化 −6。

修复的 9 题来自 28 道 SFT regression，7 题是 SFT 和 Base 都错的题。后一部分是否对应更一般化的新推理能力，在 held-out 上未获得支持。

GRPO v3 在 Dev-200 上的新增退化中，推理崩塌和过度推理几乎不存在，而 Prompt C 的新增退化里这两类占了 6 / 17。这一差异是否在 held-out 上保持，需进一步分析。

题意理解错误在两种干预下均持续出现（各 4 题），说明这一类错误尚未被当前方法稳定解决。

与 Prompt C 相比，在 Dev-200 上 GRPO v3 在修复数接近的前提下，新增退化减少约 35%，净效果由负转正。但这一优势未能在 held-out 上复现。

样本级重叠分析显示：在 Dev-200 上，修复集合的 Jaccard overlap 为 28.6%，新增退化集合为 40.0%。这提示两类干预作用于不同的失败模式，但可能共享一批 intervention-sensitive samples。

核心修正：基于 held-out 上的复现失败，GRPO v3 相比 SFT 的提升应定位为 positive trend on dev set, unconfirmed on held-out。SFT 是目前该设置下最稳健的性能改善来源。

## 12.6 GRPO Training Dynamics

KL divergence 用于衡量当前策略相对于参考策略的偏移程度。

GRPO v2 与 v3 的 KL 统计如下：

| KL 统计量 | GRPO v2 | GRPO v3 |
| --- | --- | --- |
| 前 10 步均值 | 4.7e-04 | 5.8e-05 |
| 后 10 步均值 | 1.9e-03 | 2.2e-03 |
| 全期均值 | 1.6e-03 | 1.3e-03 |
| 中位数 | 1.4e-03 | 1.3e-03 |
| 最后一步 | 1.5e-03 | 1.4e-03 |

### 训练初期

v3 在训练前 10 步中的平均 KL 为 5.8e-05，明显低于 v2 的 4.7e-04，约低 8 倍。

v3 使用了更大的 KL 系数（beta 从 0.01 提高到 0.04），因此这一结果与「更强的 KL 约束降低训练初期策略偏移」的预期一致。

但由于 v2 与 v3 同时改变了学习率、num_generations、梯度累积、生成长度和训练步数等多个参数，目前不能将这一差异单独归因于 beta。

### 训练后期

随着训练进行，两组实验的 KL 逐渐进入相近量级：

v2 后 10 步均值：1.9e-03

v3 后 10 步均值：2.2e-03

最后一步也非常接近：

v2：1.5e-03

v3：1.4e-03

全期平均值同样处于相同数量级：

v2：1.6e-03

v3：1.3e-03

因此目前观察到：v3 在训练初期具有明显更小的策略偏移，但这一差异并没有持续到训练末期。

一个可能的解释是，更强的 KL 约束降低了训练早期的更新幅度，而随着训练持续进行，策略仍然逐渐累积偏离参考策略。

与此同时，v3 的训练长度为 1000 steps，而 v2 只有 300 steps，因此更长的训练窗口本身也可能影响最终观察到的 KL 分布。

## 12.7 Reward Signal Audit（初步）

为理解 GRPO v3 未能泛化的原因，我们对训练过程中的 reward 信号进行了初步审计。

统计 GRPO v3 训练日志中的 frac_reward_zero_std 指标（表示 group 内 reward 完全一致、即无有效 advantage 的 group 比例）：

| 统计量 | 值 |
| --- | --- |
| 记录步数 | 200 |
| 均值 | 0.310 |
| ≥ 0.5 的步数占比 | 20.5% |
| ≥ 0.75 的步数占比 | 4.0% |
| == 1.0 的步数占比 | 0.5% |

### 初步观察

- 平均 31% 的 group 为 zero-variance（全对或全错），约 69% 的 group 具有有效 advantage；
- 极端情况（全 zero-variance）几乎不存在；
- 从 zero-variance 比例看，GRPO v3 的训练信号是正常且充足的。

因此，GRPO v3 未能在 held-out 上泛化的原因，很可能不是由 zero-variance group 比例过高导致的。需要从 reward 设计、训练样本覆盖、KL 约束等其他角度进一步排查。

### 后续计划

基于以上审计结果，下一阶段将进行：

- 更细粒度的 Reward 审计：逐 step 分析 group 内 reward 分布，包括 [0,0,...,0]、[1,1,...,1]、部分正确 group 的比例，以及 reward 与真实 correctness 的一致率；
- Reward Design 消融：以 exact match 为 anchor（R1），逐步测试 format reward（R2）和 repetition/invalid penalty（R3）；
- 多 seed 实验：排除训练随机性对结论的影响；
- MiniGRPO from scratch：从零实现最小 GRPO 以验证对算法的理解。

## 12.8 关于 KL 稳态的初步假设

一个值得进一步研究的现象是：

尽管 v2 与 v3 的 beta 相差 4 倍、v3 的训练步数超过 v2 的 3 倍、两组训练初期的 KL 差异明显，但两者训练末期的 KL 最终都落在约 1e-3 的数量级。

这提示一种可能的假设：

在当前模型、数据集和 Reward 设置下，KL 可能存在某种相对稳定的训练区间，而 beta 的作用更多体现在控制策略偏移的速度，而不一定完全决定长期 KL 水平。

但目前这一现象还不能被解释为任务本身决定了固定的 KL 稳态值，原因包括：

- v2 与 v3 并非严格控制变量实验；
- 两组训练长度不同；
- 学习率、Group Size 和生成长度等参数也同时发生变化；
- 当前只有两组主要实验，样本不足以建立普适规律。

因此，该现象目前仅作为 training dynamics hypothesis 保留。

后续可以通过固定其他参数、只改变 beta 的消融实验验证：

- beta 是否主要影响 KL 上升速度；
- 不同 beta 是否最终收敛到相似 KL 区间；
- KL 与最终数学准确率之间是否存在稳定关系。

特别值得关注的问题是：

更低的 KL 是否一定意味着更好的下游 Accuracy？

当前结果已经提示答案未必如此：KL 本身更适合作为策略漂移指标，而不能直接等同于推理能力提升。

## 12.9 实验设计与协议更新

基于 held-out 300 条上的复现失败，我们对项目的数据集角色划分进行了修订：

| 数据集 | 条数 | 角色 |
| --- | --- | --- |
| 训练集 | 6973 | 训练 |
| 原 validation 全集 | 500 | Development / Analysis Set |
| GSM8K official test | 1319 | Final Test（所有配置冻结后才使用） |

### 核心修订

- 原 Dev-200 与 Held-out-300 合并为 Development Set（500 条），作为所有后续消融实验的统一评测集；
- 由于两部分数据均已被观察，不再将其中任一部分视为严格 held-out；
- GSM8K official test 保持完全不碰，仅在所有设计冻结后对最终 1~2 个配置各运行一次。

这一调整使得实验协议更加诚实：不再假装某一部分数据是 held-out，而是接受当前全部 500 条已被使用，并通过最终的 official test 提供无偏的对外数字。

这次修订本身也是项目的一个重要发现：

在小规模评测集上做配置选择可能引入严重的过拟合，凸显了独立 held-out 协议在 RL 后训练中的必要性。

## 12.10 Multi-Seed 复现实验

> **术语说明**：本节的 `tail300` 指原 held-out 300 条（heldout300.jsonl）。由于该子集在早期分析中已被观察过（用于 GRPO v2 vs v3 配置选择），它已不再构成严格意义上的独立 held-out。此处改名为 `tail300` 以更准确地反映其角色。真正未受污染的数据集是 GSM8K official test。

为排除 GRPO 相比 SFT 的差异来自训练随机性的可能，我们固定 GRPO v3 的全部超参数（max_steps=1000, num_generations=8, learning_rate=5e-6, beta=0.04, max_completion_length=256），仅改变随机种子，分别以 seed 42 / 1234 / 2026 训练三个独立模型，并在 dev500 与 tail300 上评测。

### 12.10.1 三 Seed 完整结果

| Model | dev500 | tail300 (former held-out) |
| :--- | :---: | :---: |
| Base | 36.8% | 35.3% |
| SFT | 46.8%（234 / 500） | 45.7%（137 / 300） |
| GRPO seed 42 | 46.6% | 43.7% |
| GRPO seed 1234 | 47.4% | 45.3% |
| GRPO seed 2026 | 46.2% | 45.0% |
| **GRPO mean ± std** | **46.7% ± 0.50pp** | **44.7% ± 0.72pp** |

### 12.10.2 三条关键结论

**① GRPO 训练方差极小（std 仅 0.5~0.7pp）**

三个 seed 在 dev500 上的结果落在 46.2%~47.4% 之间，标准差 0.50pp；在 tail300 上落在 43.7%~45.3%，标准差 0.72pp。

这一较小的方差提示：dev200 与 tail300 之间的方向反转不太可能仅由训练随机性解释。但当前结果尚不能唯一归因于 configuration-selection bias——也可能与两个子集的题型/难度分布差异、prompt 覆盖不同等因素有关。

**② GRPO 相比 SFT 没有稳定提升**

| 数据集 | SFT | GRPO mean | Delta |
| :--- | :---: | :---: | :---: |
| dev500 | 46.8% | 46.7% | **−0.1pp** |
| tail300 | 45.7% | 44.7% | **−1.0pp** |

在 tail300 上，三个 GRPO seed 均略低于 SFT（43.7% / 45.3% / 45.0% vs 45.7%）。在 dev500 上，GRPO 三个 seed 的均值（46.7%）与 SFT（46.8%）几乎持平。

**③ 配置选择偏差是一个合理但未被唯一确认的解释**

之前观察到 GRPO v3 在 dev200 上相比 SFT 提升 +2.5pp，但在 tail300 上反而下降 −2.0pp。当时推测这是 configuration-selection bias（dev200 曾被用于 v2 vs v3 的配置比较）导致的。

三 seed 复现实验与这一推测一致：GRPO 在 dev 集上的表面优势在 tail300 上未能复现，且这一模式在不同 seed 下稳定存在。但由于 dev200 与 tail300 的差异也可能来自题型/难度分布或 prompt 覆盖等因素，当前结果尚不能唯一确定配置选择偏差是主要机制。

### 12.10.3 项目核心结论

基于以上完整实验，项目核心结论可总结为：

> 在小规模 LLM（Qwen2.5-0.5B）与 GSM8K 数学推理任务上，**SFT 是最稳健的性能提升来源**，将 Base 从 36.8% 提升至 46.8%（+10.0pp），且在 dev 与 held-out 上方向一致。
>
> **GRPO 在当前配置与数据规模下未能进一步改善推理能力**。三 seed 复现显示 GRPO 相比 SFT 的差异较小（dev500 delta −0.1pp，tail300 delta −1.0pp），且 dev 集上的表面优势在 tail300 上未能复现。
>
> 当前的四层 Reward Audit（Signal Availability / Reward Quality / Counterfactual / Reasoning Quality）未发现 reward availability 的明显异常，但尚未排除 reward sparsity、difficulty bias 和 verifier noise。问题可能位于 policy update、数据覆盖、优化目标与泛化机制等多个层面，需要更严格的消融实验进一步研究。

## 12.11 Multi-Seed 实验的意义

本节记录 Multi-Seed 实验在整个项目中的方法论价值：

1. **区分方差与偏差**：单 seed 实验中，dev200 上的 +2.5pp 和 tail300 上的 −2.0pp 无法判断是"系统性差异"还是"随机波动"。三 seed 后标准差仅 0.5~0.7pp，提示该差异更可能来自系统因素而非随机波动，但尚不能唯一归因于配置选择偏差。

2. **验证泛化失败的稳定性**：如果只在 seed 42 上观察，可能被质疑"只是运气不好"。三 seed 全部低于 SFT 后，结论变得更加可靠。

3. **为后续消融建立基线**：任何未来的 GRPO 消融实验，都需要在相同数据划分和相同 seed 数量下进行比较，才能判断改进是否真实。


## 十三、计划中的 GRPO 消融实验

计划探索以下方向，结果待实验完成后补充：

- **Reward 设计**
- **Group Size 与 `num_generations`**
- **KL 正则化**
- **Sampling Temperature**

## 十四、从零实现 MiniGRPO

本节预留用于验证对 GRPO 算法的理解，不作为成熟训练框架的替代方案。实现与实验结果待补充。

计划流程如下：

1. **Prompt 输入**
2. **Group Rollout**：对同一 Prompt 采样多个输出
3. **Reward 计算**
4. **Group-relative Advantage 估计**
5. **Token Log Probability 计算**
6. **Policy Ratio 计算**
7. **Clipping**
8. **KL 项计算**
9. **GRPO Loss 组合**

## 十五、未来工作

后续探索方向包括：

- Hard Example Mining
- Curriculum Learning
- Synthetic Math Data
- Verifier 的进一步改进
- Distillation
- Quantization
- Agent 与 Tool-use RL