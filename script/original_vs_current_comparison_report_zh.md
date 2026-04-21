# UniFormTSV 原版与当前增强版对比报告

## 1. 对比基准

本报告对比的是你昨天晚上最初上传的 UniFormTSV 项目版本与当前增强后的版本。

在当前 Git 历史中，对比基准如下：

| 项目状态 | Git 版本 | 说明 |
| --- | --- | --- |
| 原版项目 | `051b0fc` | 我开始修改前的项目状态 |
| 当前增强版 | `624464a` | 已完成 C-TCAR、能源数据集、路由实验与本地 PatchTST backbone 扩展后的状态 |

从原版到当前版本，总体代码变化为：

| 指标 | 数量 |
| --- | ---: |
| 变更文件 | 38 |
| 新增代码/文档行 | 3284 |
| 修改/替换行 | 178 |
| 新增文件 | 14 |
| 删除项目文件 | 0 |

## 2. 总体变化结论

原版项目更像是一个时间序列缺失值填补模型的统一运行框架。它可以运行 SAITS、TEFN、MOMENT、UniFormTSV、GPT4TS、TSLANet 等模型，也有 PhysioNet、空气质量、交通、ETT 等数据集入口，但整体实验规模较小，缺少能源系统专项数据，缺少自动化大规模 benchmark，也没有论文中强调的“根据场景选择模型”的路由机制。

当前增强版已经从“单模型填补实验框架”扩展成了“能源时间序列 C-TCAR 模型路由实验系统”。现在它不仅能训练或评估多个模型，还能针对不同数据集、缺失率、统计特征、自相关特征、频谱特征和因果代理特征，学习当前场景应该选择哪个模型。

## 3. 数据集层面的新增

### 原版项目

原版主要支持：

| 数据类型 | 原版支持情况 |
| --- | --- |
| 医疗缺失值填补基准 | `physionet_2012` |
| 空气质量 | `air_quality`、`italy_air_quality`、`beijing_multisite_air_quality` |
| 交通 | `pems_traffic` |
| ETT 电力变压器数据 | `etth1`、`etth2`、`ettm1`、`ettm2` |

这些数据可以用于通用时间序列缺失值填补，但和论文中的能源系统干预/路由问题还有距离。

### 当前增强版

我新增了面向能源系统的公共数据集准备与加载逻辑：

| 新增数据集 | 含义 |
| --- | --- |
| `appliances_energy` | 家庭电器能耗与室内外环境变量 |
| `household_power` | 家庭用电功率序列 |
| `citylearn_zone5` | 建筑/区域级能耗场景 |
| `opsd_germany` | 德国公开电力系统负荷/发电数据 |
| `solar` / `solar_alabama` | 光伏发电序列，高维 137 特征 |
| `eld` / `electricity_load_diagrams` | Electricity Load Diagrams，高维 370 特征 |

对应新增文件：

- `pypots/data/dataset/energy_preparation.py`
- `pypots/data/dataset/load_prepare_dataset.py`

现在项目可以自动准备更多能源场景，并支持滑动窗口切分、样本数限制、随机缺失生成和随机种子控制。

## 4. 模型层面的新增

### 原版项目

原版主要强调深度填补模型和 LLM/Transformer 类模型，例如：

| 类型 | 模型 |
| --- | --- |
| 深度填补模型 | `saits`、`tefn` |
| 时间序列/Transformer 模型 | `tslanet`、`moment`、`uniformtsv`、`gpt4ts`、`timemixerpp` |
| LLM 类模型 | `llm4imp`、`timellm` |

但原版缺少非常重要的简单基线。对于能源负荷这类连续平滑序列，简单基线往往很强。如果没有这些基线，实验结论会不够有说服力。

### 当前增强版

我新增了统计与传统时间序列基线：

| 新增模型 | 作用 |
| --- | --- |
| `mean` | 均值填补基线 |
| `median` | 中位数填补基线 |
| `locf` | Last Observation Carried Forward，连续能源序列中的强基线 |

对应新增文件：

- `pipeline/imputations/baselines.py`

我还为 `UniFormTSV` 和 `MOMENT` 增加了本地可训练的 PatchTST-style backbone：

| 新增能力 | 作用 |
| --- | --- |
| `PatchTST` backbone | 在没有云 GPU 或 HuggingFace 大模型的情况下，本机也能端到端训练 Transformer 风格 backbone |
| 高维输出投影 | 支持 Solar 137 维和 ELD 370 维，避免 `d_model=64` 与输出维度不匹配 |
| `n_layers` 透传 | 让模型深度可以从命令行和 benchmark 脚本中控制 |

对应修改文件：

- `pypots/nn/modules/uniformtsv/backbone.py`
- `pypots/nn/modules/moment/backbone.py`
- `pypots/imputation/uniformtsv/core.py`
- `pypots/imputation/moment/core.py`
- `pypots/imputation/uniformtsv/model.py`
- `pypots/imputation/moment/model.py`

## 5. 训练与实验强度的新增

### 原版项目

原版默认训练配置较轻：

| 配置 | 原版默认 |
| --- | ---: |
| `epochs` | 10 |
| `patience` | 3 |
| 大规模批量 benchmark | 无 |
| 多随机种子实验 | 无统一脚本 |
| 多缺失率实验 | 无统一脚本 |
| 自动汇总指标 | 无 |
| 自动路由分析 | 无 |

### 当前增强版

我新增了完整 benchmark runner：

- `run_energy_benchmark.sh`
- `run_ctcar_200_benchmark.sh`

当前 runner 支持：

| 能力 | 当前增强版 |
| --- | --- |
| 多数据集循环 | 支持 |
| 多模型循环 | 支持 |
| 多缺失率循环 | 支持 |
| 多随机种子循环 | 支持 |
| CUDA/MPS/CPU 自动识别 | 支持 |
| 单次运行超时 | 支持 |
| 总时间预算 | 支持 |
| `.done` 标记跳过已完成实验 | 支持 |
| 日志保存 | 支持 |
| 统一输出目录 | 支持 |

脚本默认训练强度也提升为：

| 配置 | 当前 runner 默认 |
| --- | ---: |
| `epochs` | 50 |
| `patience` | 10 |
| 缺失率 | `0.1`、`0.3`、`0.5`、`0.7` |
| 随机种子 | `42`、`123`、`456` |
| 默认数据集 | 10+ |
| 默认模型 | 10 个左右 |

由于本机是 Apple 芯片 16GB 内存，我实际已经完成的是一轮本机可承受的中等规模实验，使用 MPS 后端，并对较重模型设置了超时保护。

## 6. C-TCAR 与模型路由的新增

### 原版项目

原版没有实现论文中的核心思想：根据场景条件选择合适模型。它更多是“指定一个模型，然后训练/评估这个模型”。

### 当前增强版

我新增了两个路由体系：

| 路由方式 | 输入特征 | 输出 |
| --- | --- | --- |
| 元信息路由 | 数据集名称、缺失率、序列长度、特征维度等 | 当前场景推荐的最佳模型 |
| C-TCAR 路由 | 统计特征、自相关特征、频谱特征、滞后依赖图特征、混杂代理特征、缺失率 | 当前场景推荐的最佳模型 |

C-TCAR 特征抽取脚本：

- `script/ctcar_features.py`

路由分析脚本：

- `script/model_routing_analysis.py`
- `script/ctcar_routing_analysis.py`

报告生成脚本：

- `script/energy_benchmark_200_report.py`

这里的 C-TCAR 是 causal-aware proxy，也就是“因果感知代理表征”。它已经把论文中“根据条件表征进行模型路由”的思想接入项目，但当前还不是完整的干预预测复现，因为项目目前没有真实 intervention/action/outcome 标签，也没有 ATE 或 counterfactual ground truth。

## 7. 当前已经跑出的实验结果

当前本机实验输出位置：

`output/imputation/mps/energy_benchmark_200`

实验规模：

| 项目 | 数量 |
| --- | ---: |
| 指标记录 | 514 |
| 数据集 | 11 |
| 模型 | 9 |
| 缺失率 | 4 |
| 随机种子 | 2 |
| 数据集/缺失率场景 | 44 |

实际覆盖数据集：

`appliances_energy`、`citylearn_zone5`、`eld`、`etth1`、`etth2`、`ettm1`、`ettm2`、`household_power`、`opsd_germany`、`physionet_2012`、`solar`

实际覆盖模型：

`gpt4ts`、`locf`、`mean`、`median`、`moment`、`saits`、`tefn`、`tslanet`、`uniformtsv`

平均 MAE 排名：

| 模型 | 平均 MAE |
| --- | ---: |
| `gpt4ts` | 0.169397 |
| `locf` | 0.231610 |
| `saits` | 0.405146 |
| `median` | 0.636868 |
| `mean` | 0.672744 |
| `moment` | 0.788573 |
| `uniformtsv` | 0.788573 |
| `tslanet` | 0.888237 |
| `tefn` | 1.121273 |

按 44 个数据集/缺失率场景统计的胜场：

| 模型 | 胜场 |
| --- | ---: |
| `locf` | 27 |
| `saits` | 12 |
| `gpt4ts` | 5 |

这个结果说明：当前项目的重点不应该是简单声称“复杂模型一定最好”，而应该是强调“不同能源场景需要不同模型，模型路由有必要”。这正好更贴近论文中 learning to route 的思想。

## 8. 路由结果

元信息路由结果：

| 指标 | 数值 |
| --- | ---: |
| 训练准确率 | 0.8636 |
| 留一场景准确率 | 0.7500 |
| 留一数据集准确率 | 0.6818 |

C-TCAR 路由结果：

| 指标 | 数值 |
| --- | ---: |
| 训练准确率 | 1.0000 |
| 留一场景准确率 | 0.7500 |
| 留一数据集准确率 | 0.6818 |
| 场景数 | 44 |
| 特征数 | 81 |
| 可预测最佳模型类别 | `gpt4ts`、`locf`、`saits` |

这说明 C-TCAR 路由器已经可以学习“什么数据场景适合什么模型”的规律。当前它和元信息路由在泛化指标上持平，后续如果加入真实干预标签和更大规模能源数据，C-TCAR 的优势会更容易体现。

## 9. 新增文件清单

| 新增文件 | 作用 |
| --- | --- |
| `pipeline/imputations/baselines.py` | 新增 `mean`、`median`、`locf` 基线 |
| `pypots/data/dataset/energy_preparation.py` | 新增能源数据集下载、准备、滑窗和标准化逻辑 |
| `run_energy_benchmark.sh` | 多模型、多数据集、多缺失率、多 seed benchmark runner |
| `run_ctcar_200_benchmark.sh` | 论文复现/增强版一键实验入口 |
| `script/collect_metrics.py` | 汇总实验指标 |
| `script/ctcar_features.py` | 抽取 C-TCAR 特征 |
| `script/ctcar_routing_analysis.py` | C-TCAR 路由器训练与评估 |
| `script/model_routing_analysis.py` | 元信息路由器训练与评估 |
| `script/energy_benchmark_200_report.py` | 自动生成实验报告 |
| `script/energy_benchmark_200_report.md` | 英文实验报告 |
| `script/energy_benchmark_200_report_zh.md` | 中文实验说明稿 |
| `script/energy_experiment_plan.md` | 能源实验计划 |
| `script/energy_experiment_report.md` | 能源实验阶段报告 |
| `script/project_evolution_paper.md` | 项目演化说明稿 |

## 10. 与原版相比，现在项目能做到什么

当前项目相比原版，新增能力可以概括为：

1. 可以跑更多能源系统数据集，而不是只跑通用缺失值填补数据集。
2. 可以比较深度模型、LLM/Transformer 模型和简单强基线，而不是只比较复杂模型。
3. 可以进行多数据集、多模型、多缺失率、多随机种子的自动化实验。
4. 可以在本机 MPS 上完成中等规模实验，也能迁移到 LightCC 云 GPU 上继续扩大规模。
5. 可以自动汇总 MAE、MSE、RMSE、MRE 等指标。
6. 可以基于实验结果学习“当前场景应该选择哪个模型”。
7. 可以抽取 C-TCAR 特征，把论文中的条件表征/路由思想接入项目。
8. 可以输出中文和英文实验报告，支撑毕业论文或项目说明。

## 11. 仍然没有完全做到的部分

为了避免把项目包装得过头，这里也明确写出当前限制：

1. 当前 C-TCAR 还不是完整的因果干预预测复现，而是因果感知代理表征。
2. 当前项目还缺少真实 intervention/action/outcome 标签。
3. 当前还没有 ATE 或 counterfactual ground truth，因此不能声称已经完整复现论文中的因果干预预测任务。
4. TimeMixer++、T5 backbone、更大规模 GPT4TS/TSLANet、多 seed、更长 epoch 更适合放到 LightCC GPU 上继续跑。
5. 本机实验已经证明流程可运行、方法可扩展、结果有区分度，但如果要达到正式论文级大规模结果，建议继续上云 GPU 扩展实验。

## 12. 一句话总结

原版 UniFormTSV 是一个时间序列缺失值填补模型运行框架；当前增强版已经变成一个面向能源系统的 C-TCAR 模型路由实验平台。它新增了能源数据集、简单强基线、本地 PatchTST backbone、大规模 benchmark runner、C-TCAR 特征、模型路由器、实验指标汇总和中英文报告，并且已经在本机完成 514 条实验指标记录与 44 个场景的路由验证。
