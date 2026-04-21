# UniFormTSV 项目阶段性技术报告：从单一时间序列插补实验到能源场景模型路由框架

## 摘要

UniFormTSV 项目原本主要围绕时间序列缺失值插补任务展开，能够调用若干 PyPOTS 风格的插补模型进行训练和测试，但整体实验设置较简单：数据集范围有限、默认模型和训练强度偏弱、能源系统场景不足，也缺少对不同模型在不同数据分布下表现差异的系统分析。

本阶段工作参考用户提供的论文 `Learning_to_Route_Models_for_Causal_Intervention_Prediction_in_Energy_Systems.pdf` 的核心思想：能源系统中的预测或插补任务不应只依赖单一模型，而应根据数据来源、缺失强度、时间尺度和变量结构选择合适模型。基于这一思想，本项目被扩展为一个面向能源时间序列的多数据集、多模型、多缺失率 benchmark 框架，并进一步加入了 C-TCAR 特征提取和模型路由分析模块。

当前项目已经支持 4 个公开能源数据集、1 个非能源对照数据集、7 类模型或基线方法，并完成了 99 个实验组合。初步结果表明，复杂模型并不总是最优：在 `appliances_energy` 数据集上，简单的 LOCF 基线显著优于神经网络模型；而在 `citylearn_zone5`、`household_power`、`opsd_germany` 和 `physionet_2012` 上，SAITS 更稳定。进一步地，项目现在能够为 15 个数据集/缺失率场景抽取 80 维 C-TCAR 特征，并训练 RandomForest router 进行模型选择。这一结果直接支持了“模型选择/模型路由”而非“固定单一模型”的实验路线。

## 1. 项目原有基础

### 1.1 原项目目标

原项目 UniFormTSV 主要面向时间序列缺失值插补任务。项目中已经包含若干模型接口，例如：

- SAITS
- TimeMixer++
- UniFormTSV
- MOMENT
- TimeLLM
- GPT4TS
- TEFN
- TSLANet

这些模型通过 `main.py` 和 `pipeline/imputations/` 下的训练流程被调用，整体结构接近“选择一个模型、选择一个数据集、训练并输出 MAE/MSE/RMSE/MRE 指标”的实验框架。

### 1.2 原项目存在的问题

在本阶段改造之前，项目存在几个明显不足：

1. **环境不可直接运行**
   用户终端中 `python` 命令不存在，项目依赖环境没有就绪，无法立即运行训练或复现实验。

2. **数据集不足，能源场景薄弱**
   原项目主要依赖已有 benchmark 数据，缺少更贴近能源系统的公开数据源。对于一个希望结合能源系统论文思路的项目来说，数据集支撑不足。

3. **实验矩阵较弱**
   原有运行方式更像单模型测试，缺少多模型、多缺失率、多数据源的系统对比。

4. **缺少强基线**
   原项目偏重神经网络模型，但没有系统加入 Mean、Median、LOCF 等传统插补方法。没有这些基线时，很难证明复杂模型真的有效。

5. **缺少模型路由分析**
   虽然项目名字和论文思路都指向“选择模型”或“路由模型”，但原项目没有形成“在什么场景下选择哪个模型”的分析产物。

6. **Apple Silicon/MPS 支持不足**
   当前机器是 macOS 环境，Torch MPS 可用，但项目设备选择逻辑原先不够适配，导致部分模型无法顺利在本机跑起来。

## 2. 本阶段的主要修改

### 2.1 构建可运行环境

由于系统中没有 `python` 命令，本阶段首先在项目内创建了 `.venv` 虚拟环境，并安装项目训练所需依赖，包括：

- PyTorch
- PyPOTS 相关依赖
- benchpots
- pygrinder
- tsdb
- transformers
- peft
- datasets
- scikit-learn

环境验证后，项目可以使用：

```bash
.venv/bin/python main.py ...
```

或直接运行 benchmark 脚本：

```bash
./run_energy_benchmark.sh
```

### 2.2 新增公开能源数据集

本阶段新增了统一的能源数据集下载和预处理模块：

`pypots/data/dataset/energy_preparation.py`

新增支持的数据集包括：

| 数据集 | 来源 | 作用 |
| --- | --- | --- |
| `appliances_energy` | UCI Appliances Energy Prediction | 家庭环境、天气、家电能耗时间序列 |
| `household_power` | UCI Individual Household Electric Power Consumption | 家庭电力消耗和分项功率数据 |
| `citylearn_zone5` | CityLearn Climate Zone 5 | 多建筑能源需求、天气、碳强度、太阳能数据 |
| `opsd_germany` | Open Power System Data | 德国电力负荷、风电、光伏、价格相关时间序列 |

这些数据集被统一处理为滑动窗口格式，并自动完成：

- 时间序列排序
- 数值列筛选
- 重采样或缺失处理
- 标准化
- train/validation/test 划分
- 人工缺失掩码生成
- `n_steps`、`n_features` 元信息返回

### 2.3 扩展数据加载流程

修改了：

`pypots/data/dataset/load_prepare_dataset.py`

现在项目既能加载原有 benchmark 数据集，也能加载新增能源数据集，并支持以下参数：

- `--window_stride`
- `--max_samples`
- `--random_seed`

其中 `--max_samples` 不只作用于新增能源数据集，也适配了部分已有数据集，便于在本地机器上快速做小规模验证。

### 2.4 扩展命令行参数和默认配置

修改了：

`argument_parser.py`

主要变化包括：

- 默认模型改为更容易跑通的 `saits`
- 新增能源数据集名称说明
- 新增 `learning_rate`、`weight_decay`
- 新增 `window_stride`、`max_samples`、`random_seed`
- 调整 patch stride 默认值
- 将 TEFN 的 `n_fod` 默认值降到更安全的配置

### 2.5 修复随机种子问题

原 `main.py` 中固定使用：

```python
set_random_seed(2025)
```

这会导致命令行传入的 `--random_seed` 只影响部分数据处理，而不能真正影响模型初始化和训练过程。本阶段将其修改为：

```python
set_random_seed(args.random_seed)
```

这样后续多 seed 实验才真正具有意义。

### 2.6 新增传统强基线

新增文件：

`pipeline/imputations/baselines.py`

并在 `main.py` 中接入：

- `mean`
- `median`
- `locf`

加入传统基线后，实验不再只是“复杂模型之间互相比较”，而是可以回答更关键的问题：复杂模型是否真的超过了简单规则。

实验结果显示，`locf` 在 `appliances_energy` 上显著优于所有神经模型。这说明传统基线非常必要，否则项目可能会错误地把复杂模型表现解释为有效。

### 2.7 修复 BaseImputer 设备传参错误

在加入 Mean/Median/LOCF 基线时发现，PyPOTS 的 `BaseImputer` 构造函数中错误地把设备写死为：

```python
device="device"
```

这会导致无参基线在初始化时触发设备解析错误。本阶段将其修复为：

```python
device=device
```

该修复使传统基线能够在本机 MPS 环境下正常运行。

### 2.8 增强 MPS 支持

修改了：

`pypots/base.py`

现在设备选择逻辑支持：

1. CUDA
2. Apple Silicon MPS
3. CPU

本机没有 NVIDIA CUDA，但 Torch MPS 可用，因此当前实验主要在 `mps` 上运行。

### 2.9 修复 UniFormTSV/MOMENT 相关问题

为了让 UniFormTSV 和 MOMENT 能够正常跑通，本阶段修复了以下问题：

- T5 encoder-decoder 类型加载错误
- `T5EncoderModel` 与 `decoder_inputs_embeds` 不匹配
- `d_model` 与 T5 hidden size 不一致
- patch stride 和 patch length 使用不一致
- reconstruction 长度需要裁剪回 `n_steps`

相关文件包括：

- `pypots/nn/modules/uniformtsv/backbone.py`
- `pypots/nn/modules/uniformtsv/modules.py`
- `pypots/imputation/uniformtsv/core.py`
- `pypots/nn/modules/moment/backbone.py`
- `pypots/nn/modules/moment/modules.py`
- `pypots/imputation/moment/core.py`

### 2.10 修复 TimeMixer++ 频率选择问题

TimeMixer++ 的 FFT 周期选择中可能选中 0 频率，导致除零风险。本阶段修改了：

`pypots/nn/modules/timemixerpp/layers.py`

现在会排除 0 频率，再选择 top-k 周期。

### 2.11 新增 benchmark 脚本

新增：

`run_energy_benchmark.sh`

该脚本支持通过环境变量配置：

- 模型列表
- 数据集列表
- 缺失率列表
- 随机种子列表
- epoch
- patience
- batch size
- d_model
- n_layers
- TEFN 的 N_FOD
- max samples
- window stride

同时支持断点式运行：每个成功组合都会写入 `.done` 文件，重复运行时会跳过已经完成的组合。

### 2.12 新增指标汇总脚本

新增：

`script/collect_metrics.py`

该脚本会扫描所有实验输出目录中的 `*_metrics.json`，聚合为统一的 CSV：

`output/imputation/mps/energy_benchmark/metrics_summary.csv`

CSV 中包含：

- run
- model
- dataset
- missing_rate
- seed
- MAE
- MSE
- RMSE
- MRE
- metrics_file

### 2.13 新增模型路由分析

新增：

`script/model_routing_analysis.py`

该脚本从 `metrics_summary.csv` 中生成：

- 每个数据集/缺失率下的最优模型
- 第二名模型
- 最优模型相对第二名的收益
- 候选模型数量
- 一个轻量决策树 router
- router 规则和诊断结果

输出目录：

`output/imputation/mps/energy_benchmark/routing/`

包括：

- `routing_table.csv`
- `router.joblib`
- `router_diagnostics.json`
- `router_rules.txt`

### 2.14 新增 C-TCAR 特征提取与路由分析

为了更贴近论文中的 C-TCAR 方法，本阶段继续新增：

`script/ctcar_features.py`

该脚本实现了一个 80 维 C-TCAR 表示，包括：

| 特征组 | 维度 | 当前实现 |
| --- | ---: | --- |
| Statistical features | 20 | 均值、方差、分位数、偏度、峰度、缺失率等 |
| Temporal features | 24 | lag 1 到 lag 24 的平均自相关强度 |
| Spectral features | 19 | FFT 能量分布、频谱熵、主频、频带能量比例等 |
| Causal structural features | 10 | 基于 lagged dependency graph 的密度、入度、出度、路径强度等代理特征 |
| Causal confounding features | 7 | 共同父节点、backdoor proxy、调整集大小、混杂偏差 proxy 等 |

需要说明的是，当前实现还不是完整 DYNOTEARS + do-calculus 的因果系统。由于项目当前没有真实干预标签，因果结构部分采用 lagged dependency graph 作为轻量代理：用上一时刻变量到下一时刻变量的依赖强度构造有向图，再从图中提取 causal structure 和 confounding proxy。

同时新增：

`script/ctcar_routing_analysis.py`

该脚本使用 C-TCAR 特征训练 RandomForest router，并输出：

- `ctcar_features.csv`
- `ctcar_router_training_frame.csv`
- `ctcar_router.joblib`
- `ctcar_router_diagnostics.json`
- `ctcar_feature_importances.csv`
- `ctcar_router_predictions.csv`
- `ctcar_routing_report.md`

## 3. 当前实验设计

### 3.1 数据集

当前实验覆盖 5 个数据集：

| 数据集 | 类型 |
| --- | --- |
| `appliances_energy` | 能源 |
| `household_power` | 能源 |
| `citylearn_zone5` | 能源 |
| `opsd_germany` | 能源 |
| `physionet_2012` | 非能源对照 |

### 3.2 模型与基线

当前实验覆盖 7 类模型或基线：

| 方法 | 类型 |
| --- | --- |
| `mean` | 传统统计基线 |
| `median` | 传统统计基线 |
| `locf` | 时间连续性基线 |
| `saits` | 深度插补模型 |
| `tefn` | 轻量神经模型 |
| `timemixerpp` | 时间序列神经模型 |
| `uniformtsv` | 基于 T5 的时间序列模型 |

### 3.3 缺失率

所有主要实验使用三个缺失率：

- 0.1
- 0.3
- 0.5

### 3.4 当前实验规模

当前已经完成 99 个实验组合：

- `mean/median/locf`：5 数据集 × 3 缺失率
- `saits`：5 数据集 × 3 缺失率
- `tefn`：5 数据集 × 3 缺失率
- `timemixerpp`：5 数据集 × 3 缺失率
- `uniformtsv`：3 能源数据集 × 3 缺失率

## 4. 当前实验结果

### 4.1 每个场景下的最优模型

| 数据集 | 缺失率 | 最优模型 | MAE | MSE | RMSE | MRE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| appliances_energy | 0.1 | locf | 0.1128 | 0.1550 | 0.3938 | 0.1119 |
| appliances_energy | 0.3 | locf | 0.1195 | 0.1609 | 0.4011 | 0.1188 |
| appliances_energy | 0.5 | locf | 0.1320 | 0.1720 | 0.4147 | 0.1309 |
| citylearn_zone5 | 0.1 | saits | 0.1333 | 0.0514 | 0.2267 | 0.1638 |
| citylearn_zone5 | 0.3 | saits | 0.1519 | 0.0660 | 0.2569 | 0.1868 |
| citylearn_zone5 | 0.5 | saits | 0.2122 | 0.1117 | 0.3342 | 0.2607 |
| household_power | 0.1 | saits | 0.1795 | 0.2085 | 0.4566 | 0.2481 |
| household_power | 0.3 | saits | 0.2056 | 0.2445 | 0.4944 | 0.2823 |
| household_power | 0.5 | saits | 0.2565 | 0.3035 | 0.5510 | 0.3524 |
| opsd_germany | 0.1 | saits | 0.1050 | 0.0276 | 0.1660 | 0.1151 |
| opsd_germany | 0.3 | saits | 0.1255 | 0.0389 | 0.1971 | 0.1376 |
| opsd_germany | 0.5 | saits | 0.1840 | 0.0775 | 0.2784 | 0.2023 |
| physionet_2012 | 0.1 | saits | 0.2694 | 0.2976 | 0.5455 | 0.3820 |
| physionet_2012 | 0.3 | saits | 0.3004 | 0.3624 | 0.6020 | 0.4243 |
| physionet_2012 | 0.5 | saits | 0.3485 | 0.4173 | 0.6460 | 0.4925 |

### 4.2 能源数据集平均 MAE

| 模型 | 能源数据集平均 MAE |
| --- | ---: |
| saits | 0.2308 |
| locf | 0.2330 |
| timemixerpp | 0.3558 |
| tefn | 0.7300 |
| median | 0.7594 |
| uniformtsv | 0.7782 |
| mean | 0.7905 |

从平均结果看，SAITS 和 LOCF 的整体差距非常小，但它们擅长的场景不同：

- LOCF 在 `appliances_energy` 上非常强。
- SAITS 在其余数据集上更稳定。

这说明简单地报告一个平均分数不够，必须进一步分析“哪个模型适合哪个场景”。

## 5. 模型路由分析

### 5.1 路由思想

本项目当前的路由任务可以定义为：

给定数据集类型、时间窗口长度、特征数、缺失率等信息，选择在该场景下 MAE 最低的模型。

这不是严格意义上的因果干预预测，但它与用户提供论文中的思想一致：能源系统任务中存在多个候选模型，不同场景下最优模型不同，因此需要学习或构建一个模型选择机制。

### 5.2 当前 router 结果

原始轻量决策树 router 使用数据集名称、特征数、窗口长度和缺失率等简单元信息。其诊断结果为：

| 指标 | 数值 |
| --- | ---: |
| Training accuracy | 1.000 |
| Leave-one-scenario-out accuracy | 1.000 |
| Leave-one-dataset-out accuracy | 0.600 |

解释：

- `Training accuracy=1.0` 表示 router 可以完全拟合当前 15 个数据集/缺失率场景。
- `Leave-one-scenario-out=1.0` 表示在当前小矩阵内部，按单场景留一验证仍能恢复最优选择。
- `Leave-one-dataset-out=0.6` 更严格，说明如果面对完全没见过的数据集，当前 router 的泛化能力还不足。

因此当前 router 可以作为项目阶段性成果，但还不能作为最终可靠的泛化模型。后续需要更多能源数据集、更多 seed、更多缺失机制和更丰富的数据元特征。

### 5.3 当前 C-TCAR router 结果

新增 C-TCAR router 使用 80 维 C-TCAR 特征加缺失率作为输入，使用 RandomForest 进行模型选择。当前诊断结果为：

| 指标 | 数值 |
| --- | ---: |
| Training accuracy | 1.000 |
| Leave-one-scenario-out accuracy | 1.000 |
| Leave-one-dataset-out accuracy | 0.800 |

与只使用简单元信息的 router 相比，C-TCAR router 的 leave-one-dataset-out accuracy 从 0.600 提升到 0.800。这个结果说明，统计、时序、频域和因果结构代理特征确实为模型选择提供了额外信息。

当前 C-TCAR router 学到的类别主要是 `locf` 和 `saits`：它将 `appliances_energy` 路由到 `locf`，将 `citylearn_zone5`、`household_power`、`opsd_germany` 和 `physionet_2012` 路由到 `saits`。这与 benchmark 中的最优模型一致。

## 6. 现在项目可以做到什么

经过本阶段修改后，项目现在可以完成以下任务：

1. **自动下载并处理多个公开能源数据集**
   可以直接使用 `appliances_energy`、`household_power`、`citylearn_zone5`、`opsd_germany`。

2. **运行多模型插补 benchmark**
   支持传统基线、SAITS、TEFN、TimeMixer++、UniFormTSV 等模型。

3. **对不同缺失率进行系统测试**
   当前支持 0.1、0.3、0.5，也可以通过脚本扩展。

4. **在 Apple Silicon 本机上运行实验**
   已适配 MPS，避免只能依赖 CUDA。

5. **自动汇总实验指标**
   所有 JSON 指标可以汇总成统一 CSV。

6. **自动生成模型路由分析**
   可以输出每个场景的最优模型、第二名模型、收益差距和轻量 router。

7. **抽取 C-TCAR 特征并训练 C-TCAR router**
   可以为每个数据集/缺失率场景生成 80 维 C-TCAR 表示，并基于这些特征训练 RandomForest 模型选择器。

7. **继续扩展到更大实验矩阵**
   脚本已经支持多 seed、更大样本量和更多模型，只是本机资源会限制运行速度。

## 7. 当前局限

尽管项目已经明显增强，但还存在以下局限：

1. **当前正式结果主要是单 seed**
   当前大部分结果使用 seed 42。虽然代码已经修复了随机种子控制，但要写成更强结论，还需要多 seed 重跑。

2. **UniFormTSV 尚未体现优势**
   UniFormTSV 能跑通，但在当前小样本设置下没有超过 SAITS 或 LOCF。这可能与样本量、训练轮次、backbone 冻结方式和超参数有关。

3. **TEFN 对 `n_fod` 极其敏感**
   默认 `n_fod=16` 会造成指数级内存膨胀，必须用较小配置才能在本机跑。

4. **当前仍是插补任务，不是完整因果干预预测任务**
   本项目已经加入 C-TCAR 风格的统计、时序、频域、因果结构和混杂代理特征，但还没有真实干预变量标签、do-operator 估计或反事实预测评估。

5. **router 的泛化能力还需更多数据集验证**
   当前 leave-one-dataset-out accuracy 为 0.6，说明面对完全未知数据集时仍有不确定性。

## 8. 后续可扩展方向

### 8.1 多 seed 大规模实验

建议使用：

```bash
EPOCH=50 PATIENCE=10 MAX_SAMPLES=20000 WINDOW_STRIDE=8 SEEDS="42 123 456" ./run_energy_benchmark.sh
```

这样可以得到更可靠的均值和方差。

### 8.2 增加更多能源数据集

可以继续加入：

- ETT 系列电力变压器数据
- Electricity Load Diagrams
- PEMS 交通数据作为跨领域对照
- 建筑能耗模拟数据
- 电网扰动或需求响应相关数据

### 8.3 增加缺失机制

当前主要是 MCAR 风格人工缺失。后续可以加入：

- block missing
- sensor outage
- feature-level missing
- time-of-day dependent missing
- event-driven missing

### 8.4 引入干预变量

为了更贴近用户提供论文，可以在 CityLearn 数据中进一步构造：

- storage action
- cooling/heating demand
- solar generation
- carbon intensity
- price
- weather shift

然后把插补框架扩展为“干预变量条件下的预测/反事实预测”。

### 8.5 优化 UniFormTSV

可以尝试：

- 增加样本量
- 调整 finetuning mode
- 使用 LoRA 或部分解冻
- 调整 patch length/stride
- 使用更合适的 T5 backbone
- 对能源数据做更强 prompt 或时间特征编码

## 9. 结论

本阶段工作将 UniFormTSV 从一个较简单的时间序列插补实验项目，扩展为一个面向能源系统的多数据集、多模型、多缺失率 benchmark 和模型路由分析框架。

最重要的发现是：不同能源时间序列场景下最优模型并不相同。`appliances_energy` 上 LOCF 显著最优，而 `citylearn_zone5`、`household_power`、`opsd_germany` 等场景下 SAITS 更稳。这个结果说明，项目后续不应只追求“训练一个更复杂的大模型”，而应围绕“如何根据数据场景选择模型”继续发展。

因此，当前项目已经具备了更强的实验说服力：它不仅展示了模型训练结果，还展示了模型选择的必要性，并初步形成了可复现的 router 分析流程。

## 参考数据源

- UCI Appliances Energy Prediction: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- UCI Individual Household Electric Power Consumption: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- CityLearn Climate Zone 5: https://github.com/citylearn-project/CityLearn/tree/v1.0.0/data/Climate_Zone_5
- Open Power System Data: https://data.open-power-system-data.org/time_series/
