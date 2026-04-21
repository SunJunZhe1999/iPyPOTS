# UniFormTSV 能源 C-TCAR 扩展实验说明稿

## 摘要

本项目原本更接近一个时序缺失值填补与模型对比框架。结合《Learning to Route Models for Causal Intervention Prediction in Energy Systems》的思路后，我把项目扩展成一个能源时序场景下的“模型路由”实验系统：不仅训练或评估单个模型，而是根据数据集、缺失率、统计特征、频谱特征、时序自相关特征和因果代理特征，判断当前场景应该选哪个模型。

当前本机实验已覆盖 11 个数据集、9 类模型、4 个缺失率、2 个随机种子，共收集 514 条指标记录。C-TCAR 路由器在 44 个数据集/缺失率场景上达到了 0.75 的留一场景准确率和 0.6818 的留一数据集准确率。

## 原项目做到的事情

原项目主要提供了统一入口来运行时间序列缺失值填补模型，包括均值、中位数、LOCF、SAITS、TEFN、UniFormTSV、MOMENT、GPT4TS、TSLANet 等模型。它可以加载 TSDB/BenchPOTS 数据集，构造不同缺失率下的训练、验证、测试数据，并保存 MAE、MSE、RMSE、MRE 等指标。

但原项目的问题也比较明显：实验数据集偏少，默认模型矩阵不够完整，训练强度较低，缺少论文中强调的“根据场景选择模型”的路由机制，也没有 C-TCAR 这样的条件表征。

## 本次新增内容

我新增并验证了更多能源相关数据集，包括 Appliances Energy、Household Power、CityLearn Zone5、OPSD Germany、ETT H1/H2/M1/M2、Solar Alabama、Electricity Load Diagrams，再加上 PhysioNet 作为非能源但常用的缺失值填补基准。

我扩展了 benchmark runner，让它支持更大的模型矩阵、更高训练轮次、单次运行超时、整体时间预算、MPS/CUDA/CPU 后端识别，以及针对高维数据集的维度配置。

我为 UniFormTSV 和 MOMENT 加入了本地 PatchTST-style backbone，使它们在没有 HuggingFace 大模型或云 GPU 的情况下也能进行端到端训练。同时补上了高维输出投影，让 Solar 的 137 维和 ELD 的 370 维不会因为 `d_model=64` 发生输出维度不匹配。

我实现了 C-TCAR 特征抽取脚本，生成 80 维特征，包括统计特征、24 阶自相关特征、频谱特征、滞后依赖图结构特征和混杂代理特征。因为当前项目没有真实干预标签和 ATE ground truth，所以这里是 causal-aware proxy，不是完整 do-calculus 复现。

我新增了两套路由分析：一套使用数据集元信息和缺失率，一套使用 C-TCAR 特征和缺失率。二者都会训练 RandomForest 路由器，输出最佳模型表、诊断指标、特征重要性、预测结果和 markdown 报告。

## 当前实验规模

当前正式结果位于：

`output/imputation/mps/energy_benchmark_200`

核心结果如下：

| 项目 | 数量 |
| --- | ---: |
| 指标记录 | 514 |
| 数据集 | 11 |
| 模型 | 9 |
| 缺失率 | 4 |
| 随机种子 | 2 |
| 场景数 | 44 |
| C-TCAR 特征行 | 44 |

模型包括：`mean`、`median`、`locf`、`saits`、`tefn`、`uniformtsv`、`moment`、`gpt4ts`、`tslanet`。

## 主要结论

第一，项目现在不再只是“训练一个模型”，而是能做模型选择。实验显示 LOCF 在大量连续能源负载数据上非常强，SAITS 在 PhysioNet、Household Power 等场景中更有优势，GPT4TS 在若干小规模覆盖场景中表现突出。

第二，C-TCAR 路由机制已经接入并可运行。当前 C-TCAR 路由器的留一场景准确率为 0.75，留一数据集准确率为 0.6818，和元数据路由器持平。这说明它已经能学习到“什么场景该选什么模型”的规律，但还需要更多真实干预标签和更多数据集来进一步体现因果表征的优势。

第三，深度模型不一定在所有能源场景上压过简单模型。尤其在 ELD 等高维负载数据上，简单的 LOCF/统计基线非常强。这是一个重要结果：项目的说服力来自模型路由，而不是强行宣称复杂模型永远最好。

第四，当前 Mac 本机可以完成中等规模复现和方法验证，但 TimeMixer++、T5 backbone、多 seed 大规模 GPT4TS/TSLANet 补全更适合放到 LightCC GPU 上继续跑。

## 局限与下一步

当前 C-TCAR 是因果感知代理表征，不等于论文中完整的干预预测设置。要做到更强复现，需要接入真实 intervention/action/outcome 标签，构造 ATE 或 counterfactual ground truth，并在云 GPU 上补齐 GPT4TS、TimeMixer++、T5/encoder-decoder backbone、多 seed 和更长 epoch。

下一步如果接入 LightCC，可以直接复用 `run_ctcar_200_benchmark.sh` 和 `run_energy_benchmark.sh`，把 `TRANSFORMER_BACKBONE` 切回 `t5-small` 或更大的时间序列 foundation model，并增加 `SEEDS`、`EPOCH`、`MAX_SAMPLES`，形成更接近论文级别的大规模实验。
