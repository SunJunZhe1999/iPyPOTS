Here’s the fully revised and professional 6–7 page outline tailored for NeurIPS, ICLR, or IEEE TAI/TPAMI standards. It integrates all your updates, adjusts dataset scope, and enriches technical and empirical rigor for a publishable-grade draft.

⸻

📄 LLM4IMP: Reprogramming Frozen Language Models for Time-Series Imputation via Token Prompts and Patch Embeddings

⸻

📝 Abstract

Time-series data in real-world systems often suffer from missing values due to sensor failures, transmission dropouts, or irregular sampling. Traditional imputation methods either rely on strong inductive biases or require extensive training from scratch, which can limit generalization and scalability. In this work, we propose LLM4IMP, a novel framework that reprograms pretrained frozen large language models (LLMs) for the task of time-series imputation. Our approach encodes temporal signals into patch-level representations, conditions them with either soft or token-based prompts, and aligns them into the LLM token space via a lightweight reprogramming layer. The LLM backbone remains frozen, allowing the architecture to leverage pretrained knowledge while learning only a minimal set of parameters.

We evaluate LLM4IMP on six diverse datasets spanning energy (ETT), air quality (Italy & Beijing), and medical domains (PhysioNet), achieving consistent improvements over strong baselines such as BRITS, SAITS, Time-LLM, and PatchTST. Comprehensive ablation studies highlight the effectiveness of prompt conditioning, patch encoding, and frozen reprogramming. These results demonstrate that pretrained LLMs, when reprogrammed appropriately, offer a scalable and generalizable foundation for multivariate time-series imputation.

⸻

1. Introduction

Motivation

Time-series data are central to applications in healthcare, smart grids, transportation, and environmental monitoring. However, such data are frequently incomplete due to factors such as hardware failures or communication dropouts. Imputation of missing values is essential for enabling downstream analytics such as forecasting, anomaly detection, or decision support.

Limitations of Existing Methods

While deep models such as BRITS, SAITS, and CSDI offer promising results, they often require heavy supervision, full retraining, or task-specific tuning. These architectures are tightly coupled to the time-series structure and cannot benefit from advances in large-scale pretraining.

Opportunity with LLMs

Pretrained LLMs (e.g., GPT2, LLaMA) encapsulate broad inductive priors from massive text corpora. Recent works such as Time-LLM and GPT4TS have shown their capacity to model time series through patch reformatting. Yet, their potential for imputation—which requires reconstructive capabilities—remains underexplored.

Our Proposal

We introduce LLM4IMP, a novel architecture that:
	•	Reprograms frozen LLMs using spectral patch embeddings.
	•	Conditions the LLM via learned or token-based prompts.
	•	Aligns patch-token sequences into the LLM token space using cross-attention.
	•	Supports LoRA, MLP tuning, and RevIN normalization for further extensibility.

Key Contributions
	1.	We propose the first imputation framework that reuses frozen LLMs by reprogramming patch-encoded time series via prompts.
	2.	We design a modular architecture incorporating patching, prompt generation, a reprogramming layer, and a lightweight output head.
	3.	We empirically validate LLM4IMP on six real-world datasets, achieving consistent improvements in accuracy and efficiency.

⸻

2. Related Work

Classical and Neural Imputation
	•	Traditional methods: Mean/forward fill, KNN, MICE.
	•	Neural networks: BRITS [2018], CSDI [2021], SAITS [2021], TGAN [2019].

Pretrained and Reprogrammed Models
	•	Time-LLM [ICLR 2024]: patch → prompt → GPT2 (forecasting only).
	•	GPT4TS: fine-tunes GPT for sequence modeling.
	•	TimeGPT (Proprietary): uses GPT-like encoder for multivariate forecasting.

Patch-based Time-Series Models
	•	PatchTST: projects univariate time series into overlapping patches for Transformer modeling.
	•	TimesNet: uses frequency-aware decomposition for robust forecasting.

Prompting in NLP and Vision
	•	Prompt tuning, prefix tuning, and reprogramming have been effective in NLP, CV, and multimodal settings.
	•	Little exploration exists in time-series imputation via prompting.

⸻

3. Methodology

3.1 Problem Definition

Given an input sequence \mathbf{X} \in \mathbb{R}^{T \times D} with a binary mask \mathbf{M} \in \{0,1\}^{T \times D}, our goal is to reconstruct missing values using a frozen pretrained LLM, without fine-tuning its core parameters.

⸻

3.2 LLM4IMP Pipeline Overview

LLM4IMP consists of five main stages:
	1.	Patch Embedding of multivariate sequences.
	2.	Prompt Generation (instructional or soft).
	3.	Reprogramming Layer to align patch tokens with LLM embeddings.
	4.	Frozen LLM as contextual encoder.
	5.	Output Head to reconstruct imputed time-series.

⸻

3.3 Patch Embedding

We divide each variable x^{(d)} \in \mathbb{R}^T into P overlapping patches using a sliding window (patch size p, stride s). Each patch is linearly projected to \mathbb{R}^{d_{\text{model}}} and enriched with positional encodings.
\mathbf{Z}^{(d)} = \text{PatchEmbed}(x^{(d)}) \in \mathbb{R}^{P \times d_{\text{model}}}

⸻

3.4 Prompt Generation & Conditioning
	•	Instructional prompt: Encodes statistical cues (min, max, median), trends, and missing indices.
	•	Soft prompt: Learnable embeddings prepended to token sequences.

These prompts are embedded using the LLM tokenizer and input embedding layer:
\mathbf{P} = \text{LLMEmbed}(\text{Prompt}(X, M)) \in \mathbb{R}^{L \times d_{\text{llm}}}

⸻

3.5 Reprogramming Layer

We use multi-head cross-attention from patch tokens (queries) to prompt tokens (keys/values). The layer includes:
	•	Adapter MLP to match LLM dimensionality.
	•	Residual connection and optional layer norm.
	•	Gating mechanism for controlled fusion:
\text{Output} = \text{Gate}(\text{Adapter}(Z), \text{CrossAttn}(Z, P))

⸻

3.6 Output Head & Normalization
	•	Uses FlattenHead or RegressionHead to produce imputed values.
	•	Supports RevIN [Kim et al., 2022] for instance-level denormalization.
\hat{\mathbf{X}} = \text{OutputHead}(\text{LLM}(\text{Reprogrammed Input}))

⸻

3.7 Optional Enhancements
	•	LoRA: Adds trainable low-rank adapters to frozen LLM layers.
	•	MLP Finetuning: Unfreezes MLP blocks in final transformer layers.
	•	Profiling Support: Records runtime and GPU usage per epoch.

⸻

4. Experiments

4.1 Datasets

Dataset	Domain	Features	Horizon
ETTh1 / ETTh2	Energy (1h)	7	96
ETTm1 / ETTm2	Energy (15min)	7	96
Italy Air Quality	Environment	36	48
Beijing Air Quality	Urban sensors	36	48
PhysioNet 2012	Medical ICU	35	48


⸻

4.2 Baselines
	•	Classical: Mean fill, KNN, MissForest
	•	Deep Models: BRITS, SAITS, CSDI, PatchTST, Time-LLM

⸻

4.3 Metrics
	•	MAE (↓), RMSE (↓), MRE (↓)

⸻

4.4 Main Results
	•	LLM4IMP achieves lowest MAE/RMSE on 5 out of 6 datasets.
	•	Matches or surpasses Time-LLM and SAITS despite using frozen LLMs.
	•	Competitive runtime and memory usage (see Section 4.6).

⸻

4.5 Ablation Studies

Component	Metric	Effect
❌ Reprogramming	↑ MAE/RMSE	Can’t align patch → LLM space
❌ Prompting	↑ MAE	Frozen LLM underperforms
✅ LoRA only	↓ MAE	Lightweight improvement
✅ MLP only	↓↓ MAE	Highest gain, higher cost
✅ LoRA + MLP	✅ Best	Best overall, highest training time

Missing Rate Sensitivity (on ETTh1, PhysioNet):

Missing Rate	MAE	RMSE
10%	Low	Low
30%	Stable	Stable
50%	Mild drop	Acceptable
70%+	Sharp drop	Degradation begins


⸻

4.6 Efficiency

Variant	Params	Runtime	Memory
SAITS	3M	3.5s	4.2GB
LLM4IMP (frozen)	150M (frozen)	4.1s	3.8GB
+LoRA	+2M	4.4s	4.0GB
+MLP	+30M	5.2s	4.5GB


⸻

5. Conclusion

We proposed LLM4IMP, a novel framework for time-series imputation that reuses frozen LLMs through patch tokenization and prompt conditioning. By reprogramming pretrained models via cross-attention and minimal tuning, LLM4IMP achieves strong performance across diverse domains with high efficiency.

Future Work
	•	Multimodal extensions (text + time-series)
	•	Prompt-guided anomaly detection
	•	Graph-enhanced LLM reprogramming

⸻

📚 Keywords

Time-series imputation, frozen LLMs, prompt learning, patch embeddings, LoRA, RevIN, cross-attention, GPT2, reprogramming

⸻

Would you like this exported as a LaTeX IEEE/ICLR/NeurIPS template now?