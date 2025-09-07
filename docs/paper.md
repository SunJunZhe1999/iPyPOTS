# Research Proposal  
**Uncertainty-Aware Time-Series Imputation for Irregular Medical Data**

---

## 1. Background & Motivation
- Medical time series (ICU monitoring, EHR, wearable sensors) are **irregular and incomplete** due to device failures, recording errors, and cost constraints.  
- Standard imputation models (attention-based, RNNs, transformer-inspired) recover missing values but **ignore the reliability of their predictions**.  
- In high-stakes domains like **healthcare**, deterministic imputations are insufficient:
  - Example: A blood pressure value imputed as “120” without confidence could be misleading.  
  - Clinicians need to know if the model is **confident or uncertain**.

---

## 2. Research Gap
- Existing imputation models: BRITS, SAITS, Transformer-based methods.  
- Recent works on LLM-based imputation.  
- **Limitation:** All produce point estimates only.  
- **Missing piece:** *Uncertainty quantification* in multivariate, irregular medical time series.  

---

## 3. Proposed Innovation
We propose the **first uncertainty-aware imputation framework** for irregular medical time series:

1. **Probabilistic Imputation Head**  
   - Replace regression head with a **distributional output** (mean + variance).  
   - Model missing entries as samples from a Gaussian distribution \( \mathcal{N}(\mu, \sigma^2) \).  

2. **Uncertainty Calibration Mechanism**  
   - Ensure predicted variances align with actual error distribution.  
   - Evaluate with metrics such as **Negative Log Likelihood (NLL)** and **Expected Calibration Error (ECE)**.  

3. **Downstream Clinical Validation**  
   - Demonstrate that uncertainty-aware imputations improve reliability in **mortality prediction** and **length-of-stay estimation**.

---

## 4. Methodology
- **Backbone:** Transformer-style imputation model with temporal attention.  
- **Extension:** Add **Gaussian likelihood head** (outputs both \(\mu\) and \(\sigma^2\)).  
- **Training Objective:**
  \[
  \mathcal{L} = \frac{1}{2\sigma^2}(x - \mu)^2 + \frac{1}{2}\log\sigma^2
  \]

- **Datasets:**  
  - PhysioNet2012 (ICU mortality benchmark).  
  - MIMIC-III subset (clinical EHR with irregular missingness).  

- **Baselines:**  
  - Deterministic imputers: BRITS, SAITS, Transformer-based models.  
  - Compare against our uncertainty-aware variant.  

---

## 5. Expected Contributions
1. **Novelty:** First imputation framework integrating **uncertainty estimation** in irregular multivariate time series.  
2. **Safety & Trustworthiness:** Better suited for **clinical decision support**.  
3. **Evaluation Protocol:** New benchmark combining imputation error + uncertainty calibration + downstream tasks.  
4. **Extensibility:** Method applicable to **energy, finance, and IoT** domains in future work.  

---

## 6. Conference Paper Scope
- **Focus:** Methodology + ICU experiments (PhysioNet2012 + MIMIC-III).  
- **Results:** Demonstrate improved imputation + calibrated uncertainty.  
- **Impact:** Step toward **trustworthy AI for healthcare time series**.  

---

## 7. Future Journal Extension
- Extend to multi-domain datasets (wearables, traffic, energy).  
- Explore **Bayesian ensembles, variational layers, and quantile regression**.  
- Broader clinical validation across tasks (readmission, treatment effects).  

---

## 🔑 One-Sentence Pitch
**We go beyond imputing missing values: we make imputation *trustworthy* by quantifying uncertainty in medical time series, enabling safer AI for healthcare.**