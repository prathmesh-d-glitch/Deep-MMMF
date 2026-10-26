# Deep MMMF: Multi-Criteria Deep Maximum Margin Matrix Factorization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced collaborative filtering recommendation engine implementing **Deep Maximum Margin Matrix Factorization (Deep MMMF)**, **Bayesian Personalized Ranking (BPR)**, and **Coupled Threshold Embeddings** for multi-criteria hotel recommendation on the TripAdvisor benchmark.

---

## 🌟 Key Results & Breakthroughs

Evaluated against the published TripAdvisor Multi-Criteria Deep Neural Network benchmark (*Nassar et al., Knowledge-Based Systems, 2020*):

| Metric | Direction | TripAdvisor DNN Baseline | Single Deep MMMF | Multi-Criteria Deep MMMF (Ours) | Relative Gain |
|---|:---:|:---:|:---:|:---:|:---:|
| **Overall MAE** | $\downarrow$ | 0.7552 | 0.4685 ± 0.0267 | **0.3600 ± 0.0150** | **-52.33% Error Reduction** |
| **Overall NMAE** | $\downarrow$ | N/A | 0.2928 ± 0.0167 | **0.2250 ± 0.0094** | **Substantial margin improvement** |
| **MAP** | $\uparrow$ | 0.4643 | 0.6913 ± 0.0336 | **0.7214 ± 0.0353** | **+55.37% Precision Boost** |
| **FCP** | $\uparrow$ | 0.5126 | 0.5401 ± 0.0211 | **0.5410 ± 0.0293** | **+5.54% Ranking Concordance** |

---

## 📚 Foundational Papers

This work synthesizes three seminal methodologies:
1. **Multi-Criteria Collaborative Filtering**: Nassar et al., *A novel deep multi-criteria collaborative filtering model for recommendation system*, Knowledge-Based Systems (2020).
2. **Deep Maximum Margin Matrix Factorization**: Kumar et al., *Deep Maximum Margin Matrix Factorization*, Scientific Reports (2026).
3. **Bayesian Personalized Ranking**: Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI (2009).

For full theoretical analysis, architecture diagrams, and mathematical formulation, see **[project.md](project.md)** and **[PROJECT_REPORT.md](PROJECT_REPORT.md)**.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph Inputs ["Input Identifiers"]
        UID["User ID (i)"]
        IID["Item ID (j)"]
    end

    subgraph UserStream ["User Latent Path"]
        UID --> UE["Independent Embeddings U^(1..7)"]
        UE --> UTR["Psi_1 (MLP + Residual)"]
        UTR --> Utilde["Transformed Latent Vectors ~U^(1..7)"]
    end

    subgraph ItemStream ["Item Latent Path"]
        IID --> VE["Independent Embeddings V^(1..7)"]
        VE --> VTR["Psi_2 (MLP + Residual)"]
        VTR --> Vtilde["Transformed Latent Vectors ~V^(1..7)"]
    end

    Utilde & Vtilde --> Dot["Aspect Dot Products s_ij^(1..7)"]

    subgraph AspectHinge ["Criteria Margin Separation"]
        TH_CRIT["Coupled Thresholds:<br/>Theta_{i,r}^(c) = E_{i,r}^(base) + C_{c,r}^(crit)"] --> PSI3["Psi_3 Transform"]
        PSI3 & Dot --> L_CRIT["Criteria Smooth Hinge Loss"]
    end

    subgraph Aggregation ["Attention & Ranking Aggregation"]
        Dot --> ATT["Dynamic Attention:<br/>alpha_{ij}^(c) = Softmax(v^T tanh(W z))"]
        ATT --> AGG["Ranking Aggregator MLP"]
        AGG --> SHAT["Continuous Score S_hat_{ij}"]
    end

    subgraph Objectives ["Dual Optimization & Inference"]
        SHAT --> BPR["BPR Pairwise Ranking Loss"]
        SHAT --> OV_THRESH["Coupled Overall Thresholds"]
        OV_THRESH --> L_OVERALL["Overall Smooth Hinge Loss"]
        OV_THRESH --> RATINGS["Discrete Rating Prediction"]
    end
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/prathmesh-d-glitch/Deep-MMMF.git
cd Deep-MMMF
pip install -r requirements.txt
```

### Running Experiments
```bash
# Run 5-fold cross validation on Multi-Criteria Deep MMMF (default demo)
python main.py --demo

# Run Single-Criterion Deep MMMF
python main.py --demo --model single --generalization weak

# Run Cold-Start Strong Generalization (new users with frozen item representations)
python main.py --demo --model single --generalization strong

# Run with custom dataset
python main.py --data_path path/to/dataset.csv --num_ratings 5 --epochs 100
```

---

## 📁 Repository Structure

```
├── models.py              # DeepMMMF, MultiCriteriaDeepMMMF, CriteriaRankingAggregator, Smooth Hinge
├── trainer.py             # Training loops, Cosine Annealing, Strong Gen, 6-Metric Evaluation Suite
├── main.py                # CLI runner and 5-fold cross-validation pipeline
├── data_loader.py         # Data loading, integer encoding, K-fold splitting
├── prepare_data.py        # TripAdvisor synthetic generator & JSON parser
├── sample_tripadvisor.csv # Demo multi-criteria TripAdvisor dataset
├── requirements.txt       # Dependencies
├── project.md             # Exhaustive end-to-end technical documentation & paper synthesis
├── PROJECT_REPORT.md      # Comprehensive research & empirical report
└── EXPLANATION.md         # Step-by-step conceptual walkthrough & guide
```

---

## 👥 Authors & Contributors

- **Prathmesh Dubey** ([@prathmesh-d-glitch](https://github.com/prathmesh-d-glitch))
- **Aksh Dhingra** ([@adk2004](https://github.com/adk2004))
