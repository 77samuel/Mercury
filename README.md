# MERCURY: Measuring Misconception Elasticity in Small Language Models through Graduated Correction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-orange.svg)](https://pytorch.org/)

**Authors:** Samuel Stephen, R. Vignesh  
**Affiliation:** Karunya Institute of Technology and Sciences, Coimbatore, Tamil Nadu, India  
**Paper:** *MERCURY: Measuring Misconception Elasticity in Small Language Models through Graduated Correction*

---

## Overview

MERCURY is a standardised evaluation framework for measuring **correction susceptibility** in open-weight language models. It introduces:

- **Correction Ladder (L0–L4):** A four-level protocol that adds exactly one evidential component per level (bare assertion → explanation → authority → multi-source corroboration)
- **Correction Knowledge Bank (CKB v1.0):** 48 curated misconception items with verified factual content, anchored to MythBench v1.0
- **Minimum Correction Level (MCL):** The first ladder level at which a model answers correctly
- **Revision Rate (RR):** Proportion of held misconceptions corrected at any level L1–L4
- **Misconception Resistance Index (MRI):** Item-level mean MCL across models

### Key Findings

| Model | Baseline Acc. | Revision Rate | Mean MCL |
|---|---|---|---|
| TinyLlama-1.1B | 58.3% | 50.1% | 1.10 |
| Qwen2.5-1.5B | 22.9% | 83.7% | 1.52 |
| Phi-2 | 45.8% | 65.3% | 1.18 |
| Qwen2.5-7B | 64.6% | 70.7% | 1.25 |
| Gemma-2-9B-IT | 27.1% | 85.7% | 1.07 |

- All models: significant ladder effect (Cohen's h ≥ 1.08, large; exact McNemar p < 0.05)
- L1 (bare assertion) accounts for **63–100% of net positive accuracy gain** across all models
- Correction susceptibility is **not monotonically related to baseline accuracy or model size**

---

## Repository Structure

```
MERCURY/
├── README.md
├── requirements.txt
├── correction_knowledge_bank.json     # CKB v1.0 (SHA-256: 0db3b3b4...)
├── MythBench_v10.json                 # MythBench benchmark (48 items)
├── experiments/
│   ├── mercury_pilot.ipynb            # Original 4-model pilot (L0–L4)
│   ├── mercury_experiments_v5.py      # E1 adversarial, E3 TruthfulQA, E4 robustness, E5 MCL
│   └── mercury_new_models_v2.py       # Gemma-2-9B-IT pilot + intervention order
├── results/
│   ├── MERCURY_Results_v2.xlsx        # All results consolidated (10 sheets)
│   ├── results_tinyllama.csv
│   ├── results_qwen15b.csv
│   ├── results_phi2.csv
│   ├── results_qwen7b.csv
│   ├── results_gemma2_9b.csv
│   ├── adversarial_results.csv
│   ├── truthfulqa_results.csv
│   ├── robustness_v2_tinyllama.csv
│   ├── robustness_v2_qwen15b.csv
│   ├── robustness_v2_phi2.csv
│   ├── robustness_v2_qwen7b.csv
│   ├── robustness_v3_tinyllama.csv
│   ├── robustness_v3_qwen15b.csv
│   ├── robustness_v3_phi2.csv
│   ├── robustness_v3_qwen7b.csv
│   ├── new_models_pilot.csv
│   ├── new_models_intervention.csv
│   ├── mcl_distribution.csv
│   ├── statistics_results.csv
│   └── statistics_new_models.csv
└── config/
    ├── experiment_config.json
    └── experiment_config_new_models.json
```

---

## Reproducibility

All experiments use:
- **Seed:** 42
- **Decoding:** Greedy (temperature = 0.0, max_new_tokens = 32)
- **Quantisation:** 4-bit NF4 (BitsAndBytes ≥ 0.46.1, double quantisation)
- **CuDNN:** Deterministic mode (`torch.backends.cudnn.deterministic = True`)
- **Hardware:** NVIDIA Tesla T4 (16 GB VRAM), CUDA 12.8
- **CKB SHA-256:** `0db3b3b4e2821f86c08b72ac92242854d1fca1927ff90b61da29456e8fa8294a`

See `config/experiment_config.json` for the full environment specification.

---

## Installation

```bash
git clone https://github.com/77samuel/MERCURY.git
cd MERCURY
pip install -r requirements.txt
```

---

## Running Experiments

### Step 1: Pilot (4 original models, L0–L4 on MythBench)
Open `experiments/mercury_pilot.ipynb` in Kaggle or a Jupyter environment with GPU access.

### Step 2: Extended experiments (E1, E3, E4, E5)
```bash
# Paste mercury_experiments_v5.py as a new cell in the pilot notebook
# or run directly:
python experiments/mercury_experiments_v5.py
```

### Step 3: New model experiments (Gemma-2-9B-IT + intervention order)
```bash
python experiments/mercury_new_models_v2.py
```

---

## Data

- **MythBench v1.0:** Also available on Zenodo — https://zenodo.org/doi/10.5281/zenodo.20558849
- **Kaggle dataset path:** `/kaggle/input/datasets/samuelstephen77/mercury-ckb/`
- All result CSVs are in `results/`; the consolidated Excel is `results/MERCURY_Results_v2.xlsx`

---

## Models Evaluated

| Model | HuggingFace ID | Params |
|---|---|---|
| TinyLlama-1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B |
| Phi-2 | `microsoft/phi-2` | 2.7B |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | 7B |
| Gemma-2-9B-IT | `google/gemma-2-9b-it` | 9B |

---

## Citation

```bibtex
@article{stephen2025mercury,
  title={MERCURY: Measuring Misconception Elasticity in Small Language Models through Graduated Correction},
  author={Stephen, Samuel and Vignesh, R.},
  journal={[Journal Name]},
  year={2025},
  institution={Karunya Institute of Technology and Sciences}
}
```

---

## License

MIT License. See LICENSE file for details.

---

## Contact

Samuel Stephen — samuels24@karunya.edu.in  
R. Vignesh — vignesh@karunya.edu
