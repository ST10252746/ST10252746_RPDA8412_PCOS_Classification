<div align="center">

# RPDA8412 RESEARCH PROJECT - PCOS Classification Pipeline

**A comparative evaluation of traditional and neural network models for Polycystic Ovary Syndrome classification**

![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.51.0-8A2BE2)
![Status](https://img.shields.io/badge/status-academic%20submission-blue)
![Use](https://img.shields.io/badge/use-research%20only-lightgrey)

</div>

---

## About

This repository holds the code, data files and outputs for a Postgraduate Diploma in Data Analytics research project (RPDA8412) at The Independent Institute of Education.

The study compares three traditional models (Logistic Regression, Random Forest and a Support Vector Machine with an RBF kernel) with a Multi-Layer Perceptron for PCOS classification. It reports performance on a held-out test set and on a separately sourced external dataset, and it compares the explanations each model gives.

| | |
|---|---|
| **Author** | Theshara Narain (ST10252746) |
| **Supervisor** | Dr Janine Upton |
| **Programme** | Postgraduate Diploma in Data Analytics, The IIE |

> **Important:** the outputs of this project describe model behaviour on secondary data, two of three datasets being synthetic. They must not be used for diagnosis or for any clinical decision.

---

## Study design

| Step | Decision |
|---|---|
| Training and test data | Soares dataset, stratified 80/20 split, seed 42 |
| External validation | Dalvi dataset, never used to fit, tune or scale |
| Exploratory explanation | Moheddine dataset, SHAP only, using proxy inputs |
| Scaling | Min-max scaler fitted on the training partition |
| Class imbalance | SMOTE (k = 5) inside each cross-validation fold |
| Tuning | 5-fold stratified cross-validation, grid search scored on ROC-AUC |
| Primary outcome | ROC-AUC on the Dalvi dataset |
| Secondary outcomes | Recall, precision, specificity, F1-score, accuracy, missed cases, false alarms |
| Explanation | Logistic Regression coefficients, Random Forest importance, SHAP for the SVM and MLP |

---

## Repository structure

```text
.
├── PCOS_Classification_Pipeline.ipynb   # full pipeline with saved output
├── data/
│   ├── soares_pcos.csv                  # training and test data
│   ├── dalvi_pcos.csv                   # external validation data
│   └── moheddine_pcos.csv               # exploratory SHAP data
├── outputs/                             # tables and figures produced by the notebook
├── models/                              # fitted models saved by the notebook
├── data_checks.py                       # checks run on the data files outside the notebook
├──notebooks/
│  └── PCOS_Classification_Pipeline.ipynb # main project file 
├── data_checks_run.img                  # displays the data checks ran successfully
├── dataset_checksums.txt                # SHA-256 checksum of each data file
├── requirements.txt                     # packages and versions
├── .gitignore
└── README.md
```

---

## Data sources and licences

The files are secondary data from Kaggle. Each is shared here with attribution and under the licence stated by its source.

| File | Source | Records | Licence |
|---|---|---|---|
| `soares_pcos.csv` | Soares, L.S. (2025). *Polycystic Ovary Syndrome PCOS*. Kaggle | 3,000 | MIT |
| `dalvi_pcos.csv` | Dalvi, S. (2024). *PCOS Diagnosis Dataset*. Kaggle | 1,000 | CC0 1.0 |
| `moheddine_pcos.csv` | Moheddine, A. (2022). *PCOS_Dataset*. Kaggle. A copy of Kottarathil, P. (2020), *Polycystic ovary syndrome (PCOS)*, Kaggle | 541 | CC BY-NC-SA 4.0 (original source; the copy's page lists the licence as unknown) |

The Moheddine file is shared under the same CC BY-NC-SA 4.0 terms: attribution is required, use is non-commercial, and adaptations carry the same licence.

Soares and Dalvi are synthetic. Moheddine holds hospital records from 10 hospitals in Kerala, India, described by its source as anonymised.

### Checksums

The Kaggle pages give no version numbers, so each file is identified by its SHA-256 checksum (also in `dataset_checksums.txt`).

| File | SHA-256 |
|---|---|
| `soares_pcos.csv` | `7bf1ee6c94bf5ab70f856ec6add2e6be055ef7a4f28c10d07d3212ddc1c6b721` |
| `dalvi_pcos.csv` | `09c8224b43cdc47e31723906b66cae88c9327befd65d36c216b201131eb0aa71` |
| `moheddine_pcos.csv` | `b35d13e5e11bb9f4810aba949ee86aaa2ab134bab4456111035752b3cd658ed1` |

To check a file on Windows (PowerShell):

```powershell
Get-FileHash .\data\soares_pcos.csv -Algorithm SHA256
```

On Linux or macOS:

```bash
sha256sum -c dataset_checksums.txt
```

---

## Reproducing the analysis

**Environment:** Python 3.11.9 on Windows 10, random seed 42.

```bash
# 1. Clone the repository
git clone https://github.com/ST10252746/ST10252746_RPDA8412_PCOS_Classification.git
cd ST10252746_RPDA8412_PCOS_Classification

# 2. Create and activate an environment
python -m venv pcos_env
pcos_env\Scripts\activate          # Windows
# source pcos_env/bin/activate     # Linux or macOS

# 3. Install the packages
pip install -r requirements.txt

# 4. Check the data files against their checksums (see above)

# 5. Run the notebook from the top (Kernel > Restart Kernel and Run All)
jupyter lab PCOS_Classification_Pipeline.ipynb
```

The report values come from the saved run. Small differences are possible if package versions differ from `requirements.txt`.

To repeat the checks on the data files (label rule, overlap between Soares and Dalvi, split replication and Moheddine range checks), run from the repository root:

```bash
python data_checks.py
```

---

## Main packages

| Package | Version |
|---|---|
| pandas | 3.0.5 |
| NumPy | 2.4.6 |
| matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |
| imbalanced-learn | 0.14.2 |
| SHAP | 0.51.0 |
| seaborn | 0.13.2 |

---

## Limitations in brief

- Two of the three datasets are synthetic, and the Soares label follows a single rule on antral follicle count, so the Soares test set cannot separate the models.
- The scaler was fitted on the full training partition before cross-validation.
- Results come from one split, one fold assignment and one seed. ROC-AUC intervals are approximate and no significance test was run.
- The Moheddine analysis uses proxy inputs with changed meaning and scale and is exploratory.

The full account is in the report.

---

## Ethics

Ethical clearance was granted as a waiver for secondary studies by the IIE Ethics and Misconduct Committee for Student Research (reference EMCSREC/EMKNDN/025/2026, approved 13 July 2026). No identifiable personal information was used and no participants were contacted.