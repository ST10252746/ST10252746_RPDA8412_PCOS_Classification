CONFIRMED DAY 1 FACTS FOR REPORT:
- Soares: 3,000 records, 6 columns, 0 missing, 0 duplicates, 80.0/20.0 class split
- Dalvi: 1,000 records, 6 columns, 0 missing, 0 duplicates, 80.1/19.9 class split
- Moheddine: 541 records, 44 columns, 2 missing values, 0 duplicates, 67.3/32.7 class split
- Moheddine object columns: II beta-HCG(mIU/mL) and AMH(ng/mL) — numeric stored as strings
- Soares/Dalvi schema: identical — external validation can proceed without feature engineering
- SMOTE justified: 80% majority class confirmed in primary dataset
- Ethical clearance: confirmed, reference EMCSREC/EMKNDN/025/2026, approved 13 July 2026

-Day 4 findings to note:
LR DALVI NOTE: The Logistic Regression model shows strong sensitivity
on Dalvi (0.9548) but very poor specificity (0.5331) and precision
(0.3369). FP=374 out of 801 No-PCOS records. This reflects
distributional shift between Soares and Dalvi, not model failure.
The Soares test set performance (ROC-AUC=1.0000) must be contextualised
as likely a product of synthetic data with clean class boundaries.
Watch whether RF, SVM, and MLP show the same pattern on Dalvi.

Addtional note: The odds ratios for Antral_Follicle_Count and Testosterone_Level are astronomically large (1.4 x 10^14 and 6.6 x 10^10). This is a known consequence of fitting LR with C=100 on near-perfectly separable data. The observation cell above addresses this directly. 
Reminder for me: Do not report the raw odds ratio numbers in the research report without the above explanation.

Day 5 findings to note:
RF DALVI NOTE: Random Forest produces FP=559 on Dalvi vs LR FP=374.
Both models produce identical FN=9. RF has higher Dalvi ROC-AUC
(0.8743 vs 0.8463) but worse specificity at default threshold (0.3021
vs 0.5331). Non-linear boundary with max_depth=None does not help
generalisation on Dalvi - it worsens false positive rate.

RANKING DIVERGENCE: RF ranks Menstrual_Irregularity second and
Testosterone_Level third. LR ranks them in reverse order. Both agree
Antral_Follicle_Count is first. This divergence is explained by
Gini importance's sensitivity to the clean binary split on
Menstrual_Irregularity. 
Reminder for me: Report this in the interpretability comparison.

Day 6 findings to note:
SVM GRID SEARCH: Best params C=0.1, gamma='scale', CV AUC=1.0000 (20.2s, 12 combinations).
C=0.1 is the smallest value in the grid - the model preferred the widest margin.
gamma='scale' adapts the kernel coefficient to training data variance.

SVM TEST SET: Accuracy=0.9967, Recall=1.0000, ROC-AUC=1.0000 (TN=478, FP=2, FN=0, TP=120).
Identical to LR on test set. All three traditional models saturated on Soares.

SVM DALVI: Accuracy=0.8050, Recall=0.8442, ROC-AUC=0.9024 (TN=637, FP=164, FN=31, TP=168).

SVM ERROR PROFILE vs LR and RF on Dalvi:
- FP=164 (SVM) vs FP=374 (LR) vs FP=559 (RF) - SVM most conservative
- FN=31 (SVM) vs FN=9 (LR) vs FN=9 (RF) - SVM misses most PCOS cases
- SVM trades recall for precision on Dalvi due to C=0.1 wide-margin boundary
- Clinically: FN=31 is the more concerning error - missed diagnoses

SVM DALVI ROC-AUC=0.9024 is highest of three models to date (LR=0.8463, RF=0.8743).
This means at adjusted thresholds, SVM orders positives and negatives better than LR/RF.
Reminder for me: The default threshold result (FN=31) and the AUC result (best of three)
tell different stories. Both must be reported and explained in the Findings chapter.

FN PATTERN: LR and RF both produce FN=9 on Dalvi (same 9 records - to be verified).
SVM produces FN=31 - a qualitatively different boundary, not a replication of LR/RF errors.
No native feature importance for SVM. SHAP KernelExplainer applied in Day 9.

Day 7 findings to note:
MLP GRID SEARCH: Best params hidden_layer_sizes=(128, 64), alpha=0.0001,
learning_rate_init=0.01. CV AUC=1.0000. Completed in 6.8 seconds, 14 epochs
(early stopping). Smallest regularisation and largest architecture preferred -
consistent with clean Soares separability, not a general finding.

MLP TEST SET: Accuracy=0.9967, Recall=1.0000, ROC-AUC=1.0000
(TN=478, FP=2, FN=0, TP=120). Identical to LR and SVM. All four models
saturated on Soares test set.

MLP DALVI: Accuracy=0.6590, Recall=0.9246, ROC-AUC=0.8607
(TN=475, FP=326, FN=15, TP=184).

DALVI AUC RANKING (final): SVM (0.9024) > RF (0.8743) > MLP (0.8607) > LR (0.8463)

MLP ERROR PROFILE vs all models on Dalvi:
- FP=326 (MLP) vs FP=164 (SVM) vs FP=374 (LR) vs FP=559 (RF)
- FN=15 (MLP) vs FN=31 (SVM) vs FN=9 (LR) vs FN=9 (RF)
- MLP sits between SVM and LR/RF on the recall-precision trade-off
- Worst combined error profile for clinical screening: more FNs than LR/RF,
  more FPs than SVM

KEY FINDING FOR DISCUSSION: MLP does not outperform traditional models on
Dalvi AUC, ranks third. Neural network complexity is not justified by
performance gains in this study. This directly addresses the central
research question.
Reminder for me: State this carefully - the finding is specific to this
dataset and study scope. Do not overclaim.

Day 8 findings to note:
COMPARATIVE TABLE CONFIRMED - all values verified from pipeline output.

TEST SET SUMMARY: RF perfect (all metrics=1.0). LR, SVM, MLP identical
(Accuracy=0.9967, FP=2, FN=0). Test set provides no basis for model
differentiation. Do not use test set metrics to argue model superiority.

DALVI SUMMARY (full metrics):
  LR  : Acc=0.617, Prec=0.337, Rec=0.955, Spec=0.533, F1=0.498, AUC=0.8463, TN=427, FP=374, FN=9,  TP=190
  RF  : Acc=0.432, Prec=0.254, Rec=0.955, Spec=0.302, F1=0.401, AUC=0.8743, TN=242, FP=559, FN=9,  TP=190
  SVM : Acc=0.805, Prec=0.506, Rec=0.844, Spec=0.795, F1=0.633, AUC=0.9024, TN=637, FP=164, FN=31, TP=168
  MLP : Acc=0.659, Prec=0.361, Rec=0.925, Spec=0.593, F1=0.519, AUC=0.8607, TN=475, FP=326, FN=15, TP=184

DALVI AUC RANKING:    SVM (0.9024) > RF (0.8743) > MLP (0.8607) > LR (0.8463)
DALVI FN RANKING:     LR=RF (9) < MLP (15) < SVM (31)
DALVI FP RANKING:     SVM (164) < MLP (326) < LR (374) < RF (559)

KEY FINDING: AUC ranking and FN ranking disagree. SVM ranks first on
AUC but last on FN. LR and RF rank last on AUC but first on FN.
This divergence between threshold-independent and threshold-dependent
metrics is the central analytical tension for the Discussion chapter.
Reminder for me: Frame this carefully - neither ranking is universally
correct. The appropriate metric depends on clinical priorities.

RF FALSE POSITIVE RATE on Dalvi = 559/801 = 69.8%. This is extreme and
must be reported with full context: it reflects distributional shift,
not model failure per se. The Soares training data did not expose the
model to Dalvi-range AFC and BMI values.

Figure 17 (combined ROC) and Figure 18 (error profile bar chart) are
the two primary figures for the Findings chapter comparison section.

SHAP SOARES CROSS-METHOD RANKING TABLE (confirmed from figures):

| Feature                | LR Coeff | RF Gini | SVM SHAP | MLP SHAP |
|------------------------|----------|---------|----------|----------|
| Antral_Follicle_Count  |    1     |    1    |    2     |    1     |
| Menstrual_Irregularity |    3     |    2    |    1     |    2     |
| Testosterone_Level     |    2     |    3    |    3     |    3     |
| Age                    |    4     |    4    |    4     |    4     |
| BMI                    |    5     |    5    |    5     |    5     |

KEY FINDING: Three of four methods rank AFC first. SVM SHAP is the
outlier - ranks Menstrual_Irregularity first (mean|SHAP|=0.2596 vs
AFC=0.1440). Explained by RBF kernel sensitivity to binary feature
structure. MLP SHAP agrees with LR and RF on AFC first
(mean|SHAP|~0.300 vs Menstrual_Irregularity~0.113).
Age and BMI rank last across all four methods without exception.
Reminder for me: Report the SVM divergence explicitly in the
Findings chapter interpretability section. Do not smooth it over.

Feature mapping decisions:

Soares Feature	Moheddine Proxy	Justification
Age	' Age (yrs)'	Direct match
BMI	'BMI'	Direct match
Menstrual_Irregularity	'Cycle(R/I)'	Cycle regularity/irregularity - direct clinical equivalent. R=regular (2), I=irregular (4) in Moheddine encoding
Testosterone_Level	'LH(mIU/mL)'	LH (Luteinising Hormone) is the closest available hormonal proxy. Elevated LH is a recognised hormonal marker in PCOS and is used in LH:FSH ratio diagnosis. No direct testosterone column exists in Moheddine
Antral_Follicle_Count	'Follicle No. (L)'	Left ovary follicle count. Right ovary (Follicle No. (R)) is equally valid; left is selected for consistency. Limitation: Soares uses a single combined AFC value

All three proxy substitutions are limitations that must be stated explicitly in the Findings chapter.


Day 9:
SHAP MOHEDDINE RESULTS (confirmed from figures and output):

SVM MOHEDDINE RANKING:
  1. Menstrual_Irregularity (Cycle(R/I))      : 0.004359
  2. Antral_Follicle_Count (Follicle No. L)   : 0.000727
  3. Age                                       : 0.000181
  4. BMI                                       : 0.000168
  5. Testosterone_Level (LH proxy)             : 0.000042

MLP MOHEDDINE RANKING:
  1. Menstrual_Irregularity (Cycle(R/I))      : 0.358597
  2. Antral_Follicle_Count (Follicle No. L)   : 0.054357
  3. Age                                       : 0.010019
  4. Testosterone_Level (LH proxy)             : 0.006748
  5. BMI                                       : 0.003113

CRITICAL FINDING 1 - SVM SHAP COLLAPSE ON MOHEDDINE:
SVM Moheddine SHAP axis spans only -0.003 to +0.010 vs Soares
-0.50 to +0.30. The SVM model is not effectively discriminating
Moheddine records. Wide-margin boundary (C=0.1) classifies most
Moheddine records into a single probability region. This is
consistent with SVM Dalvi FN=31 and confirms poor transfer to
out-of-distribution data.
Reminder for me: This is a strong finding for the Discussion chapter.
The SVM's AUC advantage on Dalvi coexists with a near-collapsed SHAP
signal on Moheddine. Both must be reported.

CRITICAL FINDING 2 - MLP RANK SHIFT BETWEEN DATASETS:
MLP Soares: AFC ranks 1st. MLP Moheddine: Menstrual_Irregularity
ranks 1st. The MLP's internal feature weighting shifts with the
input distribution. This is evidence that MLP attribution is not
stable across datasets and should not be trusted for clinical
interpretation without dataset-specific validation.
Reminder for me: Do not overclaim - this finding is specific to
these two datasets. State it as instability relative to the study
scope only.

STABLE FINDING ACROSS ALL METHODS: Top two features consistently
Menstrual_Irregularity and Antral_Follicle_Count across all four
models and both datasets. Order varies but both always rank above
the remaining three features. This is the most defensible
interpretability conclusion in the study.

Note for me:
On Final Day - do markdown check for correct structures!!!!