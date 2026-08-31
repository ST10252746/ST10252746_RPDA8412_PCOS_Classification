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
- SVM produces fewest Dalvi FPs (164) but highest FNs (31) - different
  error profile from LR and RF; Dalvi AUC=0.9024 highest of three models to date
- Note: No feature importance produced - SHAP KernelExplainer reserved for Day 9