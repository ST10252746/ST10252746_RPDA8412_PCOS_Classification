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