"""supplementary_checks.py
Checks run outside the notebook (Annexure C). The notebook is not changed and is not re-run.
The four fitted pipelines are rebuilt from data/soares_pcos.csv and data/dalvi_pcos.csv using the
settings in Table 4 of the report. Step 0 confirms that the rebuilt models reproduce the notebook
results before any other check is read.
Run from the repository root:  python supplementary_checks.py
Needs: pandas, numpy, scipy, scikit-learn, imbalanced-learn, shap (versions in requirements.txt)."""
import warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import shap
warnings.filterwarnings("ignore")
T = "PCOS_Diagnosis"
soares = pd.read_csv("data/soares_pcos.csv"); dalvi = pd.read_csv("data/dalvi_pcos.csv")
X, y = soares.drop(columns=T), soares[T]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = MinMaxScaler().fit(Xtr)
Xtr_s, Xte_s, Xd = sc.transform(Xtr), sc.transform(Xte), sc.transform(dalvi.drop(columns=T))
yd = dalvi[T].values

def pipe(clf): return Pipeline([("smote", SMOTE(k_neighbors=5, random_state=42)), ("clf", clf)])
def models():
    return {
     "LR":  pipe(LogisticRegression(C=10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)),
     "RF":  pipe(RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1)),
     "SVM": pipe(SVC(kernel="rbf", C=0.1, gamma="scale", probability=True, random_state=42)),
     "MLP": pipe(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.0001, learning_rate_init=0.01, activation="relu",
                               solver="adam", max_iter=500, early_stopping=True, n_iter_no_change=10, random_state=42))}
M = models()
for m in M.values(): m.fit(Xtr_s, ytr)
P = {k: m.predict_proba(Xd)[:, 1] for k, m in M.items()}

print("0. REPRODUCTION OF THE NOTEBOOK (Dalvi ROC-AUC; TN FP FN TP from predict())")
for k, m in M.items():
    tn, fp, fn, tp = confusion_matrix(yd, m.predict(Xd)).ravel()
    print(f"   {k:4s} AUC {roc_auc_score(yd, P[k]):.4f}  {tn} {fp} {fn} {tp}  Soares test AUC {roc_auc_score(yte, m.predict_proba(Xte_s)[:,1]):.5f}")

# 1. DeLong test for two correlated ROC curves (Sun and Xu 2014 fast form) and Holm adjustment
def midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T_ = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T_[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T_; return out
def delong(yv, preds):
    pos = yv == 1; m = pos.sum(); n = (~pos).sum(); k = len(preds)
    tx = np.zeros((k, m)); ty = np.zeros((k, n)); tz = np.zeros((k, m + n))
    for r, p in enumerate(preds):
        px, py = p[pos], p[~pos]
        tx[r], ty[r], tz[r] = midrank(px), midrank(py), midrank(np.concatenate([px, py]))
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    return aucs, np.cov(v01) / m + np.cov(v10) / n
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for i, idx in enumerate(o):
        run = max(run, (len(ps) - i) * ps[idx]); adj[idx] = min(1, run)
    return adj
def fmtp(p): return "< 0.0001" if p < 0.0001 else f"= {p:.4f}"
print("\n1. PAIRED DELONG TESTS ON DALVI (difference = first model minus second)")
pairs = [("MLP", "LR"), ("MLP", "RF"), ("MLP", "SVM"), ("SVM", "LR"), ("SVM", "RF"), ("RF", "LR")]
rows = []
for a_, b_ in pairs:
    a, S = delong(yd, [P[a_], P[b_]]); d = a[0] - a[1]; se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    rows.append((a_, b_, d, d - 1.96 * se, d + 1.96 * se, 2 * (1 - stats.norm.cdf(abs(d / se)))))
adj = holm(np.array([r[5] for r in rows[:3]]))
for i, r in enumerate(rows):
    extra = f"  Holm-adjusted (MLP family of 3) {fmtp(adj[i])}" if i < 3 else ""
    print(f"   {r[0]} - {r[1]}: {r[2]:+.4f} (95% CI {r[3]:+.4f} to {r[4]:+.4f}) p {fmtp(r[5])}{extra}")
rng = np.random.default_rng(42); ip, ineg = np.where(yd == 1)[0], np.where(yd == 0)[0]; boot = {o: [] for o in ("LR", "RF", "SVM")}
for _ in range(2000):
    ii = np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))]); am = roc_auc_score(yd[ii], P["MLP"][ii])
    for o in boot: boot[o].append(am - roc_auc_score(yd[ii], P[o][ii]))
print("   bootstrap 95% interval for MLP minus model:", {o: tuple(float(x) for x in np.round(np.percentile(v, [2.5, 97.5]), 4)) for o, v in boot.items()})

# 2. Class predictions: predict() against probability >= 0.5
print("\n2. DECISION RULE: predict() compared with predict_proba >= 0.5 (TN FP FN TP)")
for k, m in M.items():
    for nm, Xe, ye in (("Soares test", Xte_s, yte.values), ("Dalvi", Xd, yd)):
        p = m.predict_proba(Xe)[:, 1]; a1 = m.predict(Xe); a2 = (p >= 0.5).astype(int)
        print(f"   {k:4s} {nm:11s} predict {confusion_matrix(ye, a1).ravel().tolist()}  p>=0.5 {confusion_matrix(ye, a2).ravel().tolist()}  disagree {int((a1 != a2).sum())}")

# 3. Grid search with the scaler outside the folds (as in the notebook) and inside each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grids = {
 "LR":  (LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=42), {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]}),
 "RF":  (RandomForestClassifier(random_state=42, n_jobs=-1), {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 2, 4]}),
 "SVM": (SVC(kernel="rbf", probability=True, random_state=42), {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto", 0.01, 0.1]}),
 "MLP": (MLPClassifier(activation="relu", solver="adam", max_iter=500, early_stopping=True, n_iter_no_change=10, random_state=42),
         {"clf__hidden_layer_sizes": [(64,), (64, 32), (128, 64)], "clf__alpha": [0.0001, 0.001], "clf__learning_rate_init": [0.001, 0.01]})}
print("\n3. TUNING: scaler fitted before the folds (notebook design) against scaler inside each fold")
for k, (clf, grid) in grids.items():
    for design in ("before", "inside"):
        steps = [("smote", SMOTE(k_neighbors=5, random_state=42)), ("clf", clf)]
        if design == "inside": steps = [("scaler", MinMaxScaler())] + steps
        gs = GridSearchCV(Pipeline(steps), grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(Xtr_s if design == "before" else Xtr.values, ytr)
        ms = np.array(gs.cv_results_["mean_test_score"]); b = gs.best_index_
        Xe = Xd if design == "before" else dalvi.drop(columns=T).values
        tn, fp, fn, tp = confusion_matrix(yd, gs.predict(Xe)).ravel()
        print(f"   {k:4s} {design:6s} best {gs.best_params_} mean CV AUC {ms[b]:.7f} sd {gs.cv_results_['std_test_score'][b]:.7f} "
              f"combos tied at the maximum {int((ms == ms.max()).sum())} of {len(ms)} lowest {ms.min():.7f} | Dalvi AUC {roc_auc_score(yd, gs.predict_proba(Xe)[:,1]):.4f} FP {fp} FN {fn}")
mlp = M["MLP"].named_steps["clf"]
print(f"   MLP final fit: stopped after {mlp.n_iter_} iterations, best internal validation accuracy {max(mlp.validation_scores_):.4f}, "
      f"{sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)} weights and biases")

# 4. Tuned settings refitted without SMOTE (not re-tuned)
print("\n4. WITHOUT SMOTE (same tuned settings, not re-tuned): Dalvi AUC, FP, FN")
ns = {"LR": LogisticRegression(C=10, max_iter=1000, random_state=42), "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
      "SVM": SVC(C=0.1, gamma="scale", probability=True, random_state=42),
      "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.0001, learning_rate_init=0.01, max_iter=500, early_stopping=True, n_iter_no_change=10, random_state=42)}
for k, m in ns.items():
    m.fit(Xtr_s, ytr); tn, fp, fn, tp = confusion_matrix(yd, m.predict(Xd)).ravel()
    print(f"   {k:4s} AUC {roc_auc_score(yd, m.predict_proba(Xd)[:,1]):.4f} FP {fp} FN {fn}")

# 5. SHAP: expected value, additivity, Dalvi, and an original-training background
print("\n5. SHAP KERNELEXPLAINER (SVM and MLP; positive class; probability units)")
feat = ["Age", "BMI", "Menstrual_Irregularity", "Testosterone_Level", "Antral_Follicle_Count"]
Xres, _ = SMOTE(k_neighbors=5, random_state=42).fit_resample(Xtr_s, ytr)
bgs = {"balanced (SMOTE) background": shap.kmeans(Xres, 100), "original-training background": shap.kmeans(Xtr_s, 100)}
for k in ("SVM", "MLP"):
    for bn, bg in bgs.items():
        for nm, Xe in (("Soares test", Xte_s), ("Dalvi", Xd)):
            if nm == "Dalvi" and bn.startswith("original"): continue
            ex = shap.KernelExplainer(M[k].predict_proba, bg); sv = np.asarray(ex.shap_values(Xe, nsamples="auto", silent=True))[:, :, 1]
            ev = float(np.atleast_1d(ex.expected_value)[1]); err = np.abs(sv.sum(1) + ev - M[k].predict_proba(Xe)[:, 1]).max()
            ma = np.abs(sv).mean(0); print(f"   {k:3s} {nm:11s} {bn:30s} expected value {ev:.4f} max additivity error {err:.1e} mean|SHAP| "
                  + ", ".join(f"{feat[i]} {ma[i]:.4f}" for i in np.argsort(-ma)))
