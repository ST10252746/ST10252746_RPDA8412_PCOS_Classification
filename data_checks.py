import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

FILES = {"soares": "data/soares_pcos.csv", "dalvi": "data/dalvi_pcos.csv", "moheddine": "data/moheddine_pcos.csv"}
for name, f in FILES.items():
    print(name, hashlib.sha256(open(f, "rb").read()).hexdigest())

soares = pd.read_csv(FILES["soares"])
dalvi = pd.read_csv(FILES["dalvi"])
moh = pd.read_csv(FILES["moheddine"])
moh.columns = [c.strip() for c in moh.columns]
X = ["Age", "BMI", "Menstrual_Irregularity",
     "Testosterone_Level(ng/dL)", "Antral_Follicle_Count"]
y = "PCOS_Diagnosis"

# 1. Does the label follow an antral follicle count rule?
for name, d in (("soares", soares), ("dalvi", dalvi)):
    rule = (d["Antral_Follicle_Count"] >= 12).astype(int)
    print(name, "label equals AFC>=12 in", int((rule == d[y]).sum()), "of", len(d))

# 2. Exact duplicates and nearest-neighbour distance, Dalvi to Soares
s = soares[X].astype(float); d_ = dalvi[X].astype(float)
print("exact rows shared:", len(s.merge(d_, how="inner")))
sc = MinMaxScaler().fit(s)
nn = NearestNeighbors(n_neighbors=2, metric="manhattan").fit(sc.transform(s))
d_to_s = nn.kneighbors(sc.transform(d_), 1)[0][:, 0]
s_to_s = nn.kneighbors(sc.transform(s), 2)[0][:, 1]
print("median nearest distance, Dalvi to Soares:", round(float(np.median(d_to_s)), 3))
print("median nearest distance, Soares to Soares:", round(float(np.median(s_to_s)), 3))

# 3. Split and fold replication (seed 42, as in the notebook)
Xtr, Xte, ytr, yte = train_test_split(soares[X], soares[y], test_size=0.2,
                                      stratify=soares[y], random_state=42)
print("train / test:", len(Xtr), len(Xte), "test positives:", int(yte.sum()))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for i, (a, b) in enumerate(skf.split(Xtr, ytr), 1):
    print("fold", i, "train", len(a), "validation", len(b))

# 4. Scaler exposure: training range versus held-out test range
sc_tr = MinMaxScaler().fit(Xtr)
Xte_s = sc_tr.transform(Xte)
print("test values outside the training range:", int(((Xte_s < 0) | (Xte_s > 1)).sum()))

# 5. Moheddine proxies against the Soares training range
print("Cycle(R/I) values:", sorted(moh["Cycle(R/I)"].unique()))
lo, hi = soares["Testosterone_Level(ng/dL)"].min(), soares["Testosterone_Level(ng/dL)"].max()
print("Soares testosterone range:", lo, hi)
print("Moheddine LH (mIU/mL) range:", moh["LH(mIU/mL)"].min(), moh["LH(mIU/mL)"].max())
print("LH inside Soares testosterone range:",
      int(moh["LH(mIU/mL)"].between(lo, hi).sum()), "of", len(moh))