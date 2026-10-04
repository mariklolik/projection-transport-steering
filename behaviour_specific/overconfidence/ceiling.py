import json, numpy as np, torch
from behaviour_specific.overconfidence.fit_detector import load_blob
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.label_pool import LAYERS
from behaviour_specific.overconfidence.analyze_steering import collect
from behaviour_specific.overconfidence.gate_law import auroc
from general.paths import RESULTS_DIR
import sys
d = sys.argv[1]
base = {r["id"]: r for r in collect(RESULTS_DIR / d / "rollouts")["baseline_m5"]}
F = load_blob([d], "feats")
v = torch.load(DIRECTIONS_DIR / "pts_L14.pt")["dirs"]["m4_conf"].float()
det = torch.load(DIRECTIONS_DIR / "detector_m5.pt")
ids = [i for i, r in base.items() if r["state"] in ("overconfident_wrong", "confident_right") and i in F]
X = torch.stack([F[i] for i in ids]).float()
y = np.array([base[i]["state"] == "overconfident_wrong" for i in ids], float)
def J(s):
    best = 0
    for t in np.unique(s):
        best = max(best, (s[y == 1] > t).mean() - (s[y == 0] > t).mean(), (s[y == 1] < t).mean() - (s[y == 0] < t).mean())
    return best
out = {}
for name, s in (("confidence", (X[:, LAYERS.index(14)] @ v).numpy()), ("probe", (X[:, LAYERS.index(det["layer"])] @ det["w"].float() + det["b"]).numpy())):
    a = auroc(s, y)
    out[name] = {"auroc": round(max(a, 1 - a), 3), "ceiling": round(J(s), 3)}
    print(d, name, out[name])
json.dump(out, open(RESULTS_DIR / d / "ceiling.json", "w"), indent=1)
