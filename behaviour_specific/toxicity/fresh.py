import json, os, numpy as np
from behaviour_specific.overconfidence.analyze_pooled import ci
from behaviour_specific.toxicity.analyze import sel, arrays, flags, override_sel, load_feats
from behaviour_specific.toxicity.run import OUT, load
M = os.environ.get("MODEL_IMPL", "gemma_2_2b_it")
S = json.load(open("rebuttal/protocols/confirmatory-v26-fresh.selection.json"))[M]
split = "confirm2"
b = load(split, "base"); ids = sorted(b); base = [b[i] for i in ids]
n = len(ids); w = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=10000).astype(float); one = np.ones((1, n))
get = lambda a: (lambda r: [r[i] for i in ids])(load(split, a.replace("|", "_")))
old = S["frozen_v23"]
arms = {"pts": S["pts"], "dsas": S["dsas"], "lact": S["lact"], "global": old["ungated"], "prompt": "prompt", "cast": old["cast"], "pts_v23": old["probe"]}
if "ot" in S: arms.update(ot=S["ot"], otmatch=S["otmatch"])
rows = {k: get(v) for k, v in arms.items()}
bs = {k: sel(w, base, r) for k, r in rows.items()}
pt = {k: float(sel(one, base, r)[0]) for k, r in rows.items()}
out = {"n_toxic": int(sum(x["state"] == "toxic" for x in base)), "n_clean": int(sum(x["state"] == "clean" for x in base)), "arms": {}}
for k, r in rows.items():
    t0, c0, t1, c1 = arrays(base, r)
    out["arms"][k] = {"arm": arms[k], "sel": {"point": round(pt[k], 4), "ci": ci(bs[k])}, "removal": round(float((t0*(1-t1)).sum()/t0.sum()), 4),
                      "retention": round(float((c0*c1).sum()/c0.sum()), 4), "toxic_after": round(float(t1.mean()), 4)}
    if k != "pts":
        d = bs["pts"] - bs[k]
        out["arms"][k]["pts_minus"] = {"point": round(pt["pts"] - pt[k], 4), "ci": ci(d), "p_one_sided": round(float((d <= 0).mean()), 4)}
kind, q, _ = S["pts"].split("|"); f = flags(split, kind, int(q[1:]), base, feats := load_feats(split))
ob = override_sel(w, base, f); op = float(override_sel(one, base, f)[0]); d = bs["pts"] - ob
out["refusal"] = {"point": round(op, 4), "pts_minus": {"point": round(pt["pts"] - op, 4), "ci": ci(d), "p_one_sided": round(float((d <= 0).mean()), 4)}}
rs = [get(f"rand{s}|{S['pts']}") for s in range(100)]
rp = np.array([sel(one, base, r)[0] for r in rs]); rb = np.mean([sel(w, base, r) for r in rs], axis=0); d = bs["pts"] - rb
out["random"] = {"mean": round(float(rp.mean()), 4), "p95": round(float(np.quantile(rp, .95)), 4), "max": round(float(rp.max()), 4), "ge": int((rp >= pt["pts"]).sum()),
                 "pts_minus": {"point": round(pt["pts"] - float(rp.mean()), 4), "ci": ci(d), "p_one_sided": round(float((d <= 0).mean()), 4)}}
nl = get("null|" + S["pts"]); out["null"] = {"sel": float(sel(one, base, nl)[0]), "identical": int(sum(x["text"] == y["text"] for x, y in zip(nl, base)))}
if "ot" in S:
    d = bs["ot"] - bs["otmatch"]; out["ot_minus_matched"] = {"point": round(pt["ot"] - pt["otmatch"], 4), "ci": ci(d), "p_one_sided": round(float((d <= 0).mean()), 4)}
json.dump(out, open(OUT / "confirm2.json", "w"), indent=1)
print(json.dumps({k: (v["sel"]["point"], v.get("pts_minus", {}).get("point"), v.get("pts_minus", {}).get("p_one_sided")) for k, v in out["arms"].items()}), out["refusal"], out["random"], out.get("ot_minus_matched"), out["null"])
