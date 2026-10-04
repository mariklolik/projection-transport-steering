import time, json, torch, numpy as np
from models_specific.active import load_model, chat_prompt
from behaviour_specific.toxicity.run import records, PROMPT, generate_masked, decision_scores, threshold, load, load_feats, action, DIRS
m, t = load_model()
rows = records("confirm")[0::4]
prompts = [chat_prompt(t, PROMPT.format(x=r["text"])) for r in rows]
base = [load("confirm", "base")[r["id"]] for r in rows]
feats = load_feats("confirm")
d = torch.load(DIRS / "direction.pt")
fn = action("add-0.5", d["v"].cuda(), d["norm"])
mask = decision_scores("probe", base, feats) > threshold("probe", 0.5)
generate_masked(m, t, prompts[:64], np.zeros(64, bool), lambda h: h, 128)
res = {}
for label, msk, f, sub in (("unsteered", np.zeros(len(rows), bool), lambda h: h, None), ("pts_one_pass", mask, fn, None),
                           ("regenerate_flagged", np.ones(int(mask.sum()), bool), fn, [p for p, k in zip(prompts, mask) if k])):
    ps = sub if sub is not None else prompts
    torch.cuda.synchronize(); t0 = time.time()
    generate_masked(m, t, ps, msk, f, 128)
    torch.cuda.synchronize(); res[label] = time.time() - t0
res["pts_post_hoc"] = res["unsteered"] + res["regenerate_flagged"]
res["n"], res["fire"] = len(rows), float(mask.mean())
print(json.dumps(res))
from general.inference import _length_buckets
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    for idx in _length_buckets(t, prompts, 128):
        t.padding_side = "left"
        b = t([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        m(**b, output_hidden_states=True)
torch.cuda.synchronize(); res["decision_prefill"] = time.time() - t0
res["pts_prompt_decision_total"] = res["pts_one_pass"] + res["decision_prefill"]
print(json.dumps(res))
