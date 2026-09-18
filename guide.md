# Guide — running this on the H100 box

In a nutshell: `uv sync`, set an HF token (Gemma is gated), then `make all`.
By default everything runs on the **whole MMLU test split**; add `N=<k>` to
subsample.

## 1. Setup

```bash
uv sync                       # create the env from pyproject.toml
export HF_TOKEN=hf_...         # Gemma-2 is gated; the box downloads the weights
```

The model weights are **not** in this tarball. `models_specific/gemma_2_2b_it/model.py`
loads local weights if a `weights/` folder is there, otherwise pulls
`google/gemma-2-2b-it` from the hub (that's what the H100 does). MMLU downloads
once on the first run and is cached in `data_cache/`.

Quick check with no model/GPU:

```bash
make smoke                    # every module's light self-test
```

## 2. Run

```bash
make all                      # extract directions -> compare methods -> steer, FULL split
```

or piece by piece:

| command | what it does |
|---|---|
| `make extract` | extract the overconfidence directions (CAA + probe + behavioral) + cross-analysis; `ROLLOUTS=methods_seed11,methods_seed23` reuses saved eval traces for the behavioral arm (forward-only) |
| `make compare` | run all 5 confidence-measurement methods, save summary + rollouts |
| `make steer`   | steer the reasoning trace (always-on sweep + ablate + phase-split, M2+M4 readouts) + report; needs `make extract` first |
| `make sae`     | SAE feature diffing (Gemma Scope); needs the SAE npz transferred first — see below |

Multi-GPU drivers (detached, logs under `logs/`):

```bash
nohup bash run_extract.sh > logs/extract_driver.log 2>&1 &   # persona arm on GPU0 + behavioral arm on GPU1, then probe + analysis
nohup env DIRS=caa,probe LAYER=7 bash run_steer.sh > logs/steer_driver.log 2>&1 &  # steering DP over 4 GPUs (records sharded)
```

Options (any target):

```bash
make compare N=100            # subsample to 100 questions instead of the full split
make compare SEED=11          # different eval seed
make all N=500                # quick-ish full pipeline
```

**Heads-up on the full run.** The whole MMLU test split is ~14k questions and
every method reasons (a `<think>` trace per call). M5 alone samples 20 traces
per question (~280k generations). On an H100 this is an overnight job — start
with `N=500` to sanity-check timing and outputs, then launch the full run.

## 3. What comes out (send `results/` back)

Everything lands under `results/`:

```
results/
  methods_n<N>_seed<S>/
    summary.json              # per-method: accuracy, ECE, overconfidence_gap,
                              # mean_confidence, parse rate, 4-state histogram, seconds
    rollouts/
      m2_logit.jsonl          # one rich record per question:
      m1_answer_dist.jsonl    #   system_prompt, user_prompt, every full <think>
      m3_self_report.jsonl    #   generation, the parsed final answer, gold,
      m4_yesno.jsonl          #   confidence, 4-state label, method extras
      m5_yesno_sampled.jsonl
  features/
    persona_acts.pt           # persona-trace activations [n, layers, d] (pos+neg)
    behavioral_acts.pt        # natural-trace activations + M2/M4 labels
    analysis.json             # probe AUROC by layer, persona->on-policy transfer,
                              # cosine between all candidate directions
  steering/
    rollouts/<cond>__shard<k>.jsonl   # per-condition steered rollouts (full traces)
    analysis.json             # per-condition metrics + paired bootstrap CIs on the
                              # deltas + the M2-vs-M4 cross-method table
  analysis/summary.json       # method-comparison bootstrap CIs + per-question agreement
```

`summary.json` is the machine-readable numbers; the `rollouts/*.jsonl` are the
raw answers for eyeballing by hand. **Zip `results/` and send it back** — that's
what I read to write `results.md`.

```bash
COPYFILE_DISABLE=1 tar -czvf results.tar.gz results
```

## 3b. SAE arm — needs a manual weight transfer

Like the model, pretrained SAEs can't be downloaded on the box (no internet), so
they're fetched **locally** and transferred. The registry lives in
`models_specific/gemma_2_2b_it/saes.py`:

The SAE arm is **layer-parametric** (`--sae KIND --layers L1,L2,...`):

| `--sae` kind | architecture | layers | via |
|---|---|---|---|
| `gemmascope` | JumpReLU (Gemma Scope) | any 0–25 | local npz |
| `saebench-topk` | **TopK** (SAEBench) | 5, 12, 19 | sae_lens |
| `saebench-vanilla` | vanilla ReLU (SAEBench) | 5, 12, 19 | sae_lens |

Fetch what you want and drop it in (both dirs are excluded from the code
tarball, moved like the model weights):

```
models_specific/gemma_2_2b_it/gemma_scope/layer_<L>_width_16k.npz   # gemmascope (web links in README/above)
models_specific/gemma_2_2b_it/saes/<kind>_l<L>/                     # SAEBench (run saes.download locally)
```

`saes.download` (LOCAL, needs internet) fetches a SAEBench SAE into the right
folder, e.g. `python -c "from models_specific.gemma_2_2b_it import saes; saes.download('saebench-topk', 12)"`.

Then run one layer or a **depth sweep**:

```bash
make sae SAE=gemmascope LAYERS=0,5,10,15,20,25      # sweep: feature card per layer + depth profile
make sae SAE=saebench-topk LAYERS=5,12,19          # TopK at its 3 layers
make sae SAE=gemmascope                            # single layer (default 7)
```

Traces are generated once and read at every layer in one forward. Each layer
gets a **feature card** (top over-confident features + dashboard links) and
steerable decoder dirs in `behaviour_specific/overconfidence/directions/sae_<kind>_l<L>.pt`;
the run also writes a **depth profile** (feature-diff strength vs layer, i.e.
*where* overconfidence is encoded) to `results/sae_<kind>/depth_profile.json`.
`sae_lens` is a project dependency (installed by `uv sync` from your mirror), so
SAEBench SAEs load offline via `SAE.load_from_disk`. Caveat: Gemma Scope and
SAEBench SAEs are trained on the *base* `gemma-2-2b`, applied here to `-it`.

## 4. Repackaging (if you edit the code)

```bash
make tar                      # -> ../<projectdir>.tar.gz, excluding weights/caches/results
```
