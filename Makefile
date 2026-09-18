# Project Smiles — reasoning-mode overconfidence on Gemma-2-2b-it.
#
# Usage:
#   make sync              # create the env with uv
#   make smoke             # light self-tests, no model
#   make all               # extract -> compare -> steer, on the FULL MMLU split
#   make compare N=100     # subsample to 100 questions (any target takes N=)
#   make compare SEED=11   # change the eval seed
#
# By default the benchmark runs on the WHOLE MMLU test split (n=None). Pass N=<k>
# to subsample. Results (summaries + full rollouts) land under results/.

N    ?=
SEED ?= 7
NFLAG := $(if $(N),--n $(N),)
SEEDFLAG := --seed $(SEED)

PY := PYTHONPATH=. uv run python
B  := behaviour_specific.overconfidence
BASE := $(notdir $(CURDIR))

.DEFAULT_GOAL := help
.PHONY: help sync smoke extract compare steer all clean tar

help:
	@echo "targets: sync | smoke | extract | compare | steer | all | clean | tar"
	@echo "options: N=<k> (default: full MMLU)   SEED=<s> (default: 7)"

sync:                       ## create the environment from pyproject via uv
	uv sync

smoke:                      ## run every module's light self-test (no model, no GPU)
	@for m in general.paths general.reasoning general.inference general.steering general.metrics \
	          general.storage general.sae models_specific.gemma_2_2b_it.model \
	          models_specific.gemma_2_2b_it.gemma_scope models_specific.gemma_2_2b_it.saes \
	          $(B).labeling $(B).personas \
	          $(B).mmlu.data $(B).confidence_logit $(B).confidence_answer_distribution \
	          $(B).confidence_self_reported $(B).confidence_yesno $(B).confidence_yesno_sampled \
	          $(B).compare_methods $(B).features_caa $(B).features_probe \
	          $(B).features_caa_behavioral $(B).features_sae $(B).steer_overconfidence \
	          $(B).analyze_methods $(B).analyze_features $(B).analyze_steering $(B).analyze_sae; do \
	  PYTHONPATH=. uv run python -c "import importlib,sys; m=importlib.import_module('$$m'); \
	    t=getattr(m,'_selftest',None); t() if t else print('$$m import OK')" || exit 1; \
	done

ROLLOUTS ?=
ROLLOUTSFLAG := $(if $(ROLLOUTS),--rollouts $(ROLLOUTS),)
extract:                    ## extract CAA + probe + behavioral directions; ROLLOUTS=methods_seed11,methods_seed23 reuses saved eval traces
	$(PY) -m $(B).features_caa $(NFLAG)
	$(PY) -m $(B).features_probe $(NFLAG)
	$(PY) -m $(B).features_caa_behavioral $(NFLAG) $(ROLLOUTSFLAG)
	$(PY) -m $(B).analyze_features

compare:                    ## run all 5 confidence methods; save summary + rollouts
	$(PY) -m $(B).compare_methods $(NFLAG) $(SEEDFLAG)

steer:                      ## steer the reasoning (always-on sweep + phase-split); needs `make extract` first. 4-GPU DP: run_steer.sh
	$(PY) -m $(B).steer_overconfidence $(NFLAG) $(SEEDFLAG)
	$(PY) -m $(B).analyze_steering

SAE ?= gemmascope
LAYERS ?=
LAYERSFLAG := $(if $(LAYERS),--layers $(LAYERS),)
sae:                        ## SAE feature diffing / depth sweep; SAE=gemmascope|saebench-topk|... LAYERS=0,5,10,15,20,25
	$(PY) -m $(B).features_sae --sae $(SAE) $(LAYERSFLAG) $(NFLAG)

all: extract compare steer   ## the full pipeline (SAE arm is opt-in: `make sae`)

clean:                      ## remove caches and results (keeps directions + data_cache)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf results

tar:                        ## make ../$(BASE).tar.gz (excludes weights, caches, results, .git)
	cd .. && COPYFILE_DISABLE=1 tar \
	  --exclude='$(BASE)/.git' \
	  --exclude='$(BASE)/data_cache' \
	  --exclude='$(BASE)/results' \
	  --exclude='$(BASE)/models_specific/gemma_2_2b_it/weights' \
	  --exclude='$(BASE)/models_specific/gemma_2_2b_it/gemma_scope' \
	  --exclude='$(BASE)/models_specific/gemma_2_2b_it/saes' \
	  --exclude='$(BASE)/behaviour_specific/overconfidence/directions' \
	  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.tar.gz' \
	  -czvf '$(BASE).tar.gz' '$(BASE)'
	@echo "created ../$(BASE).tar.gz"
