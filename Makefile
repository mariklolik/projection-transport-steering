PY := PYTHONPATH=. uv run --frozen python
B := behaviour_specific.overconfidence
REF := ef8de47ab7cc3a97f2b5778cbfbb5b999fe0c3c4
.DEFAULT_GOAL := help
.PHONY: help sync smoke artifacts records extract baseline steer

help:
	@echo "targets: sync | smoke | artifacts | records | extract | baseline | steer"

sync:
	uv sync --frozen

smoke:
	@for m in behaviour_specific.overconfidence.analyze_pooled behaviour_specific.overconfidence.rollouts behaviour_specific.overconfidence.benchmarks behaviour_specific.overconfidence.confidence_logit \
	          behaviour_specific.overconfidence.confidence_yesno behaviour_specific.overconfidence.diag_layers behaviour_specific.overconfidence.discriminant behaviour_specific.overconfidence.extract_projection_stats \
	          behaviour_specific.overconfidence.fit_detector behaviour_specific.overconfidence.gate_law behaviour_specific.overconfidence.label_pool \
	          behaviour_specific.overconfidence.labeling behaviour_specific.overconfidence.law_confirm behaviour_specific.overconfidence.mmlu.data behaviour_specific.overconfidence.select_configs \
	          behaviour_specific.overconfidence.readouts behaviour_specific.overconfidence.actions behaviour_specific.overconfidence.transport \
	          behaviour_specific.overconfidence.gates behaviour_specific.overconfidence.steer behaviour_specific.overconfidence.selection general.behavioral_subspace \
	          general.inference general.metrics general.online_gate general.paths \
	          general.reasoning general.steering general.storage models_specific.active \
	          models_specific.gemma_2_2b_it.model models_specific.gemma_2_9b_it.model models_specific.mistral_7b_it.model models_specific.qwen2_5_7b_it.model; do \
	  $(PY) -c "import importlib; m=importlib.import_module('$$m'); t=getattr(m,'_selftest',None); t() if t else None" || exit 1; \
	done

artifacts:
	@bundle=$$(mktemp); trap 'rm -f "$$bundle"' EXIT; \
	  curl -fsSL "https://codeload.github.com/mariklolik/projection-transport-steering/tar.gz/$(REF)" -o "$$bundle" && \
	  tar -xzf "$$bundle" --strip-components=1 "projection-transport-steering-$(REF)/results/release" "projection-transport-steering-$(REF)/behaviour_specific/overconfidence/directions"

records:
	$(PY) results/recompute_from_records.py v4_confirm det_q60_alpha-0.375

extract:
	$(PY) -m $(B).extract_projection_stats --n 400 --layer 14
	$(PY) -m $(B).diag_layers --layers 14 --dump-moments

baseline:
	$(PY) -m $(B).label_pool --split confirm

steer:
	$(PY) -m $(B).steer --split confirm --outdir v4_confirm --readouts m5 --detector detector_m5 --methods plain_alpha-0.375 --detector-configs q60_alpha-0.375
