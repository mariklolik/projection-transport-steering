# Teacher-forced proxy validity hypotheses, version 1

Status: frozen before proxy-validity generation or judging. This is an instrument-validation study, not a steering candidate.

## Estimand

For each of the 12 exposed concepts and 14 frozen Flow-step support conditions, compare the pre-existing mean and worst-view teacher-forced positive-minus-negative continuation change with open-generation AxBench-format scores on the same eight evaluation prompts. The independent unit is a concept. Within-concept ranks are primary because score levels are not comparable across concepts.

The local Qwen3-14B proxy applies the exact upstream Concept, Instruction, and Fluency prompts and parsing function. Its scores are local proxies, not GPT-4o-mini labels. Raw judge completions, parse failures, and all conditions remain visible.

## Hypotheses

- PV.1: all 1,344 planned generations and 4,032 judge completions are present, finite where defined, uniquely keyed, source-hash valid, and exactly serialized; no condition has differential generation or parse attrition.
- PV.2: the median within-concept Spearman correlation between teacher-forced held-out mean change and generated Concept score is at least `0.50`, at least 9/12 concept correlations are positive, and the frozen concept-bootstrap 95% lower endpoint for their mean is above `0.20`.
- PV.3: the same three thresholds hold for teacher-forced held-out worst-view change against generated harmonic mean.
- PV.4: pairwise ordering against full FLAS agrees between teacher-forced held-out worst view and generated harmonic mean for at least 70% of all non-full concept-condition pairs and at least 60% within every concept.
- PV.5: teacher-forced neutral KL has positive within-concept association with generated fluency loss: median Spearman at least `0.30`, at least 8/12 positive, and mean concept-bootstrap lower endpoint above zero.

All five hypotheses must pass to retain teacher-forced likelihood as a primary future method screen. Failure retires the proxy-first funnel. A pass does not select a condition, validate the judge, establish behavior improvement, or support a paper claim.
