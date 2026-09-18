# InvariantBack mapping-comprehension sentinel protocol, version 1

Freeze date: 2026-09-06. Frozen after QP sentinel completion and before mapping-comprehension model output.

The unsteered pinned Gemma-2-9B-IT model is evaluated on all 90 dataset, semantic-mapping, and identifier-vocabulary cells: three datasets, all six mappings, and A/B/C, X/Y/Z, 1/2/3, I/II/III, and one/two/three. Each cell contains nine scenario-free queries formed by three labels and the three frozen question phrasings. All three candidate identifier sequences are teacher-forced from the same prompt prefix and scored by mean token log likelihood.

A cell passes only when at least eight of nine queries select the mapped identifier and every query gives the mapped identifier a strictly positive margin over both alternatives. All 90 cells must pass. This is deliberately stricter than an aggregate accuracy threshold because held-out semantic effects are uninterpretable when their temporary key cannot be decoded.

The sentinel records every query score, prediction, margin, cell accuracy, model and software versions, source packet hash, wall time, forward count, and peak memory. It uses no activation intervention and cannot select any method or hyperparameter. Failure closes Pilot A under the frozen mapping-comprehension gate; passing allows development implementation but does not open the sealed pilot.

