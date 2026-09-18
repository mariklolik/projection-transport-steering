def generate_condition(
    generator: object,
    prompts: list[str],
    concept: str,
    specification: dict[str, object],
    max_new_tokens: int,
) -> list[dict[str, object]]:
    previous_intervention = getattr(generator, "_iv", None)
    try:
        generator._iv = specification["intervention"]
        rows = generator.generate_batch(
            prompts,
            concept,
            [2.0],
            n_steps=int(specification["n_steps"]),
            max_tokens=max_new_tokens,
            temperature=0.0,
            max_batch=len(prompts),
        )
        if len(rows) != len(prompts):
            raise RuntimeError("generation count mismatch")
        for index, (prompt, row) in enumerate(zip(prompts, rows, strict=True)):
            if (
                row["prompt_idx"] != index
                or row["prompt"] != prompt
                or row["flowtime"] != 2.0
                or not isinstance(row["generation"], str)
                or not row["generation"].strip()
            ):
                raise RuntimeError("invalid generation row")
        return rows
    finally:
        generator._iv = previous_intervention
        generator._active = False
        generator._concept_hidden = None
        generator._concept_mask = None
        generator._flowtimes = None
        generator._padding_mask = None
        generator._position_ids = None
        generator._sa_caches = None
        generator._remove_hook()
