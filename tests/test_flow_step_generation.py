from projection_transport_steering.flow_step_generation import generate_condition


class FakeGenerator:
    def __init__(self):
        self._active = False
        self._concept_hidden = None
        self._concept_mask = None
        self._flowtimes = None
        self._hook_handle = object()
        self._iv = None
        self._padding_mask = None
        self._position_ids = None
        self._sa_caches = None

    def generate_batch(self, prompts, concept, flowtimes, **kwargs):
        assert self._iv == {"scale": 1.5, "steps": [1, 2]}
        return [
            {
                "flowtime": flowtimes[0],
                "generation": f"{concept}: {prompt}",
                "prompt": prompt,
                "prompt_idx": index,
            }
            for index, prompt in enumerate(prompts)
        ]

    def _remove_hook(self):
        self._hook_handle = None


def test_generate_condition_restores_all_generator_state():
    generator = FakeGenerator()

    rows = generate_condition(
        generator,
        ["first", "second"],
        "concept",
        {
            "intervention": {"scale": 1.5, "steps": [1, 2]},
            "n_steps": 3,
            "name": "equal_12",
        },
        max_new_tokens=128,
    )

    assert [row["generation"] for row in rows] == [
        "concept: first",
        "concept: second",
    ]
    assert generator._iv is None
    assert generator._hook_handle is None
    assert generator._concept_hidden is None
    assert generator._sa_caches is None
