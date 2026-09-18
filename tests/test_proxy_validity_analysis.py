from projection_transport_steering.flow_step_support import condition_specifications
from projection_transport_steering.proxy_validity_analysis import analyze_proxy_validity


def packets():
    names = [specification["name"] for specification in condition_specifications()]
    teacher = {"conditions": {}}
    for index, name in enumerate(names):
        teacher["conditions"][name] = {
            "concept_heldout_means": [float(index + concept) for concept in range(12)],
            "concept_heldout_worst": [float(index + concept) for concept in range(12)],
            "concept_neutral_kl_medians": [float(index + 1) for _ in range(12)],
        }
    rows = []
    for concept_id in range(12):
        for index, name in enumerate(names):
            for prompt_index in range(8):
                value = float(index)
                rows.append(
                    {
                        "concept_id": concept_id,
                        "condition": name,
                        "prompt_index": prompt_index,
                        "scores": {
                            "concept": value,
                            "fluency": -value,
                            "harmonic_mean": value,
                            "instruction": value,
                        },
                    }
                )
    return teacher, rows


def test_analyze_proxy_validity_passes_aligned_instrument():
    teacher, rows = packets()

    result = analyze_proxy_validity(
        teacher,
        rows,
        list(range(12)),
        invariants_pass=True,
        bootstrap_draws=100,
    )

    assert result["decision"] == "retain_teacher_forced_screen"
    assert result["gate"]["pv2"]["positive_concepts"] == 12
    assert result["gate"]["pv3"]["median_correlation"] == 1.0
    assert result["gate"]["pv4"]["overall_agreement"] == 1.0
    assert result["gate"]["pv5"]["median_correlation"] == 1.0


def test_analyze_proxy_validity_fails_closed_on_invariant():
    teacher, rows = packets()

    result = analyze_proxy_validity(
        teacher,
        rows,
        list(range(12)),
        invariants_pass=False,
        bootstrap_draws=100,
    )

    assert result["decision"] == "retire_teacher_forced_screen"
    assert not result["gate"]["pass"]


def test_analyze_proxy_validity_fails_closed_on_missing_score():
    teacher, rows = packets()
    rows[0]["scores"] = None

    result = analyze_proxy_validity(
        teacher,
        rows,
        list(range(12)),
        invariants_pass=True,
        bootstrap_draws=100,
    )

    assert result["decision"] == "retire_teacher_forced_screen"
    assert not result["gate"]["pv1"]["pass"]
