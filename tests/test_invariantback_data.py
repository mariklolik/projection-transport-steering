from projection_transport_steering.invariantback_data import (
    allocate_groups,
    mnli_groups,
    normbank_groups,
    sc101_groups,
)


def test_normbank_groups_require_complete_labels_and_clean_text():
    rows = [
        {
            "setting": " cafe ",
            "behavior": " talk   loudly ",
            "constraints": f"role is {label}",
            "norm": label,
        }
        for label in ("taboo", "normal", "expected")
    ]
    rows.append({**rows[0], "norm": "taboo"})
    rows.append({**rows[0], "setting": "park"})

    groups = normbank_groups(rows)

    assert len(groups) == 1
    assert groups[0]["group_key"] == "cafe\ntalk loudly"
    assert set(groups[0]["endpoints"]) == {"taboo", "normal", "expected"}
    assert "  " not in groups[0]["endpoints"]["taboo"]["text"]


def test_mnli_groups_are_premise_complete_and_deterministic():
    rows = [
        {"premise": " One  premise ", "hypothesis": f"h{label}", "label": label}
        for label in (0, 1, 2)
    ]
    rows.extend(
        {"premise": "incomplete", "hypothesis": f"x{label}", "label": label}
        for label in (0, 1)
    )

    forward = mnli_groups(rows)
    reverse = mnli_groups(reversed(rows))

    assert forward == reverse
    assert len(forward) == 1
    assert set(forward[0]["endpoints"]) == {
        "entailment",
        "neutral",
        "contradiction",
    }


def test_sc101_groups_match_unique_actions_without_replacement():
    rows = [
        {"action": f"{label}-{index}", "action-moral-judgment": str(label)}
        for label in (-2, 0, 2)
        for index in range(4)
    ]
    rows.append({"action": "-2-0", "action-moral-judgment": "-2"})

    groups = sc101_groups(rows, seed=7)

    assert len(groups) == 4
    assert all(set(group["endpoints"]) == {"bad", "ok", "good"} for group in groups)
    source_ids = [
        endpoint["source_id"]
        for group in groups
        for endpoint in group["endpoints"].values()
    ]
    assert len(source_ids) == len(set(source_ids))


def test_allocation_is_deterministic_disjoint_and_pilot_can_be_sealed():
    groups = [
        {
            "dataset": "synthetic",
            "group_id": f"g{index}",
            "group_key": f"key{index}",
            "endpoints": {},
        }
        for index in range(20)
    ]

    allocation = allocate_groups(
        groups,
        {"construction": 5, "sentinel": 3, "development": 4, "pilot": 8},
        seed=11,
    )

    assert allocation == allocate_groups(
        reversed(groups),
        {"construction": 5, "sentinel": 3, "development": 4, "pilot": 8},
        seed=11,
    )
    ids = [group["group_id"] for split in allocation.values() for group in split]
    assert len(ids) == len(set(ids)) == 20
    public = {key: value for key, value in allocation.items() if key != "pilot"}
    assert "pilot" not in public

