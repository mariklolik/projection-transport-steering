from projection_transport_steering.flow_step_support import condition_specifications


def test_condition_specifications_match_frozen_design():
    conditions = condition_specifications()

    assert [condition["name"] for condition in conditions] == [
        "raw_0",
        "raw_1",
        "raw_2",
        "raw_01",
        "raw_02",
        "raw_12",
        "full",
        "equal_0",
        "equal_1",
        "equal_2",
        "equal_01",
        "equal_02",
        "equal_12",
        "official_n2",
    ]
    assert conditions[0]["intervention"] == {"steps": [0]}
    assert conditions[6]["intervention"] is None
    assert conditions[12]["intervention"] == {"scale": 1.5, "steps": [1, 2]}
    assert conditions[13]["n_steps"] == 2
    assert conditions[13]["intervention"] is None
