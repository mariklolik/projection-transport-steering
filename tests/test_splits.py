from projection_transport_steering.splits import allocate_groups, stable_group_id


def test_stable_group_id_ignores_mapping_order():
    left = stable_group_id("sha", "config", "split", {"b": 2, "a": 1})
    right = stable_group_id("sha", "config", "split", {"a": 1, "b": 2})

    assert left == right


def test_allocate_groups_is_deterministic_and_disjoint():
    groups = [f"group-{index}" for index in range(20)]
    quotas = {"fit": 5, "observer": 4, "development": 3, "test": 2}

    first = allocate_groups(groups, quotas, "salt")
    second = allocate_groups(reversed(groups), quotas, "salt")

    assert first == second
    assert {stage: list(first.values()).count(stage) for stage in quotas} == quotas
    assert len(first) == sum(quotas.values())


def test_allocate_groups_keeps_duplicate_group_in_one_stage():
    allocation = allocate_groups(["a", "a", "b", "c"], {"fit": 1, "test": 2}, "salt")

    assert len(allocation) == 3


def test_allocate_groups_rejects_insufficient_groups():
    try:
        allocate_groups(["a"], {"fit": 2}, "salt")
    except ValueError as error:
        assert "insufficient" in str(error)
    else:
        raise AssertionError("insufficient groups were accepted")
