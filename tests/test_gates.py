from hcbf.gates import apply_operational_gates


def test_empty_eligibility_is_allowed():
    result = apply_operational_gates(
        numerical_parity=True,
        pair_p95_latency_ms=12.0,
        worst_class0_to_class1_error=0.1,
        worst_class1_to_class0_error=1.0,
        collapse_count=10,
    )
    assert result.latency_pass
    assert not result.safety_pass
    assert not result.eligible


def test_all_gates_pass():
    result = apply_operational_gates(
        numerical_parity=True,
        pair_p95_latency_ms=20.0,
        worst_class0_to_class1_error=0.1,
        worst_class1_to_class0_error=0.2,
        collapse_count=0,
    )
    assert result.eligible
