from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class GateProfile:
    pair_p95_latency_ms_max: float = 33.333
    directional_error_max: float = 0.50
    require_numerical_parity: bool = True
    forbid_class_collapse: bool = True


@dataclass(frozen=True)
class GateResult:
    parity_pass: bool
    latency_pass: bool
    safety_pass: bool
    eligible: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def apply_operational_gates(
    *,
    numerical_parity: bool,
    pair_p95_latency_ms: float,
    worst_class0_to_class1_error: float,
    worst_class1_to_class0_error: float,
    collapse_count: int,
    profile: GateProfile = GateProfile(),
) -> GateResult:
    parity_pass = bool(numerical_parity or not profile.require_numerical_parity)
    latency_pass = bool(pair_p95_latency_ms <= profile.pair_p95_latency_ms_max)
    directional_pass = (
        worst_class0_to_class1_error <= profile.directional_error_max
        and worst_class1_to_class0_error <= profile.directional_error_max
    )
    collapse_pass = bool(collapse_count == 0 or not profile.forbid_class_collapse)
    safety_pass = bool(directional_pass and collapse_pass)
    return GateResult(
        parity_pass=parity_pass,
        latency_pass=latency_pass,
        safety_pass=safety_pass,
        eligible=bool(parity_pass and latency_pass and safety_pass),
    )
