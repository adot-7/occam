"""Property tests for the ablation math (`prd/03-ABLATION-AND-METRICS.md §4`).

Every test drives the same code path the engine uses; only the executor is
synthetic (see ``tests/_synthetic.py``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from occam.core.models import ConfidenceInterval
from occam.engine.ablation import (
    DEFAULT_EPS,
    FULL_REPEAT,
    AblationRow,
    AblationTable,
    ablate,
    ablate_variant,
    build_table,
    case_stratum,
    descendants,
    divergence,
    emit_table,
    influence,
    influence_ci,
    noise_rate,
    pair_outcomes,
    select_ablation_subset,
    sentinel_for,
    structural_fidelity,
    verdict,
)
from occam.engine.emit import collecting_sink, writer_sink
from occam.metrics import build_snapshot, emit_snapshot, pass_rate
from occam.metrics.stats import seeded_rng
from occam.store.reader import EventReader
from occam.store.reducer import reduce
from occam.store.writer import EventWriter
from tests._synthetic import (
    RoleEffect,
    SyntheticRunner,
    make_architecture,
    make_case,
    make_cases,
    make_role,
    make_run_result,
    pairs_from,
)

ROOT = Path(__file__).resolve().parents[1]
EVENT_SCHEMA = json.loads((ROOT / "schemas" / "events.schema.json").read_text(encoding="utf-8"))
EVENT_VALIDATOR = Draft202012Validator(EVENT_SCHEMA, format_checker=FormatChecker())

CASES = make_cases(10)
PASSING = {"c01", "c02", "c03", "c04", "c05", "c06"}


def _pipeline() -> tuple:
    """A 5-role FX pipeline shaped like the one in `03 §8`."""

    roles = [
        make_role("r_parse", name="Ledger Parser", justification="context_isolation"),
        make_role("r_rates", name="Rate Fetcher", inputs=["r_parse"], justification="parallel"),
        make_role("r_calc", name="FX Calculator", inputs=["r_rates"], justification="control"),
        make_role("r_verify", name="Verifier", inputs=["r_calc"], justification="verification"),
        make_role("r_report", name="Reporter", inputs=["r_calc"], justification="control"),
    ]
    return make_architecture(roles, final_role="r_report"), roles


def _prd_runner() -> SyntheticRunner:
    """The synthetic executor that reproduces the `03 §8` ablation table."""

    return SyntheticRunner(
        cases=CASES,
        passing=PASSING,
        drifting={"c07"},
        role_costs={
            "r_parse": 0.14,
            "r_rates": 0.54,
            "r_calc": 0.19,
            "r_verify": 0.06,
            "r_report": 0.07,
        },
        effects={
            "r_parse": RoleEffect(
                changed={case.id for case in CASES},
                broken={"c01", "c02", "c03", "c04"},
            ),
            "r_rates": RoleEffect(
                changed={case.id for case in CASES},
                broken={"c01", "c02", "c03", "c04", "c05"},
            ),
            "r_calc": RoleEffect(
                changed={case.id for case in CASES},
                broken={"c01", "c02", "c03", "c04", "c05"},
            ),
            "r_verify": RoleEffect(changed={"c07"}),
            "r_report": RoleEffect(changed={"c01", "c07", "c08"}, broken={"c01"}),
        },
    )


# --- property 1: an unreferenced role is a witness -------------------------------


def test_role_never_referenced_downstream_has_zero_divergence_and_is_a_witness() -> None:
    _, roles = _pipeline()
    inert = make_role("r_note", name="Note Taker", justification="ensemble")
    architecture = make_architecture([*roles, inert], final_role="r_report")
    all_cases = {case.id for case in CASES}
    runner = SyntheticRunner(
        cases=CASES,
        passing=PASSING,
        effects={
            role_id: RoleEffect(changed=all_cases, broken=set(PASSING))
            for role_id in ("r_parse", "r_rates", "r_calc", "r_verify", "r_report")
        },
        role_costs={role.id: 0.1 for role in roles},
    )
    full = runner.full_result(architecture)

    table = ablate(architecture, CASES, runner=runner, generation=0, full=full)
    row = table.row("r_note")

    # Nothing consumes its output, so nothing downstream can recompute.
    assert descendants(architecture, "r_note") == ()
    assert row.divergence == 0.0
    assert row.influence == 0.0
    assert row.verdict == "witness"
    assert table.witnesses == ["r_note"]
    assert table.prunable == ["r_note"]


def test_witness_verdict_survives_a_noise_floor_that_dwarfs_the_divergence() -> None:
    architecture, _ = _pipeline()
    runner = _prd_runner()
    full = runner.full_result(architecture)

    table = ablate(architecture, CASES, runner=runner, generation=0, full=full)

    # r_verify moved one answer of ten; the noise floor was one of ten.
    assert table.noise_rate == 0.1
    assert table.row("r_verify").divergence == 0.1
    assert table.row("r_verify").verdict == "witness"


# --- property 2: the sole producer of the answer is load-bearing -----------------


def test_sole_producer_of_the_final_answer_has_divergence_one_and_influence_pass_rate() -> None:
    solo = make_role("r_solo", name="Solver", justification="control")
    architecture = make_architecture([solo])
    passing = {"c01", "c02", "c03", "c04", "c05", "c06", "c07"}
    all_cases = {case.id for case in CASES}
    runner = SyntheticRunner(
        cases=CASES,
        passing=passing,
        role_costs={"r_solo": 0.4},
        effects={"r_solo": RoleEffect(changed=all_cases, broken=all_cases)},
    )
    full = runner.full_result(architecture)

    table = ablate(architecture, CASES, runner=runner, generation=0, full=full)
    row = table.row("r_solo")

    assert row.divergence == 1.0
    assert row.influence == pytest.approx(0.7)
    assert row.influence == pytest.approx(pass_rate(full))
    assert row.influence_ci.lo > 0
    assert row.verdict == "load_bearing"
    # A single role that works spends 100% of the budget on load-bearing work.
    assert table.structural_fidelity == pytest.approx(1.0)
    assert table.prunable == []


# --- property 3: noise widens the interval without flipping the estimate ---------


def _flip(rng, flags: list[bool], p: float) -> list[bool]:
    """Symmetric bit-flip channel: pure noise, no directional bias."""

    return [(not flag) if rng.random() < p else flag for flag in flags]


def test_pure_noise_flips_widen_the_ci_without_shifting_the_point_estimate_sign() -> None:
    full_flags = [True] * 16 + [False] * 4
    variant_flags = [True] * 6 + [False] * 14
    base = pairs_from(full_flags, variant_flags)
    base_point = influence(base)
    base_ci = influence_ci(base, resamples=800, seed=0, label="base")
    base_width = base_ci.hi - base_ci.lo
    assert base_point == pytest.approx(0.5)

    p = 0.15
    widths: list[float] = []
    points: list[float] = []
    for trial in range(30):
        rng = seeded_rng(trial, "noise")
        noisy = pairs_from(_flip(rng, full_flags, p), _flip(rng, variant_flags, p))
        interval = influence_ci(noisy, resamples=800, seed=trial, label="noisy")
        widths.append(interval.hi - interval.lo)
        points.append(influence(noisy))

    mean_width = sum(widths) / len(widths)
    mean_point = sum(points) / len(points)

    assert mean_width > base_width * 1.1
    assert min(widths) >= base_width - 1e-9
    # A symmetric flip channel shrinks a paired difference by (1 - 2p) in
    # expectation; it must not drag the estimate across zero.
    assert all(point > 0 for point in points)
    assert mean_point == pytest.approx(base_point * (1 - 2 * p), abs=0.08)


# --- property 4: structural fidelity ---------------------------------------------


def _row(role_id: str, share: float, row_verdict: str) -> AblationRow:
    return AblationRow(
        role_id=role_id,
        influence=0.0,
        influence_ci=ConfidenceInterval(lo=0.0, hi=0.0),
        divergence=0.0,
        cost_share=share,
        verdict=row_verdict,  # type: ignore[arg-type]
        n_cases=10,
    )


def test_structural_fidelity_is_the_load_bearing_share_of_spend() -> None:
    rows = [
        _row("a", 0.14, "load_bearing"),
        _row("b", 0.54, "load_bearing"),
        _row("c", 0.19, "load_bearing"),
        _row("d", 0.06, "witness"),
        _row("e", 0.07, "uncertain"),
    ]
    assert structural_fidelity(rows) == pytest.approx(0.87)
    # Two witnesses eating 40% of tokens: SF = 0.6 (`03 §5`).
    assert structural_fidelity(
        [_row("a", 0.6, "load_bearing"), _row("b", 0.4, "witness")]
    ) == pytest.approx(0.6)
    assert structural_fidelity([_row("a", 1.0, "harmful")]) == 0.0


def test_ablation_table_reproduces_the_prd_worked_example() -> None:
    architecture, _ = _pipeline()
    runner = _prd_runner()
    full = runner.full_result(architecture)

    table = ablate(architecture, CASES, runner=runner, generation=0, full=full)

    assert [(row.role_id, row.influence, row.divergence, row.cost_share) for row in table.rows] == [
        ("r_parse", 0.4, 1.0, 0.14),
        ("r_rates", 0.5, 1.0, 0.54),
        ("r_calc", 0.5, 1.0, 0.19),
        ("r_verify", 0.0, 0.1, 0.06),
        ("r_report", 0.1, 0.3, 0.07),
    ]
    assert [row.verdict for row in table.rows] == [
        "load_bearing",
        "load_bearing",
        "load_bearing",
        "witness",
        "uncertain",
    ]
    assert table.structural_fidelity == pytest.approx(0.87)
    assert table.n_cases == 10
    # The verification role is the witness — the Illusion paper's finding.
    assert table.row("r_verify").justification == "verification"


# --- verdict rule ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role_divergence", "ci", "noise", "expected"),
    [
        (0.0, (0.0, 0.0), 0.0, "witness"),
        (0.1, (0.2, 0.6), 0.1, "witness"),  # divergence within the noise band wins
        (0.15, (0.2, 0.6), 0.10, "witness"),  # exactly at noise + eps
        (0.9, (0.2, 0.6), 0.1, "load_bearing"),
        (0.9, (-0.6, -0.2), 0.1, "harmful"),
        (0.9, (-0.1, 0.3), 0.1, "uncertain"),
        (0.9, (0.0, 0.3), 0.1, "uncertain"),  # lo == 0 is not "clearly hurts"
    ],
)
def test_verdict_rule_matches_the_spec(role_divergence, ci, noise, expected) -> None:
    interval = ConfidenceInterval(lo=ci[0], hi=ci[1])
    assert verdict(role_divergence, interval, noise, DEFAULT_EPS) == expected


def test_uncertain_roles_are_never_prunable_but_harmful_ones_are() -> None:
    rows = [
        _row("keep", 0.4, "load_bearing"),
        _row("maybe", 0.2, "uncertain"),
        _row("inert", 0.2, "witness"),
        _row("bad", 0.2, "harmful"),
    ]
    table = AblationTable(generation=0, case_ids=["c01"], noise_rate=0.0, rows=rows)
    assert table.prunable == ["inert", "bad"]
    assert "maybe" not in table.prunable


def test_build_table_computes_the_same_numbers_offline_from_recorded_runs() -> None:
    architecture, _ = _pipeline()
    runner = _prd_runner()
    full = runner.full_result(architecture)
    knockouts = {
        role.id: runner.run_variant(
            architecture, CASES, variant=ablate_variant(role.id), ablate_role=role.id
        )
        for role in architecture.roles
    }

    table = build_table(
        architecture,
        generation=0,
        case_ids=[case.id for case in CASES],
        full=full,
        knockouts=knockouts,
        measured_noise_rate=0.1,
    )

    assert (
        table.model_dump()
        == ablate(architecture, CASES, runner=runner, generation=0, full=full).model_dump()
    )


def test_a_role_that_helps_when_removed_is_harmful() -> None:
    architecture = make_architecture([make_role("r_solo"), make_role("r_meddler")])
    all_cases = {case.id for case in CASES}
    runner = SyntheticRunner(
        cases=CASES,
        passing={"c01", "c02"},
        role_costs={"r_solo": 0.5, "r_meddler": 0.5},
        effects={
            "r_solo": RoleEffect(changed=all_cases, broken={"c01", "c02"}),
            "r_meddler": RoleEffect(changed=all_cases, fixed=all_cases),
        },
    )
    full = runner.full_result(architecture)

    table = ablate(architecture, CASES, runner=runner, generation=0, full=full)
    row = table.row("r_meddler")

    assert row.influence == pytest.approx(-0.8)
    assert row.influence_ci.hi < 0
    assert row.verdict == "harmful"
    assert "r_meddler" in table.prunable


# --- the paired machinery --------------------------------------------------------


def test_influence_is_paired_and_can_be_negative() -> None:
    # Same aggregate pass rate, answers flipping both ways: influence 0, but
    # divergence catches the churn.
    pairs = pairs_from([True, True, False, False], [True, False, True, False])
    assert influence(pairs) == pytest.approx(0.0)
    assert divergence(pairs) == pytest.approx(0.5)
    assert influence(pairs_from([False, False], [True, True])) == pytest.approx(-1.0)


def test_pairing_is_by_case_id_not_by_position() -> None:
    full = make_run_result("full", [("c01", "x", True), ("c02", "y", False)])
    knockout = make_run_result("ablate:r", [("c02", "y", False), ("c01", "z", False)])
    pairs = pair_outcomes(full, knockout, ["c01", "c02"])
    assert [pair.case_id for pair in pairs] == ["c01", "c02"]
    assert [pair.changed for pair in pairs] == [True, False]
    assert influence(pairs) == pytest.approx(0.5)


def test_pairing_rejects_a_variant_that_is_missing_a_case() -> None:
    full = make_run_result("full", [("c01", "x", True), ("c02", "y", True)])
    knockout = make_run_result("ablate:r", [("c01", "x", True)])
    with pytest.raises(ValueError, match="c02"):
        pair_outcomes(full, knockout, ["c01", "c02"])


def test_noise_floor_compares_two_full_runs_and_is_measured_without_the_cache() -> None:
    architecture, _ = _pipeline()
    runner = _prd_runner()
    full = runner.full_result(architecture)
    repeat = runner.run_variant(architecture, CASES, variant=FULL_REPEAT, use_cache=False)

    assert noise_rate(full, repeat, [case.id for case in CASES]) == pytest.approx(0.1)

    runner.calls.clear()
    ablate(architecture, CASES, runner=runner, generation=0, full=full)
    repeat_calls = [call for call in runner.calls if call[0] == FULL_REPEAT]
    assert repeat_calls == [(FULL_REPEAT, None, False)]
    assert [call[0] for call in runner.calls if call[0] != FULL_REPEAT] == [
        ablate_variant(role_id)
        for role_id in ("r_parse", "r_rates", "r_calc", "r_verify", "r_report")
    ]


def test_influence_ci_is_deterministic_for_the_same_seed_and_role() -> None:
    pairs = pairs_from([True] * 7 + [False] * 3, [False] * 10)
    first = influence_ci(pairs, resamples=2000, seed=0, label="r_solo")
    random.random()  # the bootstrap must not read the global RNG
    second = influence_ci(pairs, resamples=2000, seed=0, label="r_solo")
    assert first == second
    assert seeded_rng(0, "r_solo").random() != seeded_rng(1, "r_solo").random()
    assert seeded_rng(0, "r_solo").random() != seeded_rng(0, "r_other").random()


# --- knockout semantics and subset selection -------------------------------------


def test_descendants_are_exactly_the_roles_that_must_recompute() -> None:
    architecture, _ = _pipeline()
    assert descendants(architecture, "r_parse") == ("r_rates", "r_calc", "r_verify", "r_report")
    assert descendants(architecture, "r_calc") == ("r_verify", "r_report")
    assert descendants(architecture, "r_report") == ()
    assert sentinel_for(architecture.roles[0]) == "[no input from Ledger Parser]"


def test_ablation_subset_is_stratified_and_deterministic() -> None:
    cases = [
        make_case("c01", has_weekend_or_holiday=True),
        make_case("c02"),
        make_case("c03"),
        make_case("c04", has_bank_fee=True),
        make_case("c05"),
        make_case("c06", has_weekend_or_holiday=True),
    ]
    subset = select_ablation_subset(cases, 3)
    assert [case.id for case in subset] == ["c01", "c02", "c04"]
    assert {case_stratum(case) for case in subset} == {
        "has_weekend_or_holiday",
        "plain",
        "has_bank_fee",
    }
    assert select_ablation_subset(cases, 3) == subset
    assert select_ablation_subset(cases, 99) == cases


def test_ablate_can_target_a_subset_of_roles() -> None:
    architecture, _ = _pipeline()
    runner = _prd_runner()
    full = runner.full_result(architecture)

    table = ablate(
        architecture,
        CASES,
        runner=runner,
        generation=2,
        full=full,
        roles=["r_verify", "r_calc"],
    )

    assert [row.role_id for row in table.rows] == ["r_calc", "r_verify"]
    with pytest.raises(ValueError, match="unknown role ids"):
        ablate(architecture, CASES, runner=runner, generation=2, full=full, roles=["nope"])


# --- events ------------------------------------------------------------------------


def test_emitted_ablation_events_validate_against_the_event_schema() -> None:
    architecture, _ = _pipeline()
    runner = _prd_runner()
    full = runner.full_result(architecture)
    collected: list[dict] = []

    table = ablate(
        architecture, CASES, runner=runner, generation=0, full=full, sink=collecting_sink(collected)
    )

    assert [event["type"] for event in collected] == [
        "ablation.started",
        *["ablation.role"] * 5,
        "ablation.completed",
    ]
    for seq, event in enumerate(collected):
        EVENT_VALIDATOR.validate({"ts": "2026-09-06T09:00:00Z", "run_id": "t", "seq": seq, **event})
    assert collected[0]["data"]["case_ids"] == [case.id for case in CASES]
    assert collected[0]["data"]["noise_rate"] == 0.1
    assert collected[-1]["data"] == {
        "generation": 0,
        "structural_fidelity": 0.87,
        "witnesses": ["r_verify"],
    }
    assert emit_table(collecting_sink([]), table) is table


def test_ablation_and_metrics_events_round_trip_through_the_store(tmp_path) -> None:
    architecture, _ = _pipeline()
    runner = _prd_runner()
    full = runner.full_result(architecture)
    writer = EventWriter(tmp_path / "run", run_id="run_test")
    sink = writer_sink(writer)

    table = ablate(architecture, CASES, runner=runner, generation=0, full=full, sink=sink)
    snapshot = build_snapshot(
        generation=0,
        full=full,
        noise_rate=table.noise_rate,
        structural_fidelity=table.structural_fidelity,
    )
    emit_snapshot(sink, snapshot)
    writer.close()

    state = reduce(EventReader(tmp_path / "run").read())
    generation = state.generations["g000"]
    assert generation.ablation is not None
    assert len(generation.ablation["rows"]) == 5
    assert generation.ablation["structural_fidelity"] == 0.87
    assert generation.ablation["witnesses"] == ["r_verify"]
    assert generation.metrics is not None
    assert generation.metrics.reliability == pytest.approx(0.9)
    assert generation.metrics.structural_fidelity == pytest.approx(0.87)
