"""Acceptance checks for the committed v3 FX task packs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from occam.core import Task
from occam.store.schema import validate_task
from occam.tasks.fx_reference import reference_case
from occam.tools.fx import FXClient

ROOT = Path(__file__).resolve().parents[1]
PACKS = (ROOT / "tasks" / "fx_recon_a", ROOT / "tasks" / "fx_recon_b")
FEE_PATTERN = re.compile(r"net of bank fee INR ([\d,]+\.\d\d)")


def test_committed_packs_have_required_mix_and_manifest_contract() -> None:
    for pack in PACKS:
        manifest = yaml.safe_load((pack / "task.yaml").read_text(encoding="utf-8"))
        validate_task(manifest)
        task = Task.model_validate(manifest)
        cases = [
            json.loads(line)
            for line in (pack / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert task.checker == "fx_total"
        assert task.memory == "memory/fx_recon"
        assert len(cases) == 20
        assert all(6 <= case["meta"]["n_invoices"] <= 10 for case in cases)
        assert sum(case["meta"]["has_weekend_or_holiday"] for case in cases) >= 6
        assert sum(case["meta"]["has_bank_fee"] for case in cases) >= 5
        assert sum(case["meta"]["has_cross_ccy"] for case in cases) >= 5
        assert sum(case["meta"]["has_jpy"] for case in cases) >= 4


def test_committed_pack_tolerance_sanity_is_nontrivial() -> None:
    for pack in PACKS:
        totals = [abs(case["expected"]["total_inr"]) for case in _cases(pack)]
        assert min(totals) > 5.0
        assert max(totals) > min(totals)
        tightest_relative = min(max(5.0, 0.001 * total) / total for total in totals)
        assert tightest_relative == 0.001


def test_committed_pack_bank_fees_exceed_the_grader_tolerance() -> None:
    """A fee below tolerance would let an agent ignore D1 and still pass."""

    for pack in PACKS:
        graded = 0
        for case in _cases(pack):
            fees = [float(fee.replace(",", "")) for fee in FEE_PATTERN.findall(case["input"])]
            if not fees:
                continue
            graded += 1
            tolerance = max(5.0, 0.001 * abs(case["expected"]["total_inr"]))
            assert sum(fees) > tolerance, f"{case['id']}: fee {sum(fees)} <= tolerance {tolerance}"
        assert graded >= 5


def test_committed_case_text_alone_reproduces_the_expected_answer() -> None:
    """The ledger text an agent sees must carry the whole answer, losslessly."""

    with FXClient(cache_dir=ROOT / "data" / "fx_cache") as client:
        for pack in PACKS:
            for case in _cases(pack):
                assert reference_case({"input": case["input"]}, client=client) == case["expected"]
        assert all(call.cached for call in client.calls)


def _cases(pack: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (pack / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
