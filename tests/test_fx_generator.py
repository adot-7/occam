"""Deterministic pack-generation and mix-report checks."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import gen_fx_cases


class FakeClient:
    """Offline stand-in whose rates stay in the real INR range.

    The magnitudes matter: the generator asserts that every bank fee clears the
    grader's tolerance, and that invariant is only meaningful against rates of a
    realistic size.
    """

    def __init__(self, **_: object) -> None:
        pass

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def fx_rate(self, date: str, base: str, symbol: str) -> dict[str, object]:
        drift = 1.0 + ((sum(map(ord, date)) % 41) - 20) / 1000
        return {
            "requested_date": date,
            "rate_date": date,
            "base": base,
            "symbol": symbol,
            "rate": gen_fx_cases.APPROX_INR_RATES[base] * drift,
        }


def test_generator_reports_required_mix_and_is_byte_identical(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gen_fx_cases, "FXClient", FakeClient)
    first = tmp_path / "first" / "fx_recon_a"
    second = tmp_path / "second" / "fx_recon_a"
    report_first = gen_fx_cases.generate_pack(
        seed=7,
        n=20,
        out=first,
        cache_dir=tmp_path / "cache",
    )
    report_second = gen_fx_cases.generate_pack(
        seed=7,
        n=20,
        out=second,
        cache_dir=tmp_path / "cache",
    )

    assert report_first == report_second
    assert report_first["mix"]["holiday_weekend"] >= 6
    assert report_first["mix"]["bank_fee"] >= 5
    assert report_first["mix"]["cross_currency"] >= 5
    assert report_first["mix"]["jpy"] >= 4
    assert report_first["min_bank_fee_headroom"] > 1.0
    assert (first / "task.yaml").read_bytes() == (second / "task.yaml").read_bytes()
    assert (first / "cases.jsonl").read_bytes() == (second / "cases.jsonl").read_bytes()

    cases = [json.loads(line) for line in (first / "cases.jsonl").read_text().splitlines()]
    assert len(cases) == 20
    assert all(6 <= case["meta"]["n_invoices"] <= 10 for case in cases)


def test_generated_packs_are_written_with_lf_newlines(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gen_fx_cases, "FXClient", FakeClient)
    out = tmp_path / "fx_recon_a"
    gen_fx_cases.generate_pack(seed=7, n=20, out=out, cache_dir=tmp_path / "cache")

    for name in ("task.yaml", "cases.jsonl"):
        assert b"\r\n" not in (out / name).read_bytes()
