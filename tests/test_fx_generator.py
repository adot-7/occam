"""Deterministic pack-generation and mix-report checks."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import gen_fx_cases


class FakeClient:
    def __init__(self, **_: object) -> None:
        pass

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def fx_rate(self, date: str, base: str, symbol: str) -> dict[str, object]:
        value = 0.5 + (sum(map(ord, date + base + symbol)) % 10_000) / 100
        return {
            "requested_date": date,
            "rate_date": date,
            "base": base,
            "symbol": symbol,
            "rate": value,
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
    assert (first / "task.yaml").read_bytes() == (second / "task.yaml").read_bytes()
    assert (first / "cases.jsonl").read_bytes() == (second / "cases.jsonl").read_bytes()

    cases = [json.loads(line) for line in (first / "cases.jsonl").read_text().splitlines()]
    assert len(cases) == 20
    assert all(6 <= case["meta"]["n_invoices"] <= 10 for case in cases)
