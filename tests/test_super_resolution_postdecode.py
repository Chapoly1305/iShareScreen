"""The processing-budget gate must include rendering and expose tail misses."""
from benchmarks.super_resolution.profile_postdecode import summarize


def test_budget_uses_total_postdecode_tail_and_reports_individual_misses():
    rows = [dict(assembly_ms=2, adapter_ms=20, render_completed_ms=3,
                 postdecode_ms=25)] * 90 + [dict(assembly_ms=2, adapter_ms=20,
                 render_completed_ms=18, postdecode_ms=40)] * 10
    result = summarize(rows,33)
    assert result["postdecode_ms"]["p50_ms"] < 33
    assert result["postdecode_ms"]["p95_ms"] == 40
    assert not result["budget"]["p95_pass"]
    assert not result["budget"]["every_sample_pass"]
    assert result["budget"]["over_budget"] == 10


def test_33ms_is_not_below_budget():
    result = summarize([dict(postdecode_ms=33)],33)
    assert not result["budget"]["p95_pass"]
    assert result["budget"]["over_budget"] == 1
