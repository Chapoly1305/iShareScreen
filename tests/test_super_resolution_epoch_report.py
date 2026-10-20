"""Saved-result audit, including tampered coverage, selection and update history."""
import json
from pathlib import Path

import pytest

from test_super_resolution_epochs import tiny_campaign


def test_report_uses_the_shared_plot_renderer(tmp_path, monkeypatch):
    from benchmarks.super_resolution import epoch_plots, epoch_report
    observed = []
    monkeypatch.setattr(epoch_plots, "render", lambda *args, **kwargs: observed.append((args, kwargs)))
    result = {"audited": True}
    epoch_report.plot_curves(result, tmp_path)
    assert observed == [((result, tmp_path), {"audit_path": tmp_path / "audit.json"})]


@pytest.fixture
def completed_campaign(tiny_campaign, monkeypatch):
    from benchmarks.super_resolution import epoch_report
    runner, args, _ = tiny_campaign
    monkeypatch.setattr(epoch_report, "expected_metrics", lambda: {"mock": True})
    root = args.output.with_name("completed")
    for fold in range(5):
        args.fold = fold
        args.output = root / f"imdn-fold-{fold}"
        assert runner.run(args) == 0
    return epoch_report, root, args.folds, args.pairs


def test_epoch_audit_reconstructs_metrics_curves_and_selected_checkpoints(completed_campaign, tmp_path):
    report, root, folds, pairs = completed_campaign
    result = report.audit(root, folds, pairs, ["imdn"])
    assert len(result["jobs"]) == 5
    assert result["summary"]["imdn"]["samples"] == 5
    assert result["summary"]["imdn"]["clips"] == 5
    assert len(result["per_frame"]) == len(result["per_clip"]) == 5
    assert len(result["losses"]) == 15
    assert {r["epoch"] for r in result["curves"] if r["tier"] == "full"} == {0, 2, 3}
    assert {r["epoch"] for r in result["curves"] if r["tier"] == "selection"} == {0, 1, 2, 3}
    report.write_report(result, tmp_path / "report", plots=False)
    assert (tmp_path / "report/REPORT.md").is_file()
    assert (tmp_path / "report/per_frame.csv").is_file()
    assert (tmp_path / "report/audit.json").is_file()


def test_report_generates_shared_figures_and_audit_bound_manifest(completed_campaign, tmp_path):
    pytest.importorskip("matplotlib")
    from isharescreen.enhancement.models import sha256
    report, root, folds, pairs = completed_campaign
    result = report.audit(root, folds, pairs, ["imdn"])
    output = tmp_path / "figures"
    report.write_report(result, output)
    manifest = json.loads((output / "plot_manifest.json").read_text())
    assert manifest["source_audit_sha256"] == sha256(output / "audit.json")
    assert len(manifest["plotted_artifacts_sha256"]) == 6
    assert "Validation Performance" in (output / "full_curves.svg").read_text()
    assert "Checkpoint Selection Metrics" in (output / "selection_curves.svg").read_text()


@pytest.mark.parametrize("change,match", [
    ("coverage", "held-out samples"),
    ("summary", "metric summary"),
    ("checkpoint", "checkpoint changed"),
    ("curve", "validation curve"),
    ("updates", "training update"),
    ("extra_directory", "authoritative copies"),
])
def test_epoch_audit_rejects_tampered_artifacts(completed_campaign, change, match):
    report, root, folds, pairs = completed_campaign
    folder = root / "imdn-fold-0"
    if change in ("coverage", "summary"):
        path = folder / "best-validation.json"
        value = json.loads(path.read_text())
        if change == "coverage":
            value["samples"][0]["frame"] = 119
        else:
            value["summary"]["psnr"] += 1
        path.write_text(json.dumps(value))
    elif change == "checkpoint":
        with (folder / "best.pt").open("ab") as handle:
            handle.write(b"changed")
    elif change in ("curve", "updates"):
        path = folder / "progress.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        event = "full" if change == "curve" else "update"
        index = next(i for i, row in enumerate(rows) if row["event"] == event)
        rows.pop(index)
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    else:
        (root / "paused-source-copy").mkdir()
    with pytest.raises(ValueError, match=match):
        report.audit(root, folds, pairs, ["imdn"])


def test_metric_result_rejects_nonfinite_per_frame_values(tmp_path):
    from benchmarks.super_resolution.epoch_report import metric_result
    path = tmp_path / "metric.json"
    path.write_text(json.dumps(dict(summary=dict(psnr=30, ssim=.9, lpips=.1), samples=[
        dict(capture="c", frame=0, psnr=30, ssim=.9, lpips=float("nan"))])))
    with pytest.raises(ValueError, match="Nonfinite"):
        metric_result(path, {("c", 0)})


def test_metric_result_uses_equal_clip_not_equal_frame_weighting(tmp_path):
    from benchmarks.super_resolution.epoch_report import metric_result
    rows = [dict(capture="a", frame=i, psnr=10., ssim=.8, lpips=.2) for i in range(3)]
    rows.append(dict(capture="b", frame=0, psnr=30., ssim=1., lpips=0.))
    path = tmp_path / "metric.json"
    path.write_text(json.dumps(dict(summary=dict(psnr=20., ssim=.9, lpips=.1), samples=rows)))
    result = metric_result(path, {(r["capture"], r["frame"]) for r in rows})
    assert result["summary"]["psnr"] == 20  # Frame pooling would incorrectly give 15.
