import copy

import pytest

from benchmarks.super_resolution import epoch_plots


@pytest.fixture
def history():
    return dict(schema="sr_epoch_results_v1", complete_cv=True,
        summary={m: dict(samples=8, clips=4, bicubic=dict(psnr=26., ssim=.9, lpips=.2)) for m in ("edsr", "imdn", "safmn")},
        jobs=[dict(model=m, fold=f, epochs=4) for m in ("edsr", "imdn", "safmn") for f in range(5)],
        curves=[dict(model=m, tier=t, epoch=e, psnr=28.+e, ssim=.94+e/1000, lpips=.06-e/1000)
                for m in ("edsr", "imdn", "safmn") for t in ("full", "selection") for e in range(0, 5, 2)],
        losses=[dict(model=m, fold=f, epoch=e, loss=.02/e) for m in ("edsr", "imdn", "safmn") for f in range(5) for e in range(1,5)])


def test_renders_png_svg_and_hashed_manifest_without_mutating_scores(history, tmp_path):
    pytest.importorskip("matplotlib")
    before = copy.deepcopy(history)
    manifest = epoch_plots.render(history, tmp_path / "figures", continuation_epoch=2)
    assert history == before
    assert manifest["smoothing"] is False and manifest["continuation_epoch"] == 2
    assert len(manifest["plotted_artifacts_sha256"]) == 6
    assert all((tmp_path / "figures" / name).stat().st_size > 1000 for name in manifest["plotted_artifacts_sha256"])
    svg = (tmp_path / "figures/full_curves.svg").read_text()
    assert "Budget extension at epoch 2" in svg and "PSNR (Epochs 2–4)" in svg
    assert "Validation Performance" in svg and "not best-so-far models" not in svg
    assert "All 8 held-out frames from 4 recordings" in svg
    selection = (tmp_path / "figures/selection_curves.svg").read_text()
    assert "Small fixed subset used for checkpoint selection" not in selection
    assert "Development CV, not an independent test" not in selection
    assert "Checkpoint Selection Metrics" in selection and "Budget extension at epoch 2" in selection
    assert "not best-so-far models" not in selection
    loss = (tmp_path / "figures/loss_curves.svg").read_text()
    assert "Training Loss" in loss and "EDSR (Epochs 2–4)" in loss
    assert "five content folds per model" not in loss
    with pytest.raises(FileExistsError):
        epoch_plots.render(history, tmp_path / "figures", continuation_epoch=2)


@pytest.mark.parametrize("change", ["partial", "boundary", "curve", "loss", "nonfinite"])
def test_rejects_incomplete_or_invalid_histories_before_creating_plots(history, tmp_path, change):
    boundary = 2
    if change == "partial":
        history["complete_cv"] = False
    elif change == "boundary":
        boundary = 5
    elif change == "curve":
        history["curves"].pop()
    elif change == "loss":
        history["losses"].pop()
    else:
        history["curves"][0]["psnr"] = float("nan")
    with pytest.raises(ValueError):
        epoch_plots.render(history, tmp_path / "bad", boundary)
    assert not (tmp_path / "bad").exists()
