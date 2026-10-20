"""Local curation invariants, without public weight downloads or training."""
import copy
import hashlib

import pytest

from benchmarks.super_resolution.curate import (
    decide, nearest_representatives, pixel_difference, safe_duplicate, shared_references,
    verify_candidates, write_report,
)
from benchmarks.super_resolution.folds import make_plan
from benchmarks.super_resolution.prepare_curation import candidate_indices, inside


def test_dense_schedule_preserves_unscheduled_baseline_indices():
    assert candidate_indices(31, 10, [7, 30]) == [0, 7, 10, 20, 30]
    with pytest.raises(ValueError):
        candidate_indices(31, 0)
    with pytest.raises(ValueError):
        candidate_indices(31, 10, [31])
    with pytest.raises(ValueError):
        candidate_indices(31, 10, [True])


def test_media_paths_cannot_escape_the_candidate_root(tmp_path):
    assert inside(tmp_path, "a/b.png") == tmp_path / "a/b.png"
    with pytest.raises(ValueError, match="escapes"):
        inside(tmp_path, "../other.png")


def test_local_pixel_guard_catches_small_detail_including_partial_border_tiles():
    np = pytest.importorskip("numpy")
    a = np.zeros((35, 35, 3), dtype=np.uint8)
    b = a.copy(); b[-2:, -2:] = 255
    difference = pixel_difference(a, b)
    assert difference["mae"] < 1
    assert difference["max_tile_mae"] > 100
    assert not safe_duplicate({"hr": difference}, 1, 1)
    assert safe_duplicate({"hr": pixel_difference(a, a)}, .25, 1)


def image_pair(tmp_path, capture, frame, hr=None, lr=None, baseline=False, group=None, fold=0):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    row = dict(capture=capture, frame=frame, group=group or capture, fold=fold,
               baseline=baseline, time_s=frame / 60, content="terminal")
    if lr is None:
        lr = np.zeros((20, 20, 3), dtype=np.uint8)
    if hr is None:
        hr = lr.repeat(2, 0).repeat(2, 1)
    for role, rgb in (("hr", hr), ("lr", lr)):
        path = tmp_path / f"{capture}-{frame}-{role}.png"
        Image.fromarray(rgb).save(path)
        row[role] = path.name
        row[f"{role}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        row[f"{role}_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
    return row


def test_duplicate_guard_preserves_frozen_anchors_and_distinct_lr_degradation(tmp_path):
    np = pytest.importorskip("numpy")
    rows = [image_pair(tmp_path, "a", 0, baseline=True), image_pair(tmp_path, "a", 10)]
    lr = np.zeros((20, 20, 3), dtype=np.uint8); lr[0:4, 0:4] = 90
    rows.append(image_pair(tmp_path, "a", 20, hr=np.zeros((40, 40, 3), dtype=np.uint8), lr=lr))
    decisions = decide(rows, {1: 0, 2: 0}, tmp_path, .25, 1)
    assert [d["selected"] for d in decisions] == [True, False, True]
    assert decisions[0]["reason"] == "frozen_baseline_anchor"
    assert decisions[2]["reason"] == "native_detail_or_degradation_preserved"
    assert decisions[2]["differences"]["hr"]["mae"] == 0


def test_new_cross_group_reference_overlap_is_quarantined_without_changing_old_data(tmp_path):
    rows = [image_pair(tmp_path, "a", 0, baseline=True), image_pair(tmp_path, "b", 10, fold=1)]
    assert shared_references(rows) == {rows[0]["hr_rgb_sha256"]}
    decisions = decide(rows, {}, tmp_path, .25, 1)
    assert [d["selected"] for d in decisions] == [True, False]
    assert decisions[1]["reason"] == "quarantined_reference_overlap"


def test_exact_paired_dedup_prefers_frozen_anchor_even_when_it_occurs_later(tmp_path):
    rows = [image_pair(tmp_path, "a", 0), image_pair(tmp_path, "a", 60, baseline=True)]
    decisions = decide(rows, {}, tmp_path, .25, 1, workers=2)
    assert [d["selected"] for d in decisions] == [False, True]
    assert decisions[0]["reason"] == "exact_pair_duplicate"
    assert decisions[0]["reference_index"] == 1


def test_tool_reference_is_resolved_to_retained_exact_pair_representative(tmp_path):
    np = pytest.importorskip("numpy")
    rows = [image_pair(tmp_path, "a", 0, baseline=True), image_pair(tmp_path, "a", 10),
            image_pair(tmp_path, "a", 20, hr=np.full((40, 40, 3), 1, dtype=np.uint8))]
    decisions = decide(rows, {2: 1}, tmp_path, 2, 2)
    assert not decisions[1]["selected"] and decisions[1]["reference_index"] == 0
    assert not decisions[2]["selected"] and decisions[2]["reference_index"] == 0


def test_duplicate_suggestions_cannot_cross_captures(tmp_path):
    np = pytest.importorskip("numpy")
    rows = [image_pair(tmp_path, "a", 0, baseline=True),
            image_pair(tmp_path, "b", 10, hr=np.full((40, 40, 3), 120, dtype=np.uint8))]
    with pytest.raises(ValueError, match="crosses captures"):
        decide(rows, {1: 0}, tmp_path, .25, 1)


def test_dropped_or_quarantined_reference_cannot_remove_another_candidate(tmp_path):
    np = pytest.importorskip("numpy")
    rows = [image_pair(tmp_path, "a", 0, baseline=True), image_pair(tmp_path, "b", 0),
            image_pair(tmp_path, "b", 10, hr=np.full((40, 40, 3), 120, dtype=np.uint8))]
    result = decide(rows, {2: 1}, tmp_path, .25, 1)
    assert not result[1]["selected"]
    assert result[2]["selected"]


def test_candidate_audit_rejects_changed_fold_pixels_and_missing_anchor(tmp_path):
    np = pytest.importorskip("numpy")
    plan = make_plan({"clips": [dict(id="a", frames=100, fps="60/1"),
                                dict(id="b", frames=100, fps="60/1")]}, n_folds=2)
    rows = [image_pair(tmp_path, c["id"], 0, baseline=True, fold=c["fold"],
                       hr=np.full((40, 40, 3), i, dtype=np.uint8))
            for i, c in enumerate(plan["clips"])]
    baseline = {"pairs": [dict(r) for r in rows]}
    verify_candidates({"candidates": rows}, tmp_path, plan, baseline)
    rows[0]["fold"] = 1 - rows[0]["fold"]
    with pytest.raises(ValueError, match="frozen"):
        verify_candidates({"candidates": rows}, tmp_path, plan, baseline)
    rows[0]["fold"] = 1 - rows[0]["fold"]
    with pytest.raises(ValueError, match="omit"):
        verify_candidates({"candidates": rows[1:]}, tmp_path, plan, baseline)
    (tmp_path / rows[0]["lr"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed candidate"):
        verify_candidates({"candidates": rows}, tmp_path, plan, baseline)


def test_report_is_english_local_and_contains_all_captures(tmp_path):
    rows = [image_pair(tmp_path, "a", 0, baseline=True), image_pair(tmp_path, "a", 10)]
    decisions = decide(rows, {1: 0}, tmp_path, .25, 1)
    output = tmp_path / "review"; output.mkdir()
    summaries = write_report(rows, decisions, output, tmp_path)
    assert summaries[0]["selected"] == 1 and summaries[0]["baseline"] == 1
    assert (output / "contact-00.jpg").exists()
    text = (output / "index.html").read_text()
    assert "proposal only" in text and "https://" not in text


def test_fiftyone_public_api_uses_precomputed_embeddings_locally(tmp_path, monkeypatch):
    # Optional integration smoke test: no image reads, model downloads, or remote API.
    import importlib.util
    if importlib.util.find_spec("fiftyone") is None:
        pytest.skip("Optional FiftyOne environment not installed")
    monkeypatch.setenv("FIFTYONE_DO_NOT_TRACK", "true")
    monkeypatch.setenv("FIFTYONE_DATABASE_DIR", str(tmp_path / "database"))
    import fiftyone as fo
    import fiftyone.brain as fob
    np = pytest.importorskip("numpy")
    dataset = fo.Dataset()
    try:
        samples = [fo.Sample(filepath=str(tmp_path / f"{i}.png")) for i in range(3)]
        dataset.add_samples(samples)
        array = np.array([[1., 0.], [1., 0.], [0., 1.]])
        index = fob.compute_near_duplicates(dataset, threshold=.005, embeddings=array.copy())
        assert len(index.duplicate_ids) == 1
        # Do NOT trust neighbors_map's representative-only re-centering distances.
        mapping, scores = nearest_representatives(array, [s.id for s in samples],
                                                  index.unique_ids, index.duplicate_ids, .005)
        assert mapping == {1: 0}
        assert scores[1] == pytest.approx(0)
    finally:
        dataset.delete()


def test_reference_matching_never_accepts_distant_or_degenerate_vectors():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    mapping, scores = nearest_representatives(np.array([[1., 0.], [0., 1.]]),
                                              ["a", "b"], ["a"], ["b"], .005)
    assert mapping == scores == {}
    mapping, scores = nearest_representatives(np.ones((2, 2)),
                                              ["a", "b"], ["a"], ["b"], .005)
    assert mapping == scores == {}


def test_final_audit_checks_schedule_baseline_and_consistent_selection(tmp_path):
    from benchmarks.super_resolution.audit_curation import audit, overview_sheets
    np = pytest.importorskip("numpy")
    plan = make_plan({"clips": [dict(id="a", frames=20, fps="60/1"),
                                dict(id="b", frames=20, fps="60/1")]}, n_folds=2)
    rows = []
    for i, clip in enumerate(plan["clips"]):
        for frame in (0, 10):
            rows.append(image_pair(tmp_path, clip["id"], frame, baseline=frame == 0,
                        fold=clip["fold"], hr=np.full((40, 40, 3), i * 40 + frame, dtype=np.uint8)))
    baseline = {"pairs": [r for r in rows if r["baseline"]]}
    decisions = decide(rows, {}, tmp_path, .25, 1)
    proposal = dict(decisions=decisions, selected=rows, training_ready=False)
    candidates = dict(candidates=rows, every=10)
    summary = audit(proposal, candidates, plan, baseline)
    assert summary["additions"] == summary["baseline_preserved"] == 2
    bad = copy.deepcopy(proposal); bad["selected"].pop()
    with pytest.raises(ValueError, match="selected records"):
        audit(bad, candidates, plan, baseline)
    bad = copy.deepcopy(candidates); bad["every"] = 20
    with pytest.raises(ValueError, match="schedule"):
        audit(proposal, bad, plan, baseline)
    output = tmp_path / "sheets"; output.mkdir()
    sheets = overview_sheets(rows, {(r["capture"], r["frame"]) for r in rows}, tmp_path, output)
    assert sheets == ["coverage-00.jpg"] and (output / sheets[0]).exists()


def test_visual_quality_review_quarantines_new_frames_but_flags_legacy_anchors(tmp_path):
    from benchmarks.super_resolution.review_curation import apply_review, quality_flags
    np = pytest.importorskip("numpy")
    rows = [image_pair(tmp_path, "a", 0, baseline=True),
            image_pair(tmp_path, "a", 10, hr=np.full((40, 40, 3), 10, dtype=np.uint8)),
            image_pair(tmp_path, "a", 20, hr=np.full((40, 40, 3), 20, dtype=np.uint8))]
    parent = decide(rows, {}, tmp_path, .25, 1)
    review = {"ranges": [dict(capture="a", start=0, end=10, reason="capture_ui")]}
    result = apply_review(rows, parent, review)
    assert [d["selected"] for d in result] == [True, False, True]
    assert result[0]["future_training_eligible"] is False
    assert result[1]["reason"] == "manual_quality_quarantine"
    assert quality_flags(rows[2], review) == []
    assert all("quality_flags" not in d for d in parent)  # Parent is immutable.


def test_visual_review_restores_clean_frame_when_its_old_reference_is_quarantined(tmp_path):
    from benchmarks.super_resolution.review_curation import apply_review
    np = pytest.importorskip("numpy")
    rows = [image_pair(tmp_path, "a", 0),
            image_pair(tmp_path, "a", 10, hr=np.full((40, 40, 3), 1, dtype=np.uint8))]
    parent = [dict(index=0, selected=True, reason="distinct_representative"),
              dict(index=1, selected=False, reason="near_duplicate_confirmed", reference_index=0)]
    result = apply_review(rows, parent, {"ranges": [dict(capture="a", start=0, end=0, reason="capture_ui")]})
    assert not result[0]["selected"] and result[1]["selected"]
    assert result[1]["reason"] == "quality_reference_unavailable"


def test_native_review_preserves_any_hr_pixel_change_without_ocr(tmp_path):
    from benchmarks.super_resolution.review_curation import apply_review
    np = pytest.importorskip("numpy")
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    changed = image.copy(); changed[4, 4] = 1
    rows = [image_pair(tmp_path, "a", 0, hr=image, baseline=True),
            image_pair(tmp_path, "a", 10, hr=changed)]
    parent = [dict(index=0, selected=True, reason="frozen_baseline_anchor"),
              dict(index=1, selected=False, reason="near_duplicate_confirmed", reference_index=0,
                   differences={"hr": pixel_difference(image, changed)})]
    result = apply_review(rows, parent, {"ranges": [], "require_exact_hr_for_near_dedup": True})
    assert result[1]["selected"] and result[1]["reason"] == "hr_change_preserved_after_visual_review"
