"""Screenshot gallery checks without model downloads or GPU inference."""
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from benchmarks.super_resolution.candidate_stills import refresh, write_gallery
from benchmarks.super_resolution.render_candidate_media import write_matrix_gallery
from isharescreen.enhancement.models import sha256


def picture(root, name, **metadata):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), "white").save(path)
    return dict(file=name, sha256=sha256(path), scene="code", title="Code <Detail>", **metadata)


def test_refresh_preserves_detail_pixels_and_removes_obsolete_metadata(tmp_path):
    detail = picture(tmp_path, "images/detail.png")
    overview = picture(tmp_path, "images/overview.png", kind="overview")
    original = (tmp_path / detail["file"]).read_bytes()
    manifest = dict(pictures=[detail, overview], video="old.mp4", video_sha256="old",
                    chapters=[{}], commands=[["ffmpeg"]])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    refresh(SimpleNamespace(output=tmp_path))

    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["pictures"] == [detail]
    assert saved["presentation"] == "screenshots"
    assert not any(key in saved for key in ("video", "video_sha256", "chapters", "commands"))
    assert (tmp_path / detail["file"]).read_bytes() == original
    assert not (tmp_path / overview["file"]).exists()


def test_refresh_rejects_changed_images_without_replacing_manifest(tmp_path):
    item = picture(tmp_path, "images/detail.png")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(pictures=[item])))
    original = path.read_bytes()
    Image.new("RGB", (24, 16), "black").save(tmp_path / item["file"])

    with pytest.raises(ValueError, match="Changed or unsafe"):
        refresh(SimpleNamespace(output=tmp_path))

    assert path.read_bytes() == original
    assert not (tmp_path / "index.html").exists()


def test_screenshot_gallery_has_navigation_but_no_video_or_input_section(tmp_path):
    item = picture(tmp_path, "images/detail.png")
    write_gallery(tmp_path, dict(pictures=[item], raw_inputs=[{}]))
    page = (tmp_path / "index.html").read_text()
    assert 'href="#scene-code"' in page
    assert "Code &lt;Detail&gt;" in page
    assert "unenhanced 540p input" in page
    assert "<video" not in page
    assert "Actual Model Inputs" not in page


def test_optional_video_gallery_also_omits_separate_input_section(tmp_path):
    write_matrix_gallery(tmp_path, dict(video="comparison.mp4", chapters=[], pictures=[],
                                        raw_inputs=[{}]))
    page = (tmp_path / "index.html").read_text()
    assert '<video id="comparison"' in page
    assert "Actual Model Inputs" not in page
