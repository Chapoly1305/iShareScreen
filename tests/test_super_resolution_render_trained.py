import json
from pathlib import Path

import pytest

from benchmarks.super_resolution import render_trained, training_state
from isharescreen.enhancement.models import sha256


@pytest.fixture
def checkpoint_case(tmp_path, monkeypatch):
    sources = {"source": "pinned"}
    monkeypatch.setattr(render_trained.train_epochs, "source_hashes", lambda: sources)
    policy = dict(schema="sr_true_epoch_protocol_v1", sources=sources, folds_sha256="frozen-folds",
        schedule=dict(epochs=1500, job=dict(model="edsr", fold=2, validation_captures=["held-out"], train_captures=["training"])))
    (tmp_path/"last.pt").write_bytes(b"current-not-selected")
    (tmp_path/"protocol.json").write_text(json.dumps(policy))
    (tmp_path/"COMPLETED.json").write_text(json.dumps(dict(epochs=1500, protocol_binding=training_state.fingerprint(policy))))
    scene = dict(fold=2, capture="held-out")
    data = dict(folds_sha256="frozen-folds")
    audit = dict(complete_cv=True, folds_sha256="frozen-folds", evidence={"edsr-fold-2":{"last.pt":sha256(tmp_path/"last.pt")}})
    return tmp_path, scene, data, audit


def test_accepts_exact_current_epoch_on_held_out_capture(checkpoint_case):
    folder, scene, data, audit = checkpoint_case
    policy = render_trained.checkpoint_identity(folder,"edsr",scene,1500,data,audit)
    assert policy["schedule"]["job"]["validation_captures"] == ["held-out"]


@pytest.mark.parametrize("change", ["training_capture", "wrong_epoch", "changed_checkpoint", "changed_folds", "changed_sources"])
def test_rejects_wrong_fold_budget_or_checkpoint(checkpoint_case, change):
    folder, scene, data, audit = checkpoint_case
    epoch = 1500
    if change == "training_capture":
        scene["capture"] = "training"
    elif change == "wrong_epoch":
        epoch = 500
    elif change == "changed_checkpoint":
        (folder/"last.pt").write_bytes(b"selected-or-mutated")
    elif change == "changed_folds":
        data["folds_sha256"] = "different"
    else:
        policy = json.loads((folder/"protocol.json").read_text())
        policy["sources"] = {"source":"changed"}
        (folder/"protocol.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        render_trained.checkpoint_identity(folder,"edsr",scene,epoch,data,audit)


def test_comparison_crops_keep_identical_coordinates_and_nearest_pixels():
    from PIL import Image
    from benchmarks.super_resolution.compose_trained import crop_strip
    colors = [(10+i,20+i,30+i) for i in range(5)]
    images = [Image.new("RGB",(1920,1080),color) for color in colors]
    strip = crop_strip(images,[50,60,240,160])
    assert strip.size == (2424,360)
    assert [strip.getpixel((i*486,40)) for i in range(5)] == colors
    assert strip.getpixel((480,100)) == (148,163,184)
    with pytest.raises(ValueError,match="same valid"):
        crop_strip(images,[1900,0,240,160])


def test_comparison_title_uses_supported_font_characters():
    from benchmarks.super_resolution.compose_trained import chapter_title
    assert chapter_title({"title":"Code"},"detail") == "Code - Detail"
    assert chapter_title({"title":"Desktop"},"overview").isascii()


def test_comparison_filename_matches_requested_epoch():
    from benchmarks.super_resolution.compose_trained import video_filename
    assert video_filename(500) == "epoch-500-comparison.mp4"
    assert video_filename(1500) == "epoch-1500-comparison.mp4"
    assert video_filename(1500, pretrained=True) == "pretrained-vs-finetuned-epoch-1500.mp4"
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            video_filename(invalid)


def test_training_comparison_keeps_same_model_in_each_column():
    from PIL import Image
    from benchmarks.super_resolution.compose_trained import crop_strip, TRAINING_LABELS
    colors = [(10+i,20+i,30+i) for i in range(8)]
    images = [Image.new("RGB",(1920,1080),color) for color in colors]
    picture = crop_strip(images,[50,60,240,160],TRAINING_LABELS,4)
    assert picture.size == (1938,726)
    assert [picture.getpixel(((i%4)*486,40+(i//4)*366)) for i in range(8)] == colors
    assert picture.getpixel((480,100)) == picture.getpixel((100,360)) == (148,163,184)
    assert TRAINING_LABELS[1] == "EDSR - Pretrained" and TRAINING_LABELS[5] == "EDSR - Fine-tuned"
    with pytest.raises(ValueError):
        crop_strip(images,[50,60,240,160],TRAINING_LABELS,0)


def test_gallery_refresh_removes_embedded_title_adds_lines_and_preserves_pixels(tmp_path):
    from PIL import Image
    from benchmarks.super_resolution.compose_trained import refresh_gallery
    original = Image.new("RGB",(64,136),"white")
    colors = [(10+i,20+i,30+i) for i in range(8)]
    for i,color in enumerate(colors):
        original.paste(Image.new("RGB",(16,48),color),((i%4)*16,40+(i//4)*48))
    path = tmp_path/"detail.png"
    original.save(path)
    video = tmp_path/"comparison.mp4"
    video.write_bytes(b"unchanged-video")
    manifest = dict(epoch=1500,video=video.name,video_sha256=sha256(video),pretrained_outputs=[],
        pictures=[dict(file=path.name,sha256=sha256(path),xywh=[0,0,8,4],title="Terminal Detail")])
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    refresh_gallery(tmp_path)
    with Image.open(path) as updated:
        assert updated.size == (82,102)
        assert [updated.getpixel(((i%4)*22,(i//4)*54)) for i in range(8)] == colors
        assert updated.getpixel((16,10)) == updated.getpixel((0,48)) == (148,163,184)
    data = json.loads((tmp_path/"manifest.json").read_text())
    assert data["pictures"][0]["sha256"] == sha256(path)
    assert data["image_layout"]["embedded_titles"] is False
    assert sha256(video) == manifest["video_sha256"]
    page = (tmp_path/"index.html").read_text()
    assert page.count("<h2>Terminal Detail</h2>") == 1 and "?v=" in page
    before = path.read_bytes()
    refresh_gallery(tmp_path)
    assert path.read_bytes() == before  # Repeated updates do not crop more pixels.


def test_gallery_refresh_rejects_changed_images_before_writing(tmp_path):
    from benchmarks.super_resolution.compose_trained import refresh_gallery
    video = tmp_path/"comparison.mp4"
    video.write_bytes(b"video")
    picture = tmp_path/"detail.png"
    picture.write_bytes(b"changed")
    manifest = dict(epoch=1500,video=video.name,video_sha256=sha256(video),
        pictures=[dict(file=picture.name,sha256="wrong",xywh=[0,0,8,4],title="Detail")])
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="image changed"):
        refresh_gallery(tmp_path)
    assert picture.read_bytes() == b"changed" and not (tmp_path/"index.html").exists()


@pytest.mark.parametrize("asset", ["../outside.mp4", "/outside.mp4"])
def test_gallery_refresh_rejects_escaping_video_paths(tmp_path, asset):
    from benchmarks.super_resolution.gallery import refresh_gallery
    (tmp_path / "manifest.json").write_text(json.dumps(dict(video=asset)))
    with pytest.raises(ValueError, match="escapes"):
        refresh_gallery(tmp_path)


def test_failed_gallery_render_does_not_replace_existing_files(tmp_path, monkeypatch):
    from PIL import Image
    from benchmarks.super_resolution.gallery import refresh_gallery
    video = tmp_path / "comparison.mp4"
    video.write_bytes(b"video")
    pictures = []
    for index in range(2):
        path = tmp_path / f"detail-{index}.png"
        Image.new("RGB", (80, 88), "white").save(path)
        pictures.append(dict(file=path.name, sha256=sha256(path),
                             xywh=[0, 0, 8, 4], title="Detail"))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        epoch=1500, video=video.name, video_sha256=sha256(video), pictures=pictures
    )))
    (tmp_path / "index.html").write_text("Existing gallery")
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    save, calls = Image.Image.save, []
    def fail_second(image, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("Cannot write image")
        return save(image, *args, **kwargs)
    monkeypatch.setattr(Image.Image, "save", fail_second)
    with pytest.raises(OSError, match="Cannot write image"):
        refresh_gallery(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_gallery_escapes_titles_and_keeps_inference_imports_optional(tmp_path):
    import subprocess
    import sys
    from benchmarks.super_resolution.gallery import write_gallery
    write_gallery(tmp_path, dict(epoch=1500, video="comparison.mp4", pictures=[
        dict(file="detail.png", title='<Detail & "Text">', sha256="a"*64)
    ]))
    page = (tmp_path / "index.html").read_text()
    assert '<h2>&lt;Detail &amp; &quot;Text&quot;&gt;</h2>' in page
    assert "<Detail" not in page
    subprocess.run([sys.executable, "-c", (
        "import sys; import benchmarks.super_resolution.compose_trained; "
        "assert not {'torch', 'cv2', 'benchmarks.super_resolution.train_epochs'} "
        "& sys.modules.keys()"
    )], check=True)


@pytest.fixture
def pretrained_case(tmp_path):
    from benchmarks.super_resolution.compose_trained import MODELS, read
    lock = read(Path(render_trained.__file__).with_name("models.lock.json"))
    rows, states = [], {}
    for model in MODELS:
        path = tmp_path/model/"capture.mkv"
        path.parent.mkdir()
        path.write_bytes(model.encode())
        rows.append(dict(model=model,clip="capture",frames=10,fps="60",output=f"{model}/capture.mkv",
                         sha256=sha256(path),lossless_roundtrip_verified=True))
        states[model] = dict(state="completed",checkpoint_sha256=lock[model]["sha256"],source_sha=lock[model]["revision"])
    (tmp_path/"COMPLETED.json").write_text(json.dumps(dict(videos=3,status=states)))
    (tmp_path/"environment.json").write_text(json.dumps(dict(manifest_sha256="source-identity",precision="fp32")))
    (tmp_path/"clips.json").write_text(json.dumps(rows))
    data = dict(source_manifest_sha256="source-identity",fps=60,
                scenes=[dict(id="scene",capture="capture",start_frame=2,frames=6)])
    return tmp_path, data


def test_pretrained_outputs_match_official_weights_source_and_frame_range(pretrained_case):
    from benchmarks.super_resolution.compose_trained import pretrained_outputs, MODELS
    root, data = pretrained_case
    paths, evidence = pretrained_outputs(root,data)
    assert paths["scene"] == [root/model/"capture.mkv" for model in MODELS]
    assert len(evidence) == 3 and all(r["start_frame"] == 2 and r["excerpt_frames"] == 6 for r in evidence)


@pytest.mark.parametrize("change",["source","weights","precision","failed","duplicate","missing","range","fps","video","escape"])
def test_pretrained_outputs_reject_mismatched_or_changed_results(pretrained_case,change):
    from benchmarks.super_resolution.compose_trained import pretrained_outputs
    root, data = pretrained_case
    rows = json.loads((root/"clips.json").read_text())
    done = json.loads((root/"COMPLETED.json").read_text())
    env = json.loads((root/"environment.json").read_text())
    if change == "source":
        env["manifest_sha256"] = "different"
    elif change == "weights":
        done["status"]["edsr"]["checkpoint_sha256"] = "fine-tuned-not-official"
    elif change == "precision":
        env["precision"] = "amp-fp16"
    elif change == "failed":
        (root/"FAILED.json").write_text("{}")
    elif change == "duplicate":
        rows.append(dict(rows[0]));done["videos"] += 1
    elif change == "missing":
        rows.pop();done["videos"] -= 1
    elif change == "range":
        data["scenes"][0]["frames"] = 9
    elif change == "fps":
        rows[0]["fps"] = "30"
    elif change == "video":
        (root/"edsr/capture.mkv").write_bytes(b"changed")
    else:
        rows[0]["output"] = "../outside.mkv"
    for filename, value in (("clips.json",rows),("COMPLETED.json",done),("environment.json",env)):
        (root/filename).write_text(json.dumps(value))
    with pytest.raises(ValueError):
        pretrained_outputs(root,data)


def test_exact_poster_frame_is_decoded_from_lossless_video(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from fractions import Fraction
    from benchmarks.super_resolution.compose_trained import frame_at
    from benchmarks.super_resolution.video_benchmark import LosslessWriter,verify_lossless
    path = tmp_path/"sequence.mkv"
    writer = LosslessWriter(path,16,16,Fraction(60))
    for value in (10,120,230):
        writer.write(np.full((16,16,3),value,dtype=np.uint8))
    writer.close()
    assert verify_lossless(path,3,writer.digest.hexdigest()) == 3
    assert np.array_equal(np.asarray(frame_at(path,1)),np.full((16,16,3),120,dtype=np.uint8))
    with pytest.raises(ValueError,match="not found"):
        frame_at(path,3)
