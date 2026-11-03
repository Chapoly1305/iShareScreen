"""Data identity, source isolation and literal-label checks for the text corpus."""

import hashlib

import pytest

from benchmarks.super_resolution.text_corpus.catalog import document_family, split_for
from benchmarks.super_resolution.text_corpus.render import token_markup
from benchmarks.super_resolution.text_corpus.degrade import bitrate


def test_protocol_versions_share_a_source_family():
    titles = [
        "Matter-1.0-Core-Specification.pdf",
        "Matter-1.4-Application-Cluster-Specification.pdf",
        "23-27349-009_Matter-1.5-Core-Specification",
    ]
    families = [document_family(t, "", "unused") for t in titles]
    assert families == ["spec:matter"] * 3
    assert len({split_for(f) for f in families}) == 1
    assert document_family("A paper", "10.1234/ABCD", "key") == "doi:10.1234/abcd"


def test_source_code_is_escaped_without_dropping_punctuation():
    source = (
        'if (x < 10 && path == "a/b.json"):\n\treturn 0x01; <script>alert(1)</script>'
    )
    markup = token_markup(source)
    assert "<script>" not in markup and "&lt;script&gt;" in markup
    from html.parser import HTMLParser

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.fragments = []

        def handle_data(self, data):
            self.fragments.append(data)

    parsed = Parser()
    parsed.feed(markup)
    assert "".join(parsed.fragments) == source


def test_bitrate_mixture_and_split_are_content_deterministic():
    rates = [bitrate({"id": f"view-{i}"}) for i in range(10000)]
    assert set(rates) == {128000, 256000, 600000, 1000000, 2000000}
    assert rates.count(600000) > rates.count(128000)
    assert [split_for("document-a")] * 10 == [
        split_for("document-a") for _ in range(10)
    ]
    assert set(split_for(f"document-{i}") for i in range(100)) == {
        "train",
        "validation",
        "test",
    }


def test_scratch_construction_never_loads_checkpoint_and_reproduces_seed(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from benchmarks.super_resolution.text_corpus import scratch
    from isharescreen.enhancement.models import sha256

    source = tmp_path / "source"
    source.mkdir()
    (source / "models.py").write_text("pinned architecture fixture")
    monkeypatch.setattr(
        scratch.candidates,
        "specification",
        lambda key: dict(
            revision="pinned",
            files={
                "source/models.py": dict(sha256=sha256(source / "models.py")),
                "weights.pth": dict(sha256="unused pretrained checkpoint"),
            },
        ),
    )
    monkeypatch.setattr(scratch.candidates, "asset_home", lambda key, home: tmp_path)
    monkeypatch.setattr(
        scratch.loader,
        "build",
        lambda path: torch.nn.Sequential(
            torch.nn.Conv2d(3, 12, 1), torch.nn.PixelShuffle(2)
        ),
    )

    def forbid(*args, **kwargs):
        raise AssertionError("Pretrained checkpoint was read")

    monkeypatch.setattr(torch, "load", forbid)
    a, identity = scratch.build(tmp_path, 682)
    b, _ = scratch.build(tmp_path, 682)
    c, _ = scratch.build(tmp_path, 683)
    assert (
        identity["kind"] == "from_scratch" and not identity["pretrained_weights_loaded"]
    )
    assert all(
        torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()
    )
    assert any(
        not torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict()
    )


def test_text_crop_keeps_x2_reference_alignment_and_loss_is_differentiable():
    import numpy as np

    torch = pytest.importorskip("torch")
    from benchmarks.super_resolution.text_corpus.train import crop_batch, objective

    low = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(1, 16, 16, 3)
    high = low.repeat(2, axis=1).repeat(2, axis=2)
    arrays = dict(
        lr=low,
        hr=high,
        boxes=np.array([[8, 8, 24, 24, 3]], np.uint16),
        offsets=np.array([0, 1]),
    )
    x, y, mask = crop_batch(arrays, [0], np.random.default_rng(682), patch=8)
    assert torch.equal(x.repeat_interleave(2, 2).repeat_interleave(2, 3), y)
    assert mask.shape == (1, 1, 16, 16) and mask.sum() > 0
    pred = y.clone().requires_grad_()
    zero, _ = objective(pred, y, mask)
    assert zero.item() == 0
    loss, _ = objective(pred + 0.1, y, mask)
    loss.backward()
    assert loss > 0 and pred.grad is not None and torch.isfinite(pred.grad).all()
