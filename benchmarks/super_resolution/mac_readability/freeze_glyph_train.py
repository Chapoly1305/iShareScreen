"""Train-only Mac glyph pages using character frequencies from frozen CodeSCAN train.

These randomized glyph sequences are reconstruction training material, not
natural-language evaluation. No development/test text supplies a character,
frequency, line or document. All generated document families are train-only.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.project / "datasets/train/codescan/manifest.json"
    freeze = json.loads(
        (args.project / "datasets/sources/codescan/split-v1/FROZEN.json").read_text()
    )
    assert (
        sha(manifest) == freeze["files_sha256"]["datasets/train/codescan/manifest.json"]
    )
    rows = json.loads(manifest.read_text())["rows"]
    counter = Counter()
    for row in rows:
        source = manifest.parent / row["source_text"]
        assert sha(source) == row["files"]["source.txt"]
        counter.update(c for c in source.read_text() if "\u4e00" <= c <= "\u9fff")
    vocabulary = sorted(
        (c for c, n in counter.items() if n >= 4), key=lambda c: (-counter[c], c)
    )[:2048]
    weights = [counter[c] ** 0.5 for c in vocabulary]
    assert len(vocabulary) >= 1000
    rng = random.Random(20261103)
    batches = []
    identities = []
    for batch in range(4):
        states = []
        for document in range(16):
            identifier = f"train-glyph-{batch * 16 + document:03d}"
            lines = []
            for line in range(40):
                glyphs = "".join(rng.choices(vocabulary, weights=weights, k=36))
                if line % 5 == 0:
                    glyphs = (
                        glyphs[:24]
                        + f" {rng.randrange(100):02d}.{rng.randrange(100):02d} [{rng.randrange(1000):03d}]"
                    )
                lines.append(glyphs)
            identity = hashlib.sha256("\n".join(lines).encode()).hexdigest()
            identities.append(identity)
            for points in [6, 9] if document % 2 == 0 else [7, 8]:
                states.append(
                    dict(
                        id=identifier + f"-{points * 2}px",
                        group=identifier,
                        category="chinese",
                        state_id=len(states) + 1,
                        font="PingFang SC",
                        font_points=points,
                        dark=bool(document % 3 == 0),
                        lines=lines,
                        origin=dict(
                            kind="train-only randomized glyph sequences",
                            content_sha256=identity,
                        ),
                    )
                )
        result = dict(
            schema="mac_glyph_train_definitions_v1",
            complete=True,
            role="train_only_never_validation_or_test",
            generator_sha256=sha(Path(__file__)),
            source_manifest_sha256=sha(manifest),
            seed=20261103,
            document_groups=16,
            expected_images=32,
            states=states,
            vocabulary=[dict(character=c, train_count=counter[c]) for c in vocabulary],
            sampling="Square-root of train-only character frequency; top2048 chars appearing>=4 times; random36-character lines plus digits",
            semantic_claim=False,
            validation_or_test_used=False,
        )
        path = args.output / f"batch-{batch:02d}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        batches.append(dict(path=path.name, sha256=sha(path)))
    assert len(set(identities)) == 64
    (args.output / "FROZEN.json").write_text(
        json.dumps(
            dict(
                schema="mac_glyph_train_freeze_v1",
                complete=True,
                role="train_only",
                source_manifest_sha256=sha(manifest),
                batches=batches,
                documents=64,
                images=128,
                frozen_before_capture=True,
                validation_or_test_used=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "Frozen128 train-only native glyph variants from64 distinct random document families"
    )


if __name__ == "__main__":
    main()
