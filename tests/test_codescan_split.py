import importlib.util
from pathlib import Path
from collections import Counter


spec = importlib.util.spec_from_file_location(
    "codescan_split", Path(__file__).parents[1] / "datasets/split_codescan.py"
)
splitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)


def sample(index, repository=None, language="Python"):
    return dict(
        sample=f"{language}/owner~repo{index}/file.py",
        language=language,
        repository=repository or f"owner/repo{index}",
        directory_sha256=f"directory-{index}",
        normalized_source_sha256=f"source-{index}",
        files={"img.png": f"file-{index}"},
        pixels={"img.png": f"pixels-{index}", "bw.png": f"bw-{index}"},
    )


def test_repository_is_one_group_across_languages():
    rows = [
        sample(0, "owner/project"),
        sample(1, "owner/project", "JavaScript"),
        sample(2),
    ]
    uf, groups, duplicates, _ = splitter.clusters(rows)
    assert uf.find(0) == uf.find(1) != uf.find(2)
    assert len(groups) == 2 and not duplicates


def test_shared_file_content_merges_different_repositories():
    rows = [sample(0), sample(1), sample(2)]
    rows[1]["files"]["source.txt"] = rows[0]["files"]["img.png"]
    uf, _, _, _ = splitter.clusters(rows)
    assert uf.find(0) == uf.find(1) != uf.find(2)


def test_source_normalization_hash_prevents_code_cross_split():
    rows = [sample(0), sample(1)]
    rows[1]["normalized_source_sha256"] = rows[0]["normalized_source_sha256"]
    uf, _, _, _ = splitter.clusters(rows)
    assert uf.find(0) == uf.find(1)


def test_identical_color_pixels_are_only_admitted_once():
    rows = [sample(0), sample(1)]
    rows[1]["pixels"]["img.png"] = rows[0]["pixels"]["img.png"]
    uf, groups, duplicates, _ = splitter.clusters(rows)
    assert uf.find(0) == uf.find(1)
    assert sum(map(len, groups.values())) == 1
    assert duplicates[0]["canonical_sample"] == rows[0]["sample"]


def test_assignment_keeps_components_and_balances_languages():
    rows = []
    groups = {}
    for language in ("Python", "JavaScript"):
        for repository in range(100):
            indices = []
            for _ in range(5):
                indices.append(len(rows))
                rows.append(sample(len(rows), language=language))
            groups[indices[0]] = indices
    assignment = splitter.assign(rows, groups, 682)
    assert assignment == splitter.assign(rows, groups, 682)
    for language in ("Python", "JavaScript"):
        counts = Counter(
            assignment[group]
            for group, members in groups.items()
            for i in members
            if rows[i]["language"] == language
        )
        assert abs(counts["train"] - 400) <= 5
        assert abs(counts["validation"] - 50) <= 5
        assert abs(counts["test"] - 50) <= 5
