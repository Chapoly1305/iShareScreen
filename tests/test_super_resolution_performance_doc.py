"""The retired adapter table is replaced by the canonical test-result table."""

import csv
from pathlib import Path


def test_performance_table_order_and_finetuned_availability():
    root = Path(__file__).resolve().parents[1]
    text = (root / "results/README.md").read_text()
    section = text.split("## Primary results: independent test set\n", 1)[1].split(
        "\n## ", 1
    )[0]
    rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in section.splitlines()
        if line.startswith("|")
    ]
    with (root / "results/results.csv").open(newline="") as stream:
        expected = list(csv.reader(stream))
    # Skip the Markdown alignment row; retain all columns and published values.
    assert [rows[0], *rows[2:]] == expected
    assert {row[0] for row in rows[2:]} == {
        "EDSR",
        "IMDN",
        "SAFMN",
        "QuickSRNet",
        "PlainUSR",
        "RT4KSR",
        "EfRLFN",
    }
    quality = [float(row[3]) for row in rows[2:]]
    assert quality == sorted(quality, reverse=True)
    assert "excluding decoding" in text and "Quality uses FP32" in text
