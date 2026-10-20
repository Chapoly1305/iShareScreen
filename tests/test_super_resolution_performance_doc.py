"""Keep the maintained performance table adapter-only and correctly labeled."""
from pathlib import Path


def test_performance_table_order_and_finetuned_availability():
    path = Path(__file__).resolve().parents[1]/"benchmarks/super_resolution/PERFORMANCE.md"
    text = path.read_text()
    rows = [line.split("|")[1:-1] for line in text.splitlines() if line.startswith("|")]
    assert len(rows) == 11 and all(len(row) == 5 for row in rows)
    assert [cell.strip() for cell in rows[0]] == ["Model", "Scale", "Fine-tuned", "M4 adapter", "RTX 4090 adapter"]
    data = rows[2:]
    p95 = [float(row[-1].split("/")[1]) for row in data]
    assert p95 == sorted(p95)
    checked = {row[0].strip() for row in data if row[2].strip() == "✓"}
    assert checked == {"EDSR-baseline", "IMDN", "SAFMN"}
    assert "official pretrained" in text and "not\nthat it was used" in text
