"""Freeze new document groups before fixed-candidate confirmation scoring.

Only CodeSCAN train/validation manifests are inspected. Historical final test
is never opened. Shared rendering templates are explicitly not new domains.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import textwrap


ENGLISH = {
    "museum": [
        "The museum catalog assigns one accession number to each object. A wooden box may contain several pieces, but the container number and the item numbers serve different purposes. Labels must remain readable without removing the object from its support.",
        "Before opening a crate, photograph its exterior and record the condition of every seal. Place the packing list beside the photographs. If a listed part is absent, record the absence rather than borrowing a similar part from another collection.",
        "Storage locations use the form B-04-17: building, cabinet, and shelf. The entry B-04-71 refers to a different shelf. A location change requires a dated record even when the object moves only a short distance within the same room.",
        "Light-sensitive material is displayed for limited periods. The exhibition record includes the opening date, the closing date, and the measured illumination. A replacement label should preserve the original description and the accession number.",
        "An inventory can contain uncertain dates and incomplete descriptions. Mark uncertainty explicitly. A missing date is not equivalent to the year zero, and a question mark in an attribution must not disappear when the catalog is copied.",
    ],
    "garden": [
        "Each planting bed has a permanent identifier and a seasonal planting plan. The plan records the variety, sowing date, spacing, and expected harvest window. An empty bed remains in the inventory so that its earlier treatment can be traced.",
        "Seed packets from different lots are stored separately. Germination records use both the variety name and the lot number. Ten seedlings from a single tray do not represent ten independent seed lots, even when their heights differ.",
        "Irrigation begins only after checking the soil and the weather record. The controller displays duration in minutes, while the meter reports volume in liters. Changing the duration does not establish the amount of water delivered to each bed.",
        "The north plot receives morning shade from an adjacent wall. Its temperature readings should not be substituted for those of the open south plot. Keep the sensor position beside every reading when preparing the seasonal record.",
        "At harvest, record damaged and unripe produce as well as the usable yield. Removing unsuccessful plants from the denominator would inflate the result. Preserve the original counts when a later inspection changes the classification.",
    ],
    "observatory": [
        "The observing log records the target, instrument, exposure, and clock standard. An observation beginning at 23:58 may end on the following date. The date attached to a file name should not silently replace the interval in the log.",
        "A detector frame includes both the target and unwanted contributions. Dark frames and flat fields have distinct purposes and should be identified separately. A correction file belongs to the instrument settings under which it was acquired.",
        "Cloud cover can change during a sequence. Keep rejected frames and the reason for rejection in the observing record. The remaining frames alone cannot describe the conditions during the entire scheduled observing period.",
        "Coordinates require an explicit reference system and epoch. Two numerical pairs can refer to different positions when their conventions differ. Do not remove a leading minus sign or round a small offset before applying the conversion.",
        "A scheduled target may remain below the horizon or be obscured by a building. Record the missed observation without inventing a substitute measurement. Later analysis can distinguish instrument failure from unavailable observing conditions.",
    ],
    "archive": [
        "An archive stores both the original record and a description of its provenance. A transcription makes the text easier to search, but it does not replace the source image. Corrections to the transcription should have a visible revision history.",
        "Folder titles may use abbreviations that were meaningful to the original author. Expand an abbreviation only when its meaning is established. If two interpretations remain possible, preserve the original form and describe the uncertainty.",
        "Page numbers written on a document can differ from the order of scanned images. Record both identifiers. A blank page may establish the boundary between sections and should not be discarded merely because it contains no searchable words.",
        "A date such as 03/04/25 depends on the notation used by its author. Store the literal date alongside any normalized interpretation. The normalized value should include the assumption about day, month, and century.",
        "Access copies can have a smaller size than preservation copies. Document which version was supplied to a reader. An image that is adequate for locating a paragraph may be inadequate for distinguishing a faint decimal point in a handwritten total.",
    ],
}

CHINESE = {
    "library": [
        "图书入库时应同时记录书名、作者、版本与馆藏编号。同名书籍可能采用不同的排版和页码，不能只凭封面颜色判断是否属于同一版本。",
        "借阅记录中的归还日期可以为空，但空白并不表示已经归还。续借操作应保留原借出日期，并另外记录新的期限及办理时间。",
        "修复人员在更换书脊标签之前，应拍摄原标签并核对编号。标签上的数字零与英文字母容易混淆，需要结合登记簿逐项检查。",
        "合订本包含多个独立刊期，目录必须指出每一期在册内的位置。缺少一页和缺少整期是不同情况，备注中应准确描述。",
        "书架调整后应更新位置索引，并保留调整前后的对应关系。仅仅移动纸质标牌而没有更新目录，会使读者前往错误的位置。",
        "盘点时应报告实存数量、借出数量与待核实数量。尚未找到的书籍需要继续调查，不能直接记为损毁，也不能从清单中删除。",
    ],
    "railway": [
        "车站时刻表分别列出计划到达时间与实际到达时间。午夜前后的班次跨越日期，比较时间时必须同时查看日期，不能只比较小时。",
        "旅客换乘需要预留步行与检票时间。两趟列车之间相差十分钟，并不意味着所有站台之间都能在这一时间内完成换乘。",
        "临时调整站台时，广播、显示屏与工作人员的通知应指向同一个班次。相邻班次的车次编号接近，不能只核对目的地名称。",
        "设备检修记录需要写明开始时间、结束时间及影响范围。检修结束不等于所有设备立即恢复，恢复状态应另行确认并记录。",
        "退票凭证中的金额包含币种与小数位。手续费、原票价与实际退还金额应分列显示，负号和小数点的遗漏都会改变含义。",
        "统计晚点情况时，应保留取消班次和未获得到达时间的班次。仅计算已到达列车的平均值，无法完整反映当天的运行情况。",
    ],
    "kitchen": [
        "配方记录同时使用质量与体积时，必须明确每一项的单位。一百克与一百毫升不能直接互换，不同原料的密度也可能不同。",
        "电子秤在放置容器之后需要重新去皮。更换容器或移动设备后应检查零点，称量结果应连同原料批次一起登记。",
        "烘烤时间取决于烤箱状态、模具尺寸和原料温度。相同的设定温度并不保证食物中心达到相同温度，应保留实际观察记录。",
        "批量制作时先计算总用量，再根据容器容量分批处理。每批记录实际加入的数量，避免将整批配方误当成单份配方重复使用。",
        "储存标签应注明制作日期、开封日期和保存条件。一个日期不能同时代表这三项信息，字迹模糊的标签需要重新核实。",
        "品尝记录应保留成功与失败的批次。修改配方时标明具体变化，而不是只给新版本重新命名，否则难以确定结果变化的原因。",
    ],
    "water": [
        "水样采集表记录地点、深度、时间和容器编号。同一地点在不同深度取得的样品应分别编号，不应因为外观相似而混合。",
        "现场仪表读数与实验室分析结果采用不同的测量步骤。汇总时应注明方法和单位，不能把两个来源的数值当作重复记录删除。",
        "样品运输过程中应记录温度和时间。容器完整只能说明没有明显破损，不能证明样品性质在整个运输期间保持不变。",
        "低于检出限的结果不等于含量为零。报告应保存原有标记和检出限数值，后续计算时再说明如何处理这类观测。",
        "仪器校准使用的标准溶液有独立批号和有效期。更换标准溶液后，应记录新的校准结果，而不是覆盖前一次的记录。",
        "年度比较需要考虑采样地点和采样季节的变化。如果某一年新增了较深的采样点，应把这一变化与水质变化分别说明。",
    ],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", text)


def shingles(text):
    value = normalize(text)
    return {value[i : i + 8] for i in range(max(0, len(value) - 7))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-fit", type=Path)
    args = parser.parse_args()
    root = args.project
    out = args.output
    assert not out.exists()
    freeze = json.loads(
        (root / "datasets/sources/codescan/split-v1/FROZEN.json").read_text()
    )
    rows = {}
    for split in ["train", "validation"]:
        rel = f"datasets/{split}/codescan/manifest.json"
        assert sha(root / rel) == freeze["files_sha256"][rel]
        rows[split] = json.loads((root / rel).read_text())["rows"]
    old_path = (
        root
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/DEFINITIONS.json"
    )
    old = json.loads(old_path.read_text())
    excluded = {r["repository"] for r in rows["train"]} | {
        g["origin"]["repository"] for g in old["groups"] if "repository" in g["origin"]
    }
    groups = []
    selected = []
    for language in ["Python", "Rust", "JavaScript", "Go"]:
        for row in sorted(
            (r for r in rows["validation"] if r["language"] == language),
            key=lambda r: r["id"],
        ):
            if row["repository"] in excluded:
                continue
            path = root / "datasets/validation/codescan" / row["source_text"]
            assert sha(path) == row["files"]["source.txt"]
            lines = [
                s.rstrip()
                for s in path.read_text().expandtabs(4).splitlines()
                if s.strip()
            ]
            if len(lines) < 48:
                continue
            lines = [s[:74] for s in lines[:48]]
            groups.append(
                dict(
                    id="confirm-code-" + language.lower(),
                    category="code",
                    dark=True,
                    font="Menlo",
                    lines=lines,
                    origin=dict(
                        kind="CodeSCAN frozen validation",
                        repository=row["repository"],
                        group_id=row["group_id"],
                        sample_id=row["id"],
                        source_sha256=row["files"]["source.txt"],
                        excerpt="first48nonemptylines,expandtabs4,first74codepoints",
                    ),
                )
            )
            selected.append(row)
            excluded.add(row["repository"])
            break
        else:
            raise RuntimeError("No eligible unseen repository for " + language)
    rng = random.Random(2026110316)
    for kind in ["energy", "assay", "shipping", "survey"]:
        headings = {
            "energy": "Meter ID   Start kWh   End kWh   Delta kWh",
            "assay": "Tube ID   Dose mg   Result mg/L   Recovery %",
            "shipping": "Parcel ID   Mass kg   Volume L   Charge EUR",
            "survey": "Point ID   East m   North m   Elevation m",
        }
        lines = []
        for col in range(2):
            lines.append(headings[kind])
            for i in range(23):
                n = col * 23 + i + 1
                if kind == "energy":
                    a = rng.randrange(100000, 900000)
                    b = a + rng.randrange(0, 10000)
                    line = f"E-{n:04d}   {a / 100:9.2f}   {b / 100:9.2f}   {(b - a) / 100:8.2f}"
                elif kind == "assay":
                    line = f"A-{n:04d}   {rng.randrange(1, 10000) / 1000:7.3f}   {rng.randrange(1, 100000) / 1000:10.3f}   {rng.randrange(8000, 10501) / 100:8.2f}"
                elif kind == "shipping":
                    line = f"X-{n:04d}   {rng.randrange(1, 50000) / 1000:8.3f}   {rng.randrange(1, 90000) / 100:8.2f}   {rng.randrange(100, 50000) / 100:9.2f}"
                else:
                    line = f"Q-{n:04d}   {rng.randrange(-99999, 100000) / 100:8.2f}   {rng.randrange(-99999, 100000) / 100:9.2f}   {rng.randrange(-5000, 90000) / 100:9.2f}"
                lines.append(line)
        groups.append(
            dict(
                id="confirm-table-" + kind,
                category="table",
                dark=False,
                font="Menlo",
                lines=lines,
                origin=dict(kind="project-authored numeric document", seed=2026110316),
            )
        )
    for category, documents, font, width in [
        ("english", ENGLISH, "Helvetica", 72),
        ("chinese", CHINESE, "PingFang SC", 38),
    ]:
        for name, paragraphs in documents.items():
            lines = [
                line
                for paragraph in paragraphs
                for line in textwrap.wrap(paragraph, width)
            ]
            assert len(lines) <= 48
            groups.append(
                dict(
                    id=f"confirm-{category}-{name}",
                    category=category,
                    font=font,
                    dark=False,
                    lines=lines,
                    origin=dict(kind="project-authored new-topic prose"),
                )
            )
    assert len(groups) == 16
    layout_fit = None
    if args.layout_fit:
        layout_fit = json.loads(args.layout_fit.read_text())
        for change in layout_fit["adjustments"]:
            group = next(g for g in groups if g["id"] == change["group"])
            assert group["lines"][change["line"]] == change["original"]
            assert (
                change["original"].startswith(change["retained"])
                and change["retained_width_points"] <= 440
            )
            group["lines"][change["line"]] = change["retained"]
            group["origin"]["native_width_excerpt_adjustment"] = change
    candidate_text = {g["id"]: "\n".join(g["lines"]) for g in groups}
    candidate_hash = {
        k: hashlib.sha256(normalize(v).encode()).hexdigest()
        for k, v in candidate_text.items()
    }
    assert len(set(candidate_hash.values())) == 16
    candidate_shingles = {k: shingles(v) for k, v in candidate_text.items()}
    nearest = {
        k: dict(containment=0.0, jaccard=0.0, source=None) for k in candidate_text
    }
    counts = {}
    evidence = {}

    def inspect(text, identity, role):
        norm = normalize(text)
        assert (
            hashlib.sha256(norm.encode()).hexdigest() not in candidate_hash.values()
        ), identity
        other = shingles(text)
        counts[role] = counts.get(role, 0) + 1
        for key, current in candidate_shingles.items():
            n = len(current & other)
            contain = n / max(1, min(len(current), len(other)))
            jac = n / max(1, len(current) + len(other) - n)
            if contain > nearest[key]["containment"]:
                nearest[key] = dict(
                    containment=contain, jaccard=jac, source=identity, role=role
                )
            assert contain < 0.65, (key, identity, contain)

    for group in old["groups"]:
        inspect("\n".join(group["lines"]), group["id"], "earlier_development")
    glyph = root / "datasets/sources/mac-glyph-train-v1"
    for path in sorted(glyph.glob("batch-*/reference/manifest.json")):
        evidence[str(path.relative_to(root))] = sha(path)
        seen = set()
        for state in json.loads(path.read_text())["states"]:
            if state["group"] in seen:
                continue
            seen.add(state["group"])
            inspect("\n".join(state["lines"]), state["group"], "native_glyph_train")
    assert counts.get("native_glyph_train") == 64, counts
    for i, row in enumerate(rows["train"]):
        path = root / "datasets/train/codescan" / row["source_text"]
        assert sha(path) == row["files"]["source.txt"]
        inspect(path.read_text(), row["id"], "codescan_train")
        if (i + 1) % 1000 == 0:
            print("Audited train documents", i + 1, flush=True)
    for i, group in enumerate(groups):
        for other in groups[:i]:
            a = candidate_shingles[group["id"]]
            b = candidate_shingles[other["id"]]
            assert len(a & b) / max(1, min(len(a), len(b))) < 0.65
    for key in ["repository", "group_id", "normalized_source_sha256"]:
        assert not {r[key] for r in selected} & {r[key] for r in rows["train"]}
    states = []
    for group in groups:
        for points in [7, 9]:
            states.append(
                dict(
                    group,
                    group=group["id"],
                    id=group["id"] + f"-{points * 2}px",
                    state_id=len(states) + 1,
                    font_points=points,
                )
            )
    plan = (
        root
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102/INDEPENDENT-CONFIRMATION-PLAN-V1.json"
    )
    result = dict(
        schema="mac_confirmation_definitions_v1",
        complete=True,
        role="independent_development_confirmation_never_train_or_final_test",
        generator_sha256=sha(Path(__file__)),
        plan_sha256=sha(plan),
        old_definitions_sha256=sha(old_path),
        groups=groups,
        states=states,
        document_groups=16,
        expected_images=32,
        normalized_content_sha256=candidate_hash,
        isolation=dict(
            counts=counts,
            nearest_by_whitespace_removed_character8gram_containment=nearest,
            rejection_rule="Reject identical whitespace-normalized document or8gramintersection/minsetsize>=.65; contentaudit only, before images/OCR",
            code_repository_overlap_with_train_or_earlier_dev=0,
            train_manifest_sha256=sha(root / "datasets/train/codescan/manifest.json"),
            glyph_manifests=evidence,
        ),
        limitations=[
            "Samefonts/layouttemplates; newcontentdoesnotestablishnewfontorrealdesktopgeneralization",
            "Project-authored numeric/prose documents are controlled evaluation, not sampled natural desktops",
        ],
        layout_fit=layout_fit,
        candidate_selection_started=False,
        test_used=False,
    )
    out.mkdir(parents=True)
    (out / "DEFINITIONS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Frozen16newdocuments32states",
        [(g["id"], len(g["lines"])) for g in groups],
        flush=True,
    )


if __name__ == "__main__":
    main()
