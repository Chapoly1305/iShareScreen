"""Freeze source-known development documents before new candidate scoring.

Code uses four repository-disjoint entries of frozen CodeSCAN validation.
Other documents are project-authored. Font variants stay in their document
group. No source here may be used for training or final-test claims.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import textwrap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ENGLISH = [
    (
        "measurement",
        """A measurement needs a defined starting point and a defined ending point. A network packet arriving at the receiver is different from a newly drawn letter becoming visible. Record these events separately so that a fast decoder cannot hide a long queue.
The median describes a typical observation. The ninety-fifth percentile describes a slow observation near the tail. Neither statistic accounts for an update that never arrived. Keep the missing count beside the timing table, including incomplete connections and lost changes.
An average bandwidth limit permits bursts over shorter intervals. A short burst can fill a queue even when the long-term average is small. Specify whether the limit covers encoded payload, transport headers, acknowledgments, or the complete connection.
A reference image must show the same source state as the reconstruction. Comparing a newer reference against an older decoded frame confounds compression errors with stale content. A frame marker can identify the state, but it does not prove that every tile belongs to an atomic frame.
The exact text matters. The values 0.01 and 0.10 are not interchangeable. A missing minus sign, an altered decimal point, or a changed variable name can reverse the meaning of an otherwise attractive image.
Save the configuration and the original observations. A plot is useful for inspection, but the underlying records are needed to reproduce its aggregate. Report the device, precision, warmup, and sample count together.""",
    ),
    (
        "software",
        """A cache entry belongs to a particular content revision. When the source changes, invalidate pending work for the earlier revision. The previous result must not replace current text merely because a slow computation finishes later.
An asynchronous worker may reduce waiting in the display thread, but it still consumes compute and memory bandwidth. Measure the first update while the worker is busy, not only after all background tasks have stopped.
Timeouts and cancellation have different meanings. A timeout says that the caller stopped waiting. Cancellation says that the requested work will not continue. A system should document whether a timed-out operation can still produce a later result.
Keep immutable inputs separate from generated outputs. A new experiment should record its own configuration, log, and checkpoint. Reusing an old directory can make an apparently successful run depend on files left by another experiment.
A reproducible random sequence requires more than writing down a seed. Record the ordering of examples, the preprocessing version, and the checkpoint used for initialization. Resume behavior should preserve the optimizer and scheduler when the experiment requires continuity.
Tests should exercise observable behavior. A stream must preserve byte order, a crop must refer to the intended image region, and a stale result must never be drawn over a newer document. Passing a syntax check alone establishes none of these properties.""",
    ),
    (
        "statistics",
        """A comparison begins with a fixed question. Choose the primary outcome before examining a new candidate. Additional measurements can explain the result, but changing the primary outcome afterward makes the conclusion difficult to interpret.
Repeated frames from the same document are correlated observations. Treating them as independent can make an uncertainty interval misleadingly narrow. Resample whole documents when estimating variation across documents.
A relative reduction depends on the comparator. Moving from an error rate of 0.20 to 0.15 is a twenty-five percent reduction. Moving from 0.04 to 0.03 has the same relative reduction but a different absolute difference.
An aggregate can improve while an individual category gets worse. Show the categories and the cases that were previously correct. Numerical identifiers, punctuation, and non-Latin characters may react differently to the same enhancement.
The clean reference is also useful as a diagnostic input to the recognizer. If the recognizer fails on the original, its errors place a practical limit on what that automatic metric can establish. Do not silently remove those difficult examples.
Reserve the final test for a selected configuration. Development validation can guide iteration; the final test estimates behavior after those choices have been fixed. Repeatedly consulting the final test turns it into another development set.""",
    ),
    (
        "reading",
        """A person reading code often pauses after scrolling. The first response helps orientation, while a later clear frame supports careful inspection. These two moments need separate measurements because they serve different tasks.
Smooth animation is not the only objective of a remote desktop. A spreadsheet may remain still while the reader checks a total. An editor may show only a few changed characters. The transport should reflect these patterns of use.
Preserve the shape of letters without inventing strokes. A sharpened edge may look convincing while changing an uppercase I into a lowercase l. Visual preference and textual correctness should be inspected together.
Small colored text is sensitive to the way color samples are shared. Brightness detail and color detail follow different paths through a video codec. Compare the actual decoded output at the intended display scale.
Use a native-size crop when inspecting a disputed character. Enlarging every image can emphasize differences that are difficult to see during normal use. Keep the original view available beside any enlarged diagnostic.
The receiver can choose when to run expensive refinement. A newly changed region should appear promptly; a stable region can receive additional processing. The decision must be based on the current content rather than a timer detached from document changes.""",
    ),
]

CHINESE = [
    (
        "measurement",
        [
            "延迟测量首先需要明确起点和终点，收到数据包并不等于文字已经显示在屏幕上。",
            "应分别记录源内容变化、接收完成、解码完成和显示提交的时间，避免把不同环节混为一谈。",
            "中位数反映典型情况，百分之九十五分位数反映较慢情况，未能到达的更新还需要单独统计。",
            "长时间平均码率较低，并不表示每一秒都没有突发流量，缓冲区仍然可能迅速积满。",
            "参考图像必须与重建图像属于同一内容状态，否则画面过期会被错误地计入压缩失真。",
            "状态标记可以帮助识别内容，但不能保证多个独立图块在同一时刻完成更新。",
            "数字零点零一与零点一零具有不同含义，小数点、负号和括号都不能随意丢失。",
            "实验结果应保存配置、原始记录和设备信息，使其他人能够复现统计和检查异常。",
        ],
    ),
    (
        "software",
        [
            "缓存结果必须绑定内容版本，当源画面发生变化时，旧版本的增强结果应立即失效。",
            "后台计算完成得晚，并不意味着它可以覆盖已经显示的新文字，显示顺序必须受到保护。",
            "异步执行能够减少等待，但仍然消耗计算资源和内存带宽，需要测量并发工作时的表现。",
            "等待超时与取消任务不是同一件事，调用方停止等待之后，后台任务可能仍在继续运行。",
            "原始数据、冻结实验和新产生的结果应分别保存，避免旧文件影响新实验的结论。",
            "随机种子之外还需要记录样本顺序、预处理版本和初始化权重，才能可靠地恢复训练。",
            "测试应检查可观察的行为，例如字节顺序、裁剪位置以及旧结果是否会覆盖新内容。",
            "语法检查通过只说明程序可以被解析，不能证明数据没有混入验证集或者测试集。",
        ],
    ),
    (
        "statistics",
        [
            "比较新方法之前应确定主要指标，不应看到结果以后才选择最有利的统计方式。",
            "来自同一文档的连续帧并非独立样本，估计不确定性时应以文档组为单位重新抽样。",
            "错误率从百分之二十降到百分之十五，相对减少四分之一，绝对差异是五个百分点。",
            "总体平均值改善时，某些类别仍可能退步，因此需要同时检查类别结果和新增错误。",
            "代码标识符、数字、标点以及中文字符，可能对同一种增强处理产生不同反应。",
            "干净原图也应经过同样的识别流程，原图上的识别错误能够揭示自动指标本身的限制。",
            "验证集用于开发和选择配置，最终测试集应在选择完成后使用，不能反复参与调参。",
            "保存失败案例与保存成功案例同样必要，缺失的更新不能从统计表中悄悄消失。",
        ],
    ),
    (
        "reading",
        [
            "阅读代码时，用户常在滚动后停下来检查，首次响应和静止后的清晰度服务于不同需求。",
            "低带宽远程桌面不一定追求动画流畅，表格中的总计或者编辑器中的几行代码更需要准确。",
            "锐化后的边缘可能看起来清楚，却把一个字母变成另一个字母，因此视觉效果不能代替正确性。",
            "小字号彩色文字同时受到亮度与色度采样影响，应在实际显示比例下检查解码结果。",
            "检查有争议的字符时应保留原始尺寸，放大图可以辅助分析，但不代表正常阅读体验。",
            "接收端可以先显示新内容，再对稳定区域进行增强，调度依据应当是当前内容的真实变化。",
            "新的计算结果到达时还需要检查版本号，不能因为队列顺序或者时间误差重新显示过期画面。",
            "可靠的改进应兼顾字符正确率、响应时间、实际流量和退化案例，而不是只报告一个平均分数。",
        ],
    ),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.project
    frozen = json.loads(
        (root / "datasets/sources/codescan/split-v1/FROZEN.json").read_text()
    )
    manifests = {}
    for split in ["train", "validation"]:
        rel = f"datasets/{split}/codescan/manifest.json"
        assert sha(root / rel) == frozen["files_sha256"][rel]
        manifests[split] = json.loads((root / rel).read_text())
    train = manifests["train"]["rows"]
    validation = manifests["validation"]["rows"]
    groups = []
    selected = []
    for language in ["Python", "Rust", "JavaScript", "Go"]:
        for row in sorted(
            (r for r in validation if r["language"] == language), key=lambda r: r["id"]
        ):
            path = root / "datasets/validation/codescan" / row["source_text"]
            assert sha(path) == row["files"]["source.txt"]
            lines = [
                s.rstrip()
                for s in path.read_text().expandtabs(4).splitlines()
                if s.strip()
            ]
            if len(lines) < 45:
                continue
            # Show a documented source excerpt, not invisible overflowing text.
            lines = [s[:74] for s in lines[:48]]
            groups.append(
                dict(
                    id="code-" + language.lower(),
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
                        excerpt="first 48 nonempty lines; expand tabs to 4 spaces; first 74 code points per line",
                    ),
                )
            )
            selected.append(row)
            break
        else:
            raise RuntimeError("No eligible code document for " + language)
    assert len({r["repository"] for r in selected}) == 4
    for key in ["repository", "group_id", "normalized_source_sha256"]:
        assert not {r[key] for r in selected} & {r[key] for r in train}, key
    rng = random.Random(20261102)
    for kind in ["inventory", "weather", "transit", "ledger"]:
        lines = []
        for column in range(2):
            headings = {
                "inventory": "Part ID   Stock   Reserved   Unit price   Reorder",
                "weather": "Station   Temp C   Rain mm   Wind m/s   Valid",
                "transit": "Route   Planned   Actual   Delay min   Platform",
                "ledger": "Invoice   Debit ($)   Credit ($)   Balance ($)",
            }
            lines.append(headings[kind])
            for i in range(23):
                number = i + column * 23 + 1
                if kind == "inventory":
                    line = f"P-{number:04d}   {rng.randrange(1000):4d}   {rng.randrange(99):4d}     {rng.randrange(1, 5000) / 100:7.2f}       {rng.choice(['yes', 'no'])}"
                elif kind == "weather":
                    line = f"S-{number:04d}   {rng.randrange(-150, 400) / 10:6.1f}   {rng.randrange(500) / 10:6.1f}   {rng.randrange(250) / 10:7.1f}   {rng.choice(['true', 'false'])}"
                elif kind == "transit":
                    line = f"R-{number:03d}    {number % 24:02d}:{number * 7 % 60:02d}     {number % 24:02d}:{number * 11 % 60:02d}     {rng.randrange(-9, 40):3d}         {rng.randrange(1, 12):02d}"
                else:
                    a, b = rng.randrange(1000000), rng.randrange(1000000)
                    line = f"I-{number:04d}   {a / 100:9,.2f}   {b / 100:10,.2f}   {(a - b) / 100:11,.2f}"
                lines.append(line)
        groups.append(
            dict(
                id="table-" + kind,
                category="table",
                font="Menlo",
                dark=False,
                lines=lines,
                origin=dict(kind="project-authored numeric document", seed=20261102),
            )
        )
    for name, paragraphs in ENGLISH:
        lines = []
        for paragraph in paragraphs.splitlines():
            lines.extend(textwrap.wrap(paragraph, 72))
        assert len(lines) <= 48
        groups.append(
            dict(
                id="english-" + name,
                category="english",
                font="Helvetica",
                dark=False,
                lines=lines,
                origin=dict(kind="project-authored prose"),
            )
        )
    for name, paragraphs in CHINESE:
        lines = []
        for paragraph in paragraphs:
            lines.extend(textwrap.wrap(paragraph, 38))
        groups.append(
            dict(
                id="chinese-" + name,
                category="chinese",
                font="PingFang SC",
                dark=False,
                lines=lines,
                origin=dict(kind="project-authored Chinese prose"),
            )
        )
    assert len(groups) == 16
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
    normalized = [
        hashlib.sha256("".join(g["lines"]).encode()).hexdigest() for g in groups
    ]
    assert len(set(normalized)) == 16
    result = dict(
        schema="mac_dense_development_v1",
        role="development_validation_only_never_train_or_final_test",
        generator_sha256=sha(Path(__file__)),
        groups=groups,
        states=states,
        expected_images=32,
        document_groups=16,
        clean_ocr_floor="Retain all clean-reference OCR failures; do not remove hard lines after scoring",
        grouping="All font/theme/render/codec variants stay with their source document family",
        train_isolation=dict(
            code_repository_overlap=0,
            code_group_overlap=0,
            code_normalized_text_hash_overlap=0,
            train_manifest_sha256=sha(root / "datasets/train/codescan/manifest.json"),
            validation_manifest_sha256=sha(
                root / "datasets/validation/codescan/manifest.json"
            ),
        ),
        historical_test="No test manifest or sample read during preparation; preserve frozen final test",
        layout=dict(
            reference_pixels=[1920, 1080],
            columns=2,
            rows_per_column=24,
            font_pixels=[14, 18],
            crop_truth="Native font-measured line boxes",
        ),
        candidate_selection_started=False,
        complete=True,
    )
    (args.output / "DEFINITIONS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Frozen",
        len(states),
        "font variants across",
        len(groups),
        "development document groups",
    )


if __name__ == "__main__":
    main()
