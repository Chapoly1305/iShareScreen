"""Build a local, source-linked gallery of stratified admitted training pairs."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import time

from benchmarks.super_resolution.text_corpus.render import read, stable_order
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def run(args):
    root = args.root.resolve()
    data = read(root / "hevc-v2/pairs.json")
    release = read(root / "RELEASE.json")
    sources = {s["id"]: s for s in read(root / "sources.json")["sources"]}
    selected = []
    for kind in ["paper", "specification", "code", "web", "table", "terminal"]:
        for split in ["train", "validation", "test"]:
            for bps in [128000, 256000, 600000, 1000000, 2000000]:
                choices = [
                    p
                    for p in data["pairs"]
                    if (p["kind"], p["split"], p["requested_bps"]) == (kind, split, bps)
                ]
                if choices:
                    selected.append(min(choices, key=lambda p: stable_order(p["id"])))
    page = [
        """<!doctype html><html lang="zh"><meta charset="utf-8"><title>文字训练集</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:0 24px;color:#172334;background:#f5f7fa}article{background:white;padding:20px;margin:25px 0;border-radius:12px}section{display:grid;grid-template-columns:1fr 1fr;gap:15px}img{width:100%;height:auto;border:1px solid #d9e0e7}pre{white-space:pre-wrap}select{font:inherit;padding:6px;margin:6px}small{color:#506078}table{border-collapse:collapse}td,th{border:1px solid #c8d2dc;padding:8px 20px}a{color:#1766a2}</style><h1>文字训练集：来源与低码率图片对</h1><p>新增 10,000 个 FHD 视图，保留 4,155 对旧视频帧。新增语料按源文档分组，训练 8,068，验证 752，测试 1,180。下方是按内容、集合和码率分层选择的样本，不是对所有图片的人工审核。</p><p><a href="RELEASE.json">发布清单</a> · <a href="dataset-views-v3.json">全部视图与来源分组</a> · <a href="hevc-v2/pairs.json">全部压缩配对与实际码率</a> · <a href="quality-audit-v3.json">全量审计</a> · <a href="legacy-video-pool.json">旧视频帧池</a></p><table><tr><th>内容</th><th>视图数</th></tr>"""
    ]
    for kind, count in release["counts"].items():
        page.append(f"<tr><td>{kind}</td><td>{count}</td></tr>")
    page += [
        "</table><p>低清为 960×540，经 HEVC 420 解码；高清为对应的 1920×1080 源渲染。请求码率与实际视频码率分别记录。编码器 VBV 告警作为退化元数据保留，不代表通过了带宽限制测试。</p>",
        '<label>内容 <select id="kind"><option value="">全部</option>'
        + "".join(f"<option>{v}</option>" for v in release["counts"])
        + "</select></label>",
        '<label>集合 <select id="split"><option value="">全部</option><option>train</option><option>validation</option><option>test</option></select></label>',
        '<label>请求 Kbps <select id="rate"><option value="">全部</option>'
        + "".join(f"<option>{v}</option>" for v in [128, 256, 600, 1000, 2000])
        + "</select></label>",
    ]
    for pair in selected:
        labels = read(root / pair["labels"])
        source = sources.get(labels.get("source"))
        title = source["title"] if source else "可控生成：数字、标点与路径"
        url = (
            source["path"]
            + ("#page=" + str(labels["page"]) if labels.get("page") else "")
            if source
            else ""
        )
        source_link = (
            f'<a href="{html.escape(url, quote=True)}">{html.escape(title)}</a>'
            if source
            else html.escape(title)
        )
        page.append(
            f'<article data-kind="{pair["kind"]}" data-split="{pair["split"]}" data-rate="{pair["requested_bps"] // 1000}"><h2>{pair["kind"]} / {pair["split"]} / {pair["requested_bps"] // 1000} Kbps</h2><p>{source_link}</p><small>{pair["id"]} · actual episode video {pair["actual_episode_video_bps"] / 1000:.1f} Kbps · {"I" if pair["keyframe"] else "P"} frame · {pair["characters"]} source characters · VBV warnings {pair["episode_vbv_underflows"]}</small><section><div><p>源渲染 / HR</p><a href="{pair["hr"]}"><img loading="lazy" src="{pair["hr"]}"></a></div><div><p>HEVC 解码 / LR（点击看原始像素）</p><a href="{pair["lr"]}"><img loading="lazy" src="{pair["lr"]}"></a></div></section><details><summary>源文字与标注</summary><a href="{pair["labels"]}">文字框、原文、渲染参数</a><pre>{html.escape(" ".join(w["text"] for w in labels["words"]))}</pre></details></article>'
        )
    page.append(
        """<script>function filter(){for(let a of document.querySelectorAll('article'))a.hidden=['kind','split','rate'].some(k=>document.getElementById(k).value&&a.dataset[k]!==document.getElementById(k).value)}for(let e of document.querySelectorAll('select'))e.onchange=filter;</script></html>"""
    )
    (root / "index.html").write_text("\n".join(page))
    atomic_json(
        root / "gallery.json",
        dict(
            selected_view_ids=[p["id"] for p in selected],
            pairs_sha256=sha256(root / "hevc-v2/pairs.json"),
            html_sha256=sha256(root / "index.html"),
            script_sha256=sha256(Path(__file__)),
            finished=time.time(),
        ),
    )
    print("GALLERY", len(selected), "stratified pairs", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
