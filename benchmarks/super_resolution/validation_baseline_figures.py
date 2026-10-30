"""Standalone scientific figures and coverage audit for the validation matrix."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from benchmarks.super_resolution.validation_baselines import read
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();p=read(root/'PROTOCOL.json');inputs=read(root/'INPUTS.json');summary=read(root/'SUMMARY.json')
    expected={(r['clip'],r['config'],s['frame']) for r in inputs['runs'] for s in r['samples']}
    assert len(expected)==len(p['source_clips'])*len(p['configurations'])*3
    methods=p['models'];lookup={(r['method'],r['config']):r for r in summary['per_model']}
    backend=Counter();models={};timings={}
    for method in methods:
        folder=root/'evaluations'/method
        rows=[json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines()]
        assert len(rows)==len(expected) and {(r['clip'],r['config'],r['frame']) for r in rows}==expected
        models[method]=read(folder/'identity.json')['model']
        backend.update(r.get('pixel_metric_backend','cpu_skimage') for r in rows)
        timings[method]=dict(p50_ms=float(np.median([r['timing']['display_total_ms'] for r in rows])),
                            note='Quality sweep under concurrent CPU encoding and GPU metric load; use the dedicated archived forward benchmark for speed ranking.')
        for row in rows:
            assert all(np.isfinite(row[k]) for k in ('psnr','ssim','lpips','y_psnr','chroma_psnr'))
            if row['image']:assert (root/row['image']).is_file()
    atomic_json(root/'COVERAGE-AUDIT.json',dict(passed=True,unique_inputs=len(expected),methods=len(methods),
        total_rows=len(expected)*len(methods),pixel_metric_backends=dict(backend),
        official_model_identities=models,results_sha256={m:sha256(root/'evaluations'/m/'records.jsonl') for m in methods}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(17,16),layout='constrained')
    for col,bitrate in enumerate((600000,1000000)):
        cfgs=[c for c in p['configurations'] if c['bitrate']==bitrate]
        for row,(metric,higher) in enumerate((('psnr',True),('ssim',True),('lpips',False))):
            ax=axes[row,col]
            values=np.array([[lookup[(m,c['id'])][metric] for m in methods] for c in cfgs])
            ranks=np.empty_like(values)
            for j in range(len(methods)):
                order=np.argsort(-values[:,j] if higher else values[:,j])
                ranks[order,j]=np.arange(1,len(cfgs)+1)
            im=ax.imshow(ranks,vmin=1,vmax=len(cfgs),cmap='RdYlGn_r',aspect='auto')
            ax.set_xticks(range(len(methods)),methods,rotation=40,ha='right')
            ax.set_yticks(range(len(cfgs)),[f"{c['height']}p / {c['chroma']}" for c in cfgs])
            ax.set_title(f'{bitrate//1000} kbps — {metric.upper()} ({"higher" if higher else "lower"} is better)')
            for i in range(len(cfgs)):
                for j in range(len(methods)):
                    ax.text(j,i,f'{values[i,j]:.2f}' if metric=='psnr' else f'{values[i,j]:.3f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axes,shrink=.5,label='Rank within each model and target bitrate (1 = best)')
    fig.suptitle(f'Mac validation: official pretrained ×2 models, FP32\n{len(p["source_clips"])} complete encoded videos; three fixed sample frames/video; shared 1920×1080 display',fontsize=15)
    fig.savefig(root/'quality-matrix.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,b in zip(axes,(600000,1000000)):
        data=sorted([r for r in summary['cross_model'] if r['bitrate']==b],key=lambda r:r['mean_rank'])
        ax.barh([r['config'].replace(f'-{b//1000}k','') for r in data],[r['mean_rank'] for r in data])
        ax.invert_yaxis();ax.set_xlim(0,8);ax.set_xlabel('Mean rank across 8 methods × 3 metrics (lower is better)');ax.set_title(f'{b//1000} kbps')
    fig.savefig(root/'cross-model-ranks.png',dpi=150);plt.close(fig)
    lines=['','## 图与核验','', '![完整画质矩阵](quality-matrix.png)','',
           '色块表示同一模型、同一码率下的组合排名；数字为实际指标。每段视频等权，先平均三个固定帧。','',
           '![跨模型排名](cross-model-ranks.png)','',
           f'全部 {len(expected)*len(methods):,} 行已核对，{len(methods)} 种方法覆盖完全相同的 {len(expected):,} 个编码／帧输入，详见 COVERAGE-AUDIT.json。', '',
           '部分早期结果使用原始 skimage CPU 指标，其余使用等价的 float64 积分图 GPU 计算；完整 FHD 帧、随机图及恒定图的一致性检查见 GPU-METRIC-VALIDATION.json。', '',
           '本轮同时执行转码及 GPU 指标计算，过程耗时不能替代独立纯 forward 排行榜。当前全部模型使用 FP32；历史排行榜另含 AMP FP16，不能直接混作同一条件。']
    with (root/'REPORT.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    print(f'AUDITED {len(expected)*len(methods):,} rows; figures saved',flush=True)


if __name__=='__main__':main()
