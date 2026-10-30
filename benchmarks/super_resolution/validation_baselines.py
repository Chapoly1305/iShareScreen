"""Full-video Mac validation encoding matrix with official pretrained models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import av
import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


MODELS = ('edsr', 'imdn', 'safmn', 'quicksrnet', 'plainusr', 'rt4ksr', 'efrlfn')
SIZE = (1920, 1080)


def read(path):
    return json.loads(path.read_text())


def configurations(chromas=('420',)):
    return [dict(id=f'{h}p-{b//1000}k-{c}', height=h, width=h*16//9,
                 bitrate=b, chroma=c)
            for h in (540, 576, 720, 1080)
            for b in (600000, 1000000) for c in chromas]


def sample_video(path, indices, fps, size):
    """Seek from preceding keyframes and require the exact requested timestamps."""
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.thread_count = 2
        for index in indices:
            target = index/fps
            video.seek(int(target/stream.time_base), stream=stream, backward=True)
            found = False
            for frame in video.decode(video=0):
                actual = float(frame.pts*frame.time_base)
                if actual < target-.002:
                    continue
                assert abs(actual-target) < .002, (path, index, actual)
                assert (frame.width,frame.height) == size and not frame.is_corrupt
                yield index, frame.to_ndarray(format='rgb24')
                found = True
                break
            assert found, (path,index)


def prepare(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_path = args.project / 'datasets/validation/mac/processed_recordings/manifest.json'
    original = read(source_path)
    protocol = dict(schema='mac_validation_official_matrix_v1',
        created=time.time(), models=['baseline', *MODELS], configurations=configurations(args.chroma),
        source_manifest=str(source_path.resolve()), source_manifest_sha256=sha256(source_path),
        script_sha256=sha256(Path(__file__)), fps=30, display_size=SIZE,
        sampling='Three fixed frames at 25%, 50%, 75% of each complete video, at least 1 second after start. All configurations share exact source frames.',
        encoder='libx265 fast/zerolatency; no B frames/lookahead; 2-second GOP; one-second VBV; strict-CBR/HRD requests',
        color='8-bit full-range BT.709; Lanczos reduction; explicit '+ '/'.join('yuv'+c+'p' for c in args.chroma),
        reconstruction='baseline bicubic to FHD (1080p direct); models native x2 then area downsample to FHD',
        metrics='RGB PSNR, RGB SSIM (skimage defaults), AlexNet LPIPS, Y/CbCr PSNR, per-frame processing time',
        aggregation='Equal weight per recording after averaging sampled frames; 24 recordings share one capture session, not 24 independent sessions.',
        limits='Development validation. Full videos encoded, model quality sampled at three timestamps, not every frame; no temporal-flicker score. RGB8 recording proxy is not pristine acquisition. No source-truth OCR labels for the full set; image quality cannot establish text accuracy. Video packet rate excludes network/audio overhead. CUDA time is not Mac playback time.',
        source_clips=[])
    for clip in original['clips']:
        assert clip['fps'] == '60/1' or clip['fps'] == '60'
        reference = source_path.parent / clip['reference']
        assert sha256(reference) == clip['reference_sha256']
        frames = (clip['frames']+1)//2
        indices = [max(30, int((frames-1)*p)) for p in (.25, .5, .75)]
        assert len(set(indices)) == 3 and max(indices) < frames
        protocol['source_clips'].append(dict(id=clip['id'], content=clip['content'],
            reference=str(reference.resolve()), reference_sha256=clip['reference_sha256'],
            original_frames=clip['frames'], frames_30fps=frames, sample_indices=indices,
            session_group=clip['session_group']))
        folder = root/'references'/clip['id']; folder.mkdir(parents=True)
        for index, rgb in sample_video(reference, [j*2 for j in indices], 60, SIZE):
            Image.fromarray(rgb).save(folder/f'{index//2}.png')
        print('REFERENCES',clip['id'],flush=True)
    atomic_json(root/'PROTOCOL.json', protocol)
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED', len(protocol['source_clips']), 'full videos;', len(protocol['configurations']), 'configurations', flush=True)


def encode_one(root, protocol, clip, cfg):
    folder = root/'encodes'/clip['id']/cfg['id']; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder/'COMPLETED.json'
    if receipt.exists():
        row = read(receipt)
        assert row['protocol_sha256'] == sha256(root/'PROTOCOL.json')
        assert sha256(root/row['video']) == row['video_sha256']
        for sample in row['samples']:
            assert sha256(root/sample['lr']) == sample['lr_sha256']
        return row
    output = folder/'encoded.mp4'
    if output.exists():
        raise FileExistsError(f'Incomplete encode retained for inspection: {output}')
    kbps = cfg['bitrate']//1000
    params = (f'pools=2:frame-threads=1:keyint=60:min-keyint=1:bframes=0:rc-lookahead=0:'
              f'bitrate={kbps}:vbv-maxrate={kbps}:vbv-bufsize={kbps}:vbv-init=0.5:strict-cbr=1:hrd=1:log-level=info')
    filters = (f'select=not(mod(n\\,2)),setpts=N/(30*TB),'
               f'scale={cfg["width"]}:{cfg["height"]}:flags=lanczos:in_range=full:out_range=full:out_color_matrix=bt709')
    # select the even 60fps source frames explicitly, avoiding fps-filter phase ambiguity.
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostdin',
        '-threads', '2', '-i', clip['reference'], '-an', '-sn', '-filter_threads', '1',
        '-vf', filters, '-r', '30', '-c:v', 'libx265', '-preset', 'fast', '-tune', 'zerolatency',
        '-b:v', str(cfg['bitrate']), '-minrate', str(cfg['bitrate']),
        '-maxrate', str(cfg['bitrate']), '-bufsize', str(cfg['bitrate']),
        '-pix_fmt', 'yuv'+cfg['chroma']+'p', '-color_range', 'pc', '-colorspace', 'bt709',
        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-x265-params', params,
        '-tag:v', 'hvc1', '-movflags', '+faststart', str(output)]
    started = time.time()
    with (folder/'encode.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=log)
    stream = read_probe(output)
    assert (stream['width'], stream['height']) == (cfg['width'], cfg['height'])
    assert stream['pix_fmt'] in ('yuv'+cfg['chroma']+'p', 'yuvj'+cfg['chroma']+'p')
    assert stream['color_range'] == 'pc' and stream['color_space'] == 'bt709'
    assert stream['has_b_frames'] == 0
    samples = []
    for i, rgb in sample_video(output,clip['sample_indices'],30,(cfg['width'],cfg['height'])):
        path = folder/f'frame-{i}.png'
        Image.fromarray(rgb).save(path)
        ref = Path('references')/clip['id']/f'{i}.png'
        samples.append(dict(frame=i,time_s=i/30,lr=str(path.relative_to(root)),
            lr_sha256=sha256(path),hr=str(ref),hr_sha256=sha256(root/ref)))
    assert len(samples) == 3
    packets = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
        '-select_streams', 'v:0', '-show_packets', '-show_entries',
        'packet=size,pts_time', '-of', 'json', str(output)], text=True))['packets']
    retained = [p for p in packets if float(p['pts_time']) >= 1-1e-6]
    assert len(packets) == clip['frames_30fps']
    rate = sum(int(p['size'])*8 for p in retained)/(len(retained)/30)
    windows = defaultdict(int)
    for p in packets:
        windows[int(float(p['pts_time']))] += int(p['size'])*8
    underflows = (folder/'encode.log').read_text().count('VBV underflow')
    row = dict(clip=clip['id'], config=cfg['id'], content=clip['content'],
        protocol_sha256=sha256(root/'PROTOCOL.json'), samples=samples,
        video=str(output.relative_to(root)), video_sha256=sha256(output),
        actual_video_bps_after_first_second=rate, one_second_packet_bits=dict(windows),
        budget_pass=rate <= cfg['bitrate']*1.05 and underflows == 0,
        vbv_underflows=underflows, command=command, stream=stream,
        elapsed_s=time.time()-started)
    atomic_json(receipt, row)
    return row


def read_probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
        '-select_streams', 'v:0', '-show_streams', '-of', 'json', str(path)], text=True))['streams'][0]


def encode(args):
    root = args.root.resolve(); protocol = read(root/'PROTOCOL.json')
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(encode_one, root, protocol, clip, cfg)
                 for clip in protocol['source_clips'] for cfg in protocol['configurations']]
        for future in as_completed(tasks):
            row = future.result(); rows.append(row)
            atomic_json(root/'encode-progress.json', dict(completed=len(rows), total=len(tasks), latest=[row['clip'], row['config']]))
            print('ENCODED', len(rows), '/', len(tasks), row['clip'], row['config'], flush=True)
    rows.sort(key=lambda r:(r['clip'],r['config']))
    atomic_json(root/'INPUTS.json', dict(protocol_sha256=sha256(root/'PROTOCOL.json'),
        runs=rows, total_frames=sum(c['frames_30fps'] for c in protocol['source_clips'])*len(protocol['configurations'])))
    atomic_json(root/'ENCODING-COMPLETED.json', dict(inputs_sha256=sha256(root/'INPUTS.json'),
        runs=len(rows), samples=sum(len(r['samples']) for r in rows), finished=time.time()))


def pixel_quality(reference, output):
    scores = metrics(reference, output)
    # BT.709 full-range floating-point transform, without clipping/rounding chroma.
    matrix = np.array([[.2126,.7152,.0722],[-.114572,-.385428,.5],[.5,-.454153,-.045847]], np.float32)
    diff = (reference.astype(np.float32)-output.astype(np.float32)) @ matrix.T
    for name, error in [('y_psnr',diff[...,0]), ('chroma_psnr',diff[...,1:])]:
        mse = float(np.mean(error**2)); scores[name] = 10*math.log10(255**2/mse) if mse else None
    return {k:float(v) if v is not None and np.isfinite(v) else None for k,v in scores.items()}


def evaluate(args):
    import torch
    from isharescreen.enhancement import candidates
    from isharescreen.enhancement.models import Upscaler
    from benchmarks.super_resolution.train import perceptual_network
    from benchmarks.super_resolution.video_perceptual import normalized_tensor
    root = args.root.resolve(); protocol = read(root/'PROTOCOL.json'); inputs = read(root/'INPUTS.json')
    assert read(root/'ENCODING-COMPLETED.json')['inputs_sha256'] == sha256(root/'INPUTS.json')
    assert inputs['protocol_sha256'] == sha256(root/'PROTOCOL.json')
    cv2.setNumThreads(1); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    network, lpips_identity = perceptual_network(args.device)
    configs = {c['id']:c for c in protocol['configurations']}
    locks = read(Path(__file__).parent/'models.lock.json')
    for method in args.models:
        out = root/'evaluations'/method; out.mkdir(parents=True, exist_ok=True)
        path = out/'records.jsonl'; existing = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        done = {(r['clip'],r['config'],r['frame']) for r in existing}
        assert len(done) == len(existing)
        binding = dict(protocol_sha256=sha256(root/'PROTOCOL.json'), inputs_sha256=sha256(root/'INPUTS.json'),
                       runner_sha256=sha256(Path(__file__)), method=method)
        if (out/'identity.json').exists():assert read(out/'identity.json')['binding'] == binding
        if (out/'COMPLETED.json').exists():
            assert len(existing) == len(inputs['runs'])*3
            assert read(out/'COMPLETED.json')['records_sha256'] == sha256(path)
            continue
        model = None
        if method in ('edsr','imdn','safmn'):
            model = Upscaler(method, device=args.device, precision='fp32')
            assert model.provenance['checkpoint_sha256'] == locks[method]['sha256']
        elif method != 'baseline':
            model = candidates.TorchUpscaler(method, device=args.device, precision='fp32')
        provenance = model.provenance if model else dict(id='baseline',type='bicubic/native')
        atomic_json(out/'identity.json', dict(binding=binding, model=provenance, lpips=lpips_identity,
            host=platform.node(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            cuda=torch.version.cuda, pid=os.getpid(), started=time.time()))
        warmed = set(); pending = []; completed = len(existing); started = time.time()
        with path.open('a') as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
            def drain():
                nonlocal completed
                future, row = pending.pop(0); row.update(future.result())
                handle.write(json.dumps(row)+'\n'); handle.flush(); completed += 1
                atomic_json(out/'progress.json',dict(completed=completed,total=len(inputs['runs'])*3,elapsed_s=time.time()-started))
                if completed % 48 == 0:print('SCORED', method, completed, '/', len(inputs['runs'])*3, flush=True)
            for run in inputs['runs']:
                cfg = configs[run['config']]
                for index, sample in enumerate(run['samples']):
                    key = (run['clip'],run['config'],sample['frame'])
                    if key in done:continue
                    assert sha256(root/sample['lr']) == sample['lr_sha256']
                    assert sha256(root/sample['hr']) == sample['hr_sha256']
                    rgb = np.asarray(Image.open(root/sample['lr']).convert('RGB'))
                    reference = np.asarray(Image.open(root/sample['hr']).convert('RGB'))
                    if model:
                        if cfg['height'] not in warmed:
                            model(rgb); model(rgb); warmed.add(cfg['height'])
                        output, timing = model(rgb)
                        assert output.shape == (2*cfg['height'],2*cfg['width'],3)
                        t = time.perf_counter()
                        if output.shape[1::-1] != SIZE:output = cv2.resize(output,SIZE,interpolation=cv2.INTER_AREA)
                        timing['display_resize_ms'] = (time.perf_counter()-t)*1000
                        timing['display_total_ms'] = timing['total_ms']+timing['display_resize_ms']
                    else:
                        t = time.perf_counter()
                        output = rgb if rgb.shape[1::-1] == SIZE else cv2.resize(rgb,SIZE,interpolation=cv2.INTER_CUBIC)
                        timing = dict(display_total_ms=(time.perf_counter()-t)*1000)
                    with torch.inference_mode():
                        lpips = float(network(normalized_tensor(reference,args.device),normalized_tensor(output,args.device)).item())
                    assert math.isfinite(lpips)
                    image = None
                    if index == 1:
                        target = out/'images'/run['clip']/(run['config']+'.png')
                        target.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(output).save(target)
                        image = str(target.relative_to(root))
                    row = dict(clip=run['clip'],content=run['content'],config=run['config'],frame=sample['frame'],
                        method=method,lpips=lpips,timing=timing,image=image,budget_pass=run['budget_pass'])
                    pending.append((pool.submit(pixel_quality,reference,output),row))
                    if len(pending) >= args.workers*2:drain()
            while pending:drain()
        assert completed == len(inputs['runs'])*3
        atomic_json(out/'COMPLETED.json',dict(records=completed,records_sha256=sha256(path),finished=time.time(),elapsed_s=time.time()-started))
        del model
        torch.cuda.empty_cache()


def report(args):
    root=args.root.resolve();protocol=read(root/'PROTOCOL.json');inputs=read(root/'INPUTS.json')
    summaries=[];by_model={}; all_rows=[]
    for method in protocol['models']:
        folder=root/'evaluations'/method;path=folder/'records.jsonl'
        assert read(folder/'COMPLETED.json')['records_sha256']==sha256(path)
        rows=[json.loads(s) for s in path.read_text().splitlines()];all_rows+=rows
        grouped=defaultdict(list)
        for r in rows:grouped[r['config']].append(r)
        scores=[]
        for cfg,records in grouped.items():
            per_clip=defaultdict(list)
            for r in records:per_clip[r['clip']].append(r)
            assert len(per_clip)==len(protocol['source_clips']) and all(len(v)==3 for v in per_clip.values())
            score=dict(method=method,config=cfg,clips=len(per_clip),frames=len(records),
                budget_pass_clips=sum(next(x for x in inputs['runs'] if x['clip']==c and x['config']==cfg)['budget_pass'] for c in per_clip))
            for metric in ('psnr','ssim','lpips','y_psnr','chroma_psnr'):
                score[metric]=float(np.mean([np.mean([r[metric] for r in v]) for v in per_clip.values()]))
            for metric in ('display_total_ms','inference_ms'):
                times=[r['timing'][metric] for r in records if metric in r['timing']]
                score[metric+'_p50']=float(np.median(times)) if times else None
                score[metric+'_p95']=float(np.percentile(times,95)) if times else None
            scores.append(score)
        by_model[method]=scores;summaries+=scores
    # Compare within each requested bandwidth, giving every model equal weight.
    cross=[]
    for b in (600000,1000000):
        cfgs=[c for c in protocol['configurations'] if c['bitrate']==b]
        ranks=defaultdict(list);first=Counter()
        for method,scores in by_model.items():
            selected=[s for s in scores if s['config'] in {c['id'] for c in cfgs}]
            for metric,reverse in [('psnr',True),('ssim',True),('lpips',False)]:
                ordered=sorted(selected,key=lambda s:s[metric],reverse=reverse)
                first[ordered[0]['config']]+=1
                for rank,s in enumerate(ordered,1):ranks[s['config']].append(rank)
        cross += [dict(bitrate=b,config=c['id'],mean_rank=float(np.mean(ranks[c['id']])),
                       worst_rank=max(ranks[c['id']]),metric_wins=first[c['id']]) for c in cfgs]
    atomic_json(root/'SUMMARY.json',dict(per_model=summaries,cross_model=cross,
        rank_definition='Mean of PSNR/SSIM/LPIPS configuration ranks, equal weight for eight methods. Descriptive validation heuristic, not a unique quality or usability metric.'))
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    lines=['# 官方模型 validation 基础评测','',f'{len(protocol["source_clips"])} 段完整视频 × {len(protocol["configurations"])} 种编码组合；每段固定三个位置，共 {len(all_rows):,} 个模型／帧结果。','',
        '统一在 1920×1080 比较。所有模型为官方原始 ×2 权重，未训练。1080p 模型路径先做 ×2 再缩回 FHD；无模型路径为原尺寸直通。','',
        '## 跨模型组合排序','', '| 码率 | 组合 | 平均名次（低更好） | 最差名次 | 指标第一次数 / 24 |','| --- | --- | ---: | ---: | ---: |']
    for b in (600000,1000000):
        for row in sorted([r for r in cross if r['bitrate']==b],key=lambda r:r['mean_rank']):
            lines.append(f"| {b//1000} kbps | {row['config']} | {row['mean_rank']:.2f} | {row['worst_rank']} | {row['metric_wins']} |")
    lines+=['','## 各模型在每档码率的最佳组合','', '| 模型 | 码率 | PSNR 最佳 | SSIM 最佳 | LPIPS 最佳 |','| --- | --- | --- | --- | --- |']
    for method,scores in by_model.items():
        for b in (600000,1000000):
            selected=[s for s in scores if any(c['id']==s['config'] and c['bitrate']==b for c in protocol['configurations'])]
            winners=[max(selected,key=lambda s:s['psnr']),max(selected,key=lambda s:s['ssim']),min(selected,key=lambda s:s['lpips'])]
            lines.append(f"| {method} | {b//1000} kbps | {winners[0]['config']} | {winners[1]['config']} | {winners[2]['config']} |")
    invalid=[r for r in inputs['runs'] if not r['budget_pass']]
    lines+=['','## 边界', '',f'{len(inputs["runs"])} 个编码中有 {len(invalid)} 个未通过“去掉首秒后视频码率不超过请求 105% 且无 VBV underflow”的检查。上表保留全部结果，不能把未达标输入称为严格限带宽通过。逐项实际码率见 INPUTS.json。', '',
        '这是同一 Mac 会话的开发验证结果。全片编码，但模型只处理三个固定抽帧，未测全帧吞吐、闪烁或真实网络。PSNR/SSIM/LPIPS 不是文字正确率；当前没有覆盖这 24 段视频的原始文字真值，未报告伪造的 CER。Linux 4090 耗时不能代表 Mac 客户端。', '',
        '平均名次仅用于跨模型筛选组合；各指标和模型的最佳项可能不同。完整分数、实际码率和耗时见 summary.csv、SUMMARY.json、evaluations/。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(root/'COMPLETED.json',dict(records=len(all_rows),summary_sha256=sha256(root/'SUMMARY.json'),finished=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','encode','evaluate','report'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--chroma',nargs='+',choices=('420','444'),default=['420'],help='New sweeps default to 420; explicitly opt into chroma comparisons.')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--models',nargs='+',choices=('baseline',*MODELS),default=['baseline',*MODELS])
    args=parser.parse_args()
    globals()[args.action](args)


if __name__=='__main__':main()
