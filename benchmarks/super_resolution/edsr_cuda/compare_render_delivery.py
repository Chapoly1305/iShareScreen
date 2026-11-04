"""Compare pre-resize and native1440 delivery through the actual WGPU renderer.

Offscreen 1080p on lab4090, not panel latency or CUDA/WGPU zero-copy interop.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import socket
import time

import cv2
import numpy as np
from PIL import Image
import torch
import triton
import triton.language as tl
import wgpu

from benchmarks.super_resolution.edsr_cuda.common import Engine, digest, guard, save, stats
from compare_resize import area_float, quality, ocr, edit_distance
from isharescreen.enhancement.cuda_kernels import pack_area


@triton.jit
def pack_native(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    c = i % 4
    v = tl.load(X+c*1440*2560+i//4, (i<N)&(c<3), 0)*255
    v = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(v), 0), 255)
    tl.store(Y+i, tl.where(c==3, 255, v).to(tl.uint8), i<N)


class TimestampDevice:
    """Delegate production calls unchanged, optionally timestamp the draw pass."""
    def __init__(self, device):
        self.device = device
        self.active = False

    def __getattr__(self, name):
        return getattr(self.device, name)

    def create_command_encoder(self, **kwargs):
        encoder = self.device.create_command_encoder(**kwargs)
        return TimestampEncoder(encoder, self) if self.active else encoder

    def start(self, count):
        self.query = self.device.create_query_set(type='timestamp', count=2)
        self.buffer = self.device.create_buffer(
            size=256*count, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
        self.count, self.index, self.active = count, 0, True

    def finish(self):
        self.active = False
        assert self.index == self.count
        data = np.frombuffer(self.device.queue.read_buffer(self.buffer), np.uint64).reshape(-1, 32)
        assert np.all(data[:, 1] > data[:, 0])
        from wgpu.backends.wgpu_native._ffi import lib
        period = float(lib.wgpuQueueGetTimestampPeriod(self.device.queue._internal))
        samples = ((data[:, 1]-data[:, 0]).astype(np.float64)*period/1e6).tolist()
        ticks = data[:, :2].tolist()
        self.query.destroy()
        self.buffer.destroy()
        return dict(timestamp_period_ns=period, raw_ticks=ticks, **stats(samples))


class TimestampEncoder:
    def __init__(self, encoder, device):
        self.encoder, self.device = encoder, device

    def begin_render_pass(self, **kwargs):
        kwargs['timestamp_writes'] = dict(query_set=self.device.query,
                                         beginning_of_pass_write_index=0,
                                         end_of_pass_write_index=1)
        return self.encoder.begin_render_pass(**kwargs)

    def finish(self):
        d = self.device
        assert d.index < d.count
        self.encoder.resolve_query_set(d.query, 0, 2, d.buffer, d.index*256)
        d.index += 1
        return self.encoder.finish()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--renderer-source', type=Path, required=True)
    p.add_argument('--expected-gpu-uuid', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ['ISS_EXPECTED_GPU_UUID'] = a.expected_gpu_uuid
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    r = dict(complete=False, smoke=a.smoke, started=time.time(), host=socket.gethostname(),
             gpu_start=guard(), script_sha256=digest(__file__),
             renderer_sha256=digest(a.renderer_source), engine_sha256=digest(a.engine),
             manifest_sha256=digest(a.validation/'INPUTS.json'),
             torch=torch.__version__, triton=triton.__version__, wgpu=wgpu.__version__,
             opencv=cv2.__version__,
             scope='Same frozen EDSR output, same 1080p rgba8unorm offscreen target, '
             'production Renderer upload_enhanced/draw with linear sampler; no cursor. '
             'Legacy Area1080, float Area1080, or native RGBA1440. Stored RGB inputs, '
             'not live decode/network. Not physical display or Mac timing.',
             bridge_scope='CUDA pack -> synchronous RGBA D2H -> contiguous CPU RGB copy '
             '-> production upload_enhanced CPU RGBA packing/write_texture -> draw '
             '-> queue completion. No output readback in bridge timing. Excludes EDSR. '
             'This explicitly uses the existing CPU upload API, NOT zero-copy interop.',
             renderer_setup='Two Renderer instances: canvas960x540 accepts1080; '
             'canvas1280x720 accepts1440. Same 16:9 full-frame UVs; no crop/cursor/input '
             'mapping test. No changes to product Renderer.',
             timing_protocol='Two reversed method orders; CUDA Graph pack 50warm/1000samples; '
             'resident-texture draw 50warm/1000timestamp samples; bridge20warm/200wall '
             'samples, one at a time with queue completion. All resources preallocated '
             'except allocations performed by the documented existing CPU bridge.',
             completion_sync='wgpu-native GPUDevice._poll_wait (wgpuDevicePoll wait=true). '
             'Installed wgpu0.32 queue work-done callback has an ABI mismatch; '
             'native device wait avoids it without an image readback or library changes.',
             ocr_scope='8 deterministic central crops; Tesseract eng PSM6 OCR of HR is '
             'pseudo reference, whitespace stripped. Mixed Chinese/English, not human CER.',
             rows=[], ocr=[], timings=[])
    assert r['engine_sha256'] == '74f418fb7ad0cd03ae25d6ff701fead1414a6f180e334d13e9281263007dde74'
    save(a.output/'RESULTS.json', r)
    spec = importlib.util.spec_from_file_location('renderer_under_test', a.renderer_source)
    renderer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer_module)
    adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
    r['render_adapter'] = dict(adapter.info)
    assert '4090' in str(r['render_adapter']) and r['render_adapter']['backend_type'] == 'Vulkan'
    device = adapter.request_device_sync(required_features=['timestamp-query'])
    proxy = TimestampDevice(device)
    renderers = {1080: renderer_module.Renderer(proxy, 'rgba8unorm', 960, 540),
                 1440: renderer_module.Renderer(proxy, 'rgba8unorm', 1280, 720)}
    target = device.create_texture(size=(1920,1080,1), format='rgba8unorm',
                                   usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    view = target.create_view()
    engine = Engine(a.engine)
    x = engine.output
    outputs = {h: torch.empty((h, h*16//9, 4), device='cuda', dtype=torch.uint8)
               for h in [1080, 1440]}
    names = ['legacy_area1080', 'float_area1080', 'native1440_linear']

    def pack(name):
        out = outputs[1440 if name == 'native1440_linear' else 1080]
        n = out.numel()
        if name == 'native1440_linear':
            pack_native[(triton.cdiv(n,256),)](x, out, n, 256, enable_fp_fusion=False)
        elif name == 'legacy_area1080':
            pack_area[(triton.cdiv(n,256),)](x,out,1440*2560,2560,1,4,n,256,enable_fp_fusion=False)
        else:
            area_float[(triton.cdiv(n,256),)](x,out,False,n,256,enable_fp_fusion=False)
        return out

    def upload_draw(name, rgb):
        renderer = renderers[1440 if name == 'native1440_linear' else 1080]
        renderer.upload_enhanced(rgb)
        renderer.draw(view, 1920, 1080)

    def read_target():
        raw = device.queue.read_texture({'texture':target}, {'bytes_per_row':1920*4}, (1920,1080,1))
        rgba = np.frombuffer(raw, np.uint8).reshape(1080,1920,4)
        assert np.all(rgba[...,3] == 255)
        return rgba[...,:3].copy()

    import json
    clips = [v for v in json.loads((a.validation/'INPUTS.json').read_text())['runs']
             if v['config']=='720p-600k-420']
    assert len(clips)==24
    if a.smoke:
        clips = clips[:1]
    for ci, clip in enumerate(clips):
        samples = clip['samples'][:1] if a.smoke else clip['samples']
        for si, sample in enumerate(samples):
            for key in ['lr','hr']:
                assert digest(a.validation/sample[key])==sample[key+'_sha256']
            lr = np.array(Image.open(a.validation/sample['lr']).convert('RGB'))
            hr = np.array(Image.open(a.validation/sample['hr']).convert('RGB'))
            engine.input.copy_(torch.from_numpy(lr.copy()).permute(2,0,1).unsqueeze(0).cuda().float()/255)
            engine.run()
            native = x[0].permute(1,2,0).contiguous().cpu().numpy()*255
            assert np.isfinite(native).all()
            native8 = np.clip(np.rint(native),0,255).astype(np.uint8)
            refs = {'legacy_area1080':cv2.resize(native8,(1920,1080),interpolation=cv2.INTER_AREA),
                    'float_area1080':np.clip(np.rint(cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA)),0,255).astype(np.uint8),
                    'native1440_linear':cv2.resize(native8.astype(np.float32),(1920,1080),interpolation=cv2.INTER_LINEAR)}
            selected = ci%3==0 and si==0
            crop = (slice(270,810),slice(480,1440))
            if selected:
                Image.fromarray(hr[crop]).save(a.output/f'crop-{ci:02d}-hr.png')
                truth = '' if a.smoke else ocr(hr[crop])
            for name in names:
                rgba = pack(name).cpu().numpy()
                assert np.all(rgba[...,3]==255)
                rgb = rgba[...,:3].copy()
                if name=='native1440_linear':
                    assert np.array_equal(rgb, native8)
                else:
                    assert np.array_equal(rgb, refs[name])
                upload_draw(name, rgb)
                displayed = read_target()
                ref = np.clip(np.rint(refs[name]),0,255).astype(np.uint8)
                error = np.abs(displayed.astype(np.int16)-ref.astype(np.int16))
                # Hardware filtering can differ from ideal float interpolation/rounding.
                assert error.max()<=2, (name, error.max())
                r['rows'].append(dict(clip=clip['clip'], frame=sample['frame'], method=name,
                    **quality(hr,displayed), texture_bytes=rgb.shape[0]*rgb.shape[1]*4,
                    cpu_reference_max_error=int(error.max()),
                    cpu_reference_different_fraction=float(np.mean(error!=0))))
                if selected:
                    Image.fromarray(displayed[crop]).save(a.output/f'crop-{ci:02d}-{name}.png')
                    if not a.smoke:
                        text = ocr(displayed[crop])
                        r['ocr'].append(dict(clip=clip['clip'],method=name,hr_text=truth,text=text,
                                            reference_chars=len(truth),edits=edit_distance(truth,text)))
            save(a.output/'RESULTS.json',r)
        print('quality',ci+1,'/',len(clips),flush=True)

    # Graphs only capture conversion/packing; inference is never rerun here.
    graphs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name in names:
            for _ in range(10):
                pack(name)
            torch.cuda.synchronize()
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name],stream=stream):
                pack(name)
    torch.cuda.synchronize()
    for ri, order in enumerate([names, list(reversed(names))]):
        for name in order:
            guard()
            count = 5 if a.smoke else 1000
            for _ in range(50):
                graphs[name].replay()
            torch.cuda.synchronize()
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(count):
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            r['timings'].append(dict(round=ri,method=name,scope='cuda_pack_graph',**stats(samples)))
            rgba = pack(name).cpu().numpy()
            rgb = rgba[...,:3].copy()
            upload_draw(name,rgb)
            renderer = renderers[1440 if name=='native1440_linear' else 1080]
            for _ in range(50):
                renderer.draw(view,1920,1080)
            device._poll_wait()
            proxy.start(count)
            for _ in range(count):
                renderer.draw(view,1920,1080)
            r['timings'].append(dict(round=ri,method=name,scope='resident_draw_gpu',**proxy.finish()))

            def bridge():
                graphs[name].replay()
                h = 1440 if name=='native1440_linear' else 1080
                cpu = outputs[h].cpu().numpy()[...,:3].copy()
                upload_draw(name,cpu)
                device._poll_wait()

            for _ in range(20):
                bridge()
            samples = []
            for _ in range(5 if a.smoke else 200):
                start_wall = time.perf_counter()
                bridge()
                samples.append((time.perf_counter()-start_wall)*1000)
            r['timings'].append(dict(round=ri,method=name,scope='cpu_bridge_wall',**stats(samples)))
            save(a.output/'RESULTS.json',r)
            print('timing',ri,name,[(t['scope'],round(t['p95_ms'],6)) for t in r['timings'][-3:]],flush=True)
    base = [v for v in r['rows'] if v['method']==names[0]]
    r['summary'] = {}
    for name in names:
        rows = [v for v in r['rows'] if v['method']==name]
        assert [(v['clip'],v['frame']) for v in rows]==[(v['clip'],v['frame']) for v in base]
        delta = [v['psnr']-b['psnr'] for v,b in zip(rows,base)]
        texts = [v for v in r['ocr'] if v['method']==name]
        chars = sum(v['reference_chars'] for v in texts)
        r['summary'][name] = dict(psnr=float(np.mean([v['psnr'] for v in rows])),
            ssim=float(np.mean([v['ssim'] for v in rows])),mean_psnr_delta=float(np.mean(delta)),
            worst_psnr_delta=min(delta),best_psnr_delta=max(delta),
            improved_frames=sum(v>1e-6 for v in delta),regressed_frames=sum(v< -1e-6 for v in delta),
            ocr_edits=sum(v['edits'] for v in texts),ocr_reference_chars=chars,
            ocr_proxy_error=sum(v['edits'] for v in texts)/chars if chars else None,
            p95_ms={scope:[v['p95_ms'] for v in r['timings'] if v['method']==name and v['scope']==scope]
                    for scope in ['cuda_pack_graph','resident_draw_gpu','cpu_bridge_wall']})
    r.update(complete=True,finished=time.time(),gpu_end=guard())
    save(a.output/'RESULTS.json',r)
    print(json.dumps(r['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
