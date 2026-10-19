"""Offline interactive preview using the real session dropdown/control/renderer.

No connection or credentials needed. It validates display integration with a
decoded-frame fixture or recorded video. It does not exercise live transport.
"""
import argparse
import json
import logging
from pathlib import Path
import tempfile
import threading
import time
import webbrowser

import glfw
import numpy as np
from PIL import Image
from rendercanvas.glfw import RenderCanvas
import wgpu

from isharescreen.enhancement.live import EnhancementDisplay, EnhancementWorker
from isharescreen.frontend.desktop.gpu import Renderer
from isharescreen.gui import connect
from isharescreen.proxy.control import ControlServer
from isharescreen.proxy.media.tiles import TileFrame
from isharescreen.proxy.session import Session
if __package__:
    from .replay import VideoReplay, preview_tile
else:
    from replay import VideoReplay, preview_tile


def load_preview_tile(path):
    """Load an image or use the production AVFrame→TileFrame extractor."""
    if path.suffix.lower() in (".hevc", ".h265", ".mp4", ".mkv", ".mov"):
        import av
        with av.open(str(path)) as container:
            frame = next(container.decode(video=0))
            return preview_tile(frame)
    rgb=np.array(Image.open(path).convert("RGB"))
    height,width=rgb.shape[:2]
    # BT.709 full range, matching the decoded canvas shader.
    f=rgb.astype(np.float32)/255
    y=.2126*f[...,0]+.7152*f[...,1]+.0722*f[...,2]
    u=(f[...,2]-y)/1.8556+.5;v=(f[...,0]-y)/1.5748+.5
    def plane(p): return np.clip(np.rint(p*255),0,255).astype(np.uint8).tobytes()
    return TileFrame(plane(y),plane(u),plane(v),width,height,width,width,width,height)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True,help="RGB image or full-range BT.709 HEVC fixture/video")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--port",type=int,default=0)
    parser.add_argument("--auto-quit",type=float,default=600)
    parser.add_argument("--no-browser",action="store_true")
    parser.add_argument("--state-file",type=Path)
    parser.add_argument("--loop-video",action="store_true",help="Loop a recorded MP4/MKV/MOV while switching models")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tile=load_preview_tile(args.input)
    replay=None
    if args.input.suffix.lower() in (".mp4", ".mkv", ".mov"):
        replay=VideoReplay(args.input,loop=args.loop_video)
    width,height=tile.width,tile.height
    worker=EnhancementWorker(device=args.device)
    window=RenderCanvas(title="iShareScreen offline enhancement preview",size=(1280,720),max_fps=60)
    adapter=wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device=adapter.request_device_sync()
    ctx=window.get_context("wgpu");fmt=ctx.get_preferred_format(adapter).removesuffix("-srgb")
    ctx.configure(device=device,format=fmt,alpha_mode="opaque")
    renderer=Renderer(device,fmt,width,height);renderer.upload_tile(0,tile,height)
    display=EnhancementDisplay(worker,renderer,width,height,1);display.update_tile(0,tile,height)
    def draw():
        texture=ctx.get_current_texture()
        renderer.draw(texture.create_view(),texture.width,texture.height)
    window.request_draw(draw)
    # Use Session's actual command dispatcher and ControlServer transport.
    session=Session.__new__(Session);session._enhancement_backend=worker
    server=connect.http.server.ThreadingHTTPServer(("127.0.0.1",args.port),connect._Handler)
    connect._GUI_PORT=server.server_port
    connect._STATE.update(host="Offline frame preview",frontend="desktop",status="running",pid=__import__("os").getpid())
    temp=tempfile.TemporaryDirectory(prefix="iss-sr-preview-")
    control=ControlServer(str(Path(temp.name)/"control.sock"),on_command=session._handle_ctrl_command)
    control.start()
    threading.Thread(target=connect._control_reader,args=(str(Path(temp.name)/"control.sock"),),daemon=True).start()
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f"http://127.0.0.1:{server.server_port}/dashboard"
    print(f"PREVIEW_DASHBOARD={url}",flush=True)
    if not args.no_browser: webbrowser.open(url)
    deadline=time.monotonic()+args.auto_quit;last_publish=0
    try:
        while time.monotonic()<deadline and not window.get_closed():
            glfw.poll_events()
            fresh=replay.poll() if replay else None
            if fresh is not None:
                if (fresh.width,fresh.height)!=(width,height):raise ValueError("Recorded video geometry changed")
                renderer.upload_tile(0,fresh,height)
                display.update_tile(0,fresh,height)
            changed=display.step()
            if changed or fresh is not None or last_publish==0:window.force_draw()
            if time.monotonic()-last_publish>.25:
                state=worker.snapshot()
                playback=replay.snapshot() if replay else None
                if playback and playback["state"]=="error":raise RuntimeError(playback["error"])
                control.publish_snapshot({"enhancement":state,"playback":playback,
                                          "decoder":"offline video" if replay else "offline fixture","tiles":[],"uptime_s":0})
                if playback:state=dict(state,playback=playback)
                if args.state_file:args.state_file.write_text(json.dumps(state,indent=2))
                last_publish=time.monotonic()
            time.sleep(.005)
    finally:
        if replay:replay.close()
        worker.close();control.close();server.shutdown();window.close();temp.cleanup()


if __name__=="__main__":
    main()
