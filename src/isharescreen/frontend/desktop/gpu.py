"""WGPU planar-YUV renderer.

Three single-channel textures (Y, U, V) at full canvas resolution, one
fragment shader doing BT.709 *full-range* YUV→RGB, one full-screen
triangle. That's all.

Each tile's planes get written into a horizontal slice of the textures
via `upload_tile`; we don't allocate or compose a CPU-side full-canvas
buffer. Apple's HEVC RExt 4:4:4 source stays 4:4:4 end-to-end.

Letterboxing: `draw` scales the decoded content (`content_dims`) by a
single uniform factor to fit the target surface, then centers it, so the
on-screen aspect always matches the decoded aspect — the image is never
stretched. Bars appear (centered, symmetric) whenever the window aspect
differs from the content aspect; the window fills completely when they
match. This covers both cases: (a) the window resized to an aspect the
canvas doesn't share, and (b) the host fell back to a real frame smaller
than the advertised canvas (encoded into the top-left with the rest left
as unwritten black padding, which `uv_scale` also crops out of sampling).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import wgpu

# Opt-in cursor-overlay diagnostics. Set ISS_CURSOR_DEBUG=1 to log, on a
# throttle, why the overlay did or didn't draw each frame.
_CURSOR_DEBUG = os.environ.get("ISS_CURSOR_DEBUG") == "1"


log = logging.getLogger(__name__)


WGSL = """
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// uv_scale maps the viewport's [0,1] sampling range onto just the
// decoded-content sub-region of the LUMA texture, cropping out any black
// padding the encoder left when the real frame is smaller than the
// allocated canvas. (1,1) = sample the whole texture.
// chroma_scale is uv_scale shrunk by the chroma subsample ratio (== uv_scale
// for 4:4:4; half for 4:2:0). H.264/AVC is 4:2:0, so its half-res chroma is
// written into the top-left of the full-size U/V textures; chroma_scale makes
// the shader sample just that sub-region and the bilinear sampler upsamples
// chroma to luma resolution — no CPU 4:2:0→4:4:4 upsample needed.
struct Uniforms {
    uv_scale: vec2<f32>,
    chroma_scale: vec2<f32>,
};
@group(0) @binding(4) var<uniform> U: Uniforms;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    // One triangle that covers the viewport. UV maps so the visible
    // area [0,1]x[0,1] aligns with the texture.
    let xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(xy[i], 0.0, 1.0);
    o.uv = uv[i];
    return o;
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let luv = in.uv * U.uv_scale;
    let cuv = in.uv * U.chroma_scale;
    let y  = textureSample(y_tex, samp, luv).r;
    let cb = textureSample(u_tex, samp, cuv).r - 0.5;
    let cr = textureSample(v_tex, samp, cuv).r - 0.5;
    let r = y + 1.5748   * cr;
    let g = y - 0.187324 * cb - 0.468124 * cr;
    let b = y + 1.8556   * cb;
    return vec4<f32>(r, g, b, 1.0);
}
"""


# Biplanar variant: chroma comes from ONE rg8unorm texture carrying the
# interleaved UV plane verbatim (Apple nv24 `v is None` passthrough — the
# native VideoToolbox decoder and libav's nv24 path both emit it). Texel
# .r = Cb (U), .g = Cr (V). Deinterleaving here — a free GPU texture fetch —
# instead of on the CPU removes the single biggest cost in the live decode
# pipeline (~half a core at 4-tile/60fps; see hevc.py _LEGACY_CHROMA). Same
# BT.709 full-range matrix and uv_scale crop as the planar shader.
WGSL_BIPLANAR = """
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct Uniforms { uv_scale: vec2<f32>, };
@group(0) @binding(3) var<uniform> U: Uniforms;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(xy[i], 0.0, 1.0);
    o.uv = uv[i];
    return o;
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var uv_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv * U.uv_scale;
    let y  = textureSample(y_tex, samp, uv).r;
    let c  = textureSample(uv_tex, samp, uv);
    let cb = c.r - 0.5;
    let cr = c.g - 0.5;
    let r = y + 1.5748   * cr;
    let g = y - 0.187324 * cb - 0.468124 * cr;
    let b = y + 1.8556   * cb;
    return vec4<f32>(r, g, b, 1.0);
}
"""


# Cursor overlay: a single alpha-blended quad sampling an RGBA cursor
# pixmap, positioned in NDC by a uniform rect. Drawn over the video in the
# same pass so the host's separately-sent (enc 1104) cursor is rendered
# crisp on the client instead of relying on the local OS cursor.
WGSL_CURSOR = """
struct CurU { rect: vec4<f32>, };   // (x0, y0, x1, y1) in NDC; y0=top
@group(0) @binding(0) var<uniform> C: CurU;
@group(0) @binding(1) var cur_tex: texture_2d<f32>;
@group(0) @binding(2) var cur_samp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    // Triangle strip: TL, BL, TR, BR.
    let xs = array<f32, 4>(C.rect.x, C.rect.x, C.rect.z, C.rect.z);
    let ys = array<f32, 4>(C.rect.y, C.rect.w, C.rect.y, C.rect.w);
    let us = array<f32, 4>(0.0, 0.0, 1.0, 1.0);
    let vs_ = array<f32, 4>(0.0, 1.0, 0.0, 1.0);
    var o: VOut;
    o.pos = vec4<f32>(xs[i], ys[i], 0.0, 1.0);
    o.uv = vec2<f32>(us[i], vs_[i]);
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    // Straight alpha; the pipeline blend state composites over the video.
    return textureSample(cur_tex, cur_samp, in.uv);
}
"""


# Frame-rate-conversion (FRC) blit/interpolate: sample two full-frame RGBA
# "scene" textures (already YUV→RGB converted, cursor NOT yet applied) and
# output mix(A, B, t). t=0 → pure A, t=1 → pure B, t=0.5 → linear blend
# (the P1 interpolation baseline). The scene textures are the exact pixels
# that would have gone to the swapchain, so blitting one back (t=0/1) is a
# lossless passthrough. Interpolation happens BEFORE the cursor overlay, so
# the cursor is never blended/ghosted — see docs/frame_interpolation_*.md.
WGSL_BLIT = """
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct BlendU { t: vec4<f32>, };   // t.x = blend factor A→B
@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> B: BlendU;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(xy[i], 0.0, 1.0);
    o.uv = uv[i];
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let a = textureSample(tex_a, samp, in.uv);
    let b = textureSample(tex_b, samp, in.uv);
    return mix(a, b, B.t.x);
}
"""


class Renderer:
    """Build once, call `upload_tile` per fresh tile, then `draw`."""

    def __init__(
        self, device: wgpu.GPUDevice, surface_format: wgpu.TextureFormat,
        canvas_w: int, canvas_h: int,
    ) -> None:
        self._device = device
        self._w = canvas_w
        self._h = canvas_h

        # Real decoded-content extent within the canvas textures, grown as
        # tiles upload. 0 until the first tile lands (content_dims then
        # falls back to the full canvas). Lets draw() crop the black texture
        # padding and letterbox the real frame.
        self._content_w = 0
        self._content_h = 0

        # Two chroma layouts are supported and chosen per session from the
        # first uploaded tile (see `upload_tile`):
        #   * planar    — Y + separate U + V, three r8unorm textures (the
        #                 software / yuv444p fallback path).
        #   * biplanar  — Y r8unorm + one rg8unorm UV texture carrying the
        #                 host's interleaved chroma verbatim, deinterleaved in
        #                 the shader (Apple nv24 passthrough, the hot path —
        #                 the native VideoToolbox decoder emits it directly).
        # The planar texture set is built up front (~5 bytes/px canvas,
        # trivial) so `draw` can render the pre-cleared canvas before any
        # tile lands; the biplanar pipeline is built lazily on the first
        # nv24 tile (see `_ensure_biplanar`).
        self._mode: "str | None" = None
        # Chroma subsample ratio (chroma_width/width, chroma_height/height),
        # captured from the first tile. (1,1) for 4:4:4 (HEVC), (0.5,0.5) for
        # 4:2:0 (H.264/AVC). draw() multiplies uv_scale by this to get the
        # shader's chroma_scale, so half-res chroma sampled correctly.
        self._chroma_ratio: tuple[float, float] = (1.0, 1.0)

        tex_kw = dict(
            format=wgpu.TextureFormat.r8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            size=(canvas_w, canvas_h, 1),
        )
        self._y_tex = device.create_texture(**tex_kw)
        self._u_tex = device.create_texture(**tex_kw)
        self._v_tex = device.create_texture(**tex_kw)
        self._uv_tex = device.create_texture(
            format=wgpu.TextureFormat.rg8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            size=(canvas_w, canvas_h, 1),
        )
        # WGPU does not guarantee initialised texture contents. On Mesa
        # i915 we observed unwritten Y/U/V regions sample as bright
        # green. Pre-fill: Y=0 + UV=128 → BT.709 full-range black.
        zeros_y = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        neutral_uv = np.full((canvas_h, canvas_w), 128, dtype=np.uint8)
        neutral_uv2 = np.full((canvas_h, canvas_w, 2), 128, dtype=np.uint8)
        device.queue.write_texture(
            {"texture": self._y_tex, "origin": (0, 0, 0)}, zeros_y,
            {"offset": 0, "bytes_per_row": canvas_w},
            (canvas_w, canvas_h, 1),
        )
        for tex in (self._u_tex, self._v_tex):
            device.queue.write_texture(
                {"texture": tex, "origin": (0, 0, 0)}, neutral_uv,
                {"offset": 0, "bytes_per_row": canvas_w},
                (canvas_w, canvas_h, 1),
            )
        device.queue.write_texture(
            {"texture": self._uv_tex, "origin": (0, 0, 0)}, neutral_uv2,
            {"offset": 0, "bytes_per_row": canvas_w * 2},
            (canvas_w, canvas_h, 1),
        )
        log.info(
            "renderer: Y r8 + (U,V r8 | UV rg8) %dx%d (%.1f MB) — pre-cleared to black",
            canvas_w, canvas_h, 5 * canvas_w * canvas_h / 1e6,
        )

        sampler = device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
        )
        # 16-byte uniform: vec2 uv_scale (+8 bytes std140 tail padding).
        # Updated per-draw with the content/canvas ratio. Shared by the
        # planar and biplanar pipelines.
        self._uniform_buf = device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        device.queue.write_buffer(
            self._uniform_buf, 0,
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32).tobytes(),
        )
        tex_entry = {
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "texture": {"sample_type": wgpu.TextureSampleType.float, "view_dimension": "2d"},
        }
        uniform_entry = {"visibility": wgpu.ShaderStage.FRAGMENT,
                         "buffer": {"type": wgpu.BufferBindingType.uniform}}
        uniform_resource = {"buffer": self._uniform_buf, "offset": 0, "size": 16}
        # Captured for the lazy nv24 (biplanar) pipeline build — see
        # _ensure_biplanar. It's deferred until the first nv24 tile so platforms
        # that only ever hit the planar yuv444p path (every non-Mac build, where
        # libav emits yuv444p) never compile WGSL_BIPLANAR — whose nv24 shader
        # some GPU backends (notably d3d on Windows) can't translate.
        self._surface_format = surface_format
        self._sampler = sampler
        self._tex_entry = tex_entry
        self._uniform_entry = uniform_entry
        self._uniform_resource = uniform_resource

        # ── planar pipeline: Y + U + V (3× r8unorm) ─────────────────────
        planar_shader = device.create_shader_module(code=WGSL)
        planar_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, **tex_entry},
            {"binding": 1, **tex_entry},
            {"binding": 2, **tex_entry},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
            {"binding": 4, **uniform_entry},
        ])
        self._pipeline_planar = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[planar_layout]),
            vertex={"module": planar_shader, "entry_point": "vs"},
            fragment={
                "module": planar_shader, "entry_point": "fs",
                "targets": [{"format": surface_format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )
        self._bind_planar = device.create_bind_group(
            layout=planar_layout,
            entries=[
                {"binding": 0, "resource": self._y_tex.create_view()},
                {"binding": 1, "resource": self._u_tex.create_view()},
                {"binding": 2, "resource": self._v_tex.create_view()},
                {"binding": 3, "resource": sampler},
                {"binding": 4, "resource": uniform_resource},
            ],
        )

        # nv24 (biplanar) pipeline is built lazily on the first nv24 tile —
        # see _ensure_biplanar. Stays None on platforms that never see nv24.
        self._pipeline_biplanar = None
        self._bind_biplanar = None

        # ── cursor overlay ──────────────────────────────────────────────
        # The host sends the cursor separately (enc 1104) and we render it
        # here as a small alpha-blended quad over the video, so it stays
        # crisp and the local OS cursor can be hidden. Texture + bind group
        # are (re)built each time the shape changes; position/size come from
        # a per-draw uniform. Until the first cursor pixmap arrives,
        # `_cursor_tex` is None and the overlay draw is skipped.
        self._cursor_tex = None
        self._cursor_bind_group = None
        self._cur_w = self._cur_h = 0
        self._cur_hx = self._cur_hy = 0
        self._cursor_pos: "tuple[int, int] | None" = None  # pointer, canvas texels
        # Calibration multiplier for the cursor sprite (default 1.0,
        # ISS_CURSOR_SCALE). The base on-screen size is computed in draw() by
        # scaling the sprite with the video's uniform letterbox factor (1
        # sprite pixel = 1 content texel), so the cursor tracks the zoomed
        # content rather than staying frozen at native size.
        self._cursor_scale = 1.0
        self._cursor_sampler = device.create_sampler(
            mag_filter=wgpu.FilterMode.nearest,
            min_filter=wgpu.FilterMode.linear,
        )
        self._cursor_uniform_buf = device.create_buffer(
            size=16,  # vec4 NDC rect
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        cur_shader = device.create_shader_module(code=WGSL_CURSOR)
        self._cursor_bind_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.float,
                         "view_dimension": "2d"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
        ])
        self._cursor_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(
                bind_group_layouts=[self._cursor_bind_layout]),
            vertex={"module": cur_shader, "entry_point": "vs"},
            fragment={
                "module": cur_shader, "entry_point": "fs",
                "targets": [{
                    "format": surface_format,
                    "blend": {
                        "color": {
                            "src_factor": wgpu.BlendFactor.src_alpha,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                        "alpha": {
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                    },
                }],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_strip},
        )

        # ── frame-rate conversion (FRC) scaffolding ─────────────────────
        # Off unless the app routes frames through render_scene/present_scene
        # (gated by ISS_FRC in app.py). Two offscreen RGBA "scene" textures
        # form a 2-slot history ring: the video (YUV→RGB, no cursor) is
        # rendered into a slot, and present_scene blits mix(slotA, slotB, t)
        # to the swapchain + cursor overlay. Built lazily on first use so the
        # non-FRC path allocates nothing. Scene textures track the swapchain
        # size and are rebuilt on resize.
        self._blit_pipeline = None
        self._blit_bind_layout = None
        self._blit_uniform_buf = None
        self._blit_sampler = None
        self._frc_scene: list = [None, None]
        self._frc_scene_view: list = [None, None]
        self._frc_bg: dict = {}          # (a,b) -> bind group for the blit pass
        self._frc_scene_size: tuple = (0, 0)
        self._frc_ring_idx = 0           # next slot render_scene writes
        self._frc_have = 0               # valid scenes so far (caps at 2)

    def _ensure_biplanar(self) -> None:
        """Lazily build the nv24 (Y r8 + interleaved-UV rg8) render pipeline on
        first use. Deferred from __init__ so platforms that only ever hit the
        planar yuv444p path never compile WGSL_BIPLANAR — some GPU backends
        (notably d3d on Windows) can't translate its nv24 shader, and failing
        there for a pipeline they'll never use is pointless. Idempotent."""
        if self._pipeline_biplanar is not None:
            return
        device = self._device
        biplanar_shader = device.create_shader_module(code=WGSL_BIPLANAR)
        biplanar_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, **self._tex_entry},
            {"binding": 1, **self._tex_entry},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
            {"binding": 3, **self._uniform_entry},
        ])
        self._pipeline_biplanar = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[biplanar_layout]),
            vertex={"module": biplanar_shader, "entry_point": "vs"},
            fragment={
                "module": biplanar_shader, "entry_point": "fs",
                "targets": [{"format": self._surface_format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )
        self._bind_biplanar = device.create_bind_group(
            layout=biplanar_layout,
            entries=[
                {"binding": 0, "resource": self._y_tex.create_view()},
                {"binding": 1, "resource": self._uv_tex.create_view()},
                {"binding": 2, "resource": self._sampler},
                {"binding": 3, "resource": self._uniform_resource},
            ],
        )

    def upload_tile(self, tile_index: int, tile, slot_height: int) -> None:
        """Write tile's Y/U/V into the canvas-tile slice at row
        `tile_index * slot_height`.

        `slot_height` should be the encoder's actual CTU-padded picture
        height (typically `tile.height`, e.g. 304 rows for a 1920×1200/
        4-tile canvas). The first 3 tiles fill `slot_height` rows each;
        the last tile is bounded by `canvas_h - tile_index * slot_height`
        so we don't overrun the texture and so the bottom CTU-padding
        rows of tile 3 (Apple Screen Sharing also doesn't render those — they're
        encoder padding past `canvas_h`) are dropped from the display.
        """
        w = tile.width
        origin_y = tile_index * slot_height
        remaining = max(0, self._h - origin_y)
        rows = min(tile.height, slot_height, remaining)
        if rows <= 0:
            return
        # Lock the chroma layout to the decoder's output on the first tile.
        # A session's decode path is stable (Apple HW / VideoToolbox → nv24
        # biplanar; the software fallback → planar yuv444p), so this never
        # flips mid-stream.
        if self._mode is None:
            self._mode = "biplanar" if tile.is_nv12_passthrough else "planar"
            self._chroma_ratio = (
                tile.chroma_width / tile.width if tile.width else 1.0,
                tile.chroma_height / tile.height if tile.height else 1.0,
            )
            if self._mode == "biplanar":
                self._ensure_biplanar()
        origin = (0, origin_y, 0)
        # Grow the real decoded-content extent so draw() can crop the
        # texture's black padding and letterbox what the encoder actually
        # filled (≤ canvas when the host fell back to a smaller resolution
        # than we advertised).
        if w > self._content_w:
            self._content_w = w
        if origin_y + rows > self._content_h:
            self._content_h = origin_y + rows

        y = np.frombuffer(tile.y, dtype=np.uint8)
        if tile.y_stride == w:
            y = y[: w * tile.height].reshape(tile.height, w)
        else:
            y = y[: tile.y_stride * tile.height].reshape(
                tile.height, tile.y_stride)[:, :w]
        self._device.queue.write_texture(
            {"texture": self._y_tex, "origin": origin},
            np.ascontiguousarray(y[:rows]),
            {"offset": 0, "bytes_per_row": w},
            (w, rows, 1),
        )

        cw, ch = tile.chroma_width, tile.chroma_height
        # Same canvas-bound for chroma as for luma so the last tile's
        # padding chroma rows don't get uploaded.
        chroma_rows = min(ch, slot_height, max(0, self._h - origin_y))
        if chroma_rows <= 0:
            return

        if tile.v is None:
            # Biplanar passthrough (Apple nv24): the interleaved UV plane goes
            # straight into the rg8unorm texture — no CPU deinterleave. The
            # source row pitch is `uv_stride` bytes (≥ 2*cw); write_texture
            # consumes 2*cw bytes/row (rg8) and skips the padding tail.
            uv = np.frombuffer(tile.u, dtype=np.uint8)
            uv = uv[: tile.uv_stride * ch].reshape(ch, tile.uv_stride)
            self._device.queue.write_texture(
                {"texture": self._uv_tex, "origin": origin},
                np.ascontiguousarray(uv[:chroma_rows]),
                {"offset": 0, "bytes_per_row": tile.uv_stride},
                (cw, chroma_rows, 1),
            )
            return

        u = np.frombuffer(tile.u, dtype=np.uint8)
        v = np.frombuffer(tile.v, dtype=np.uint8)
        if tile.uv_stride == cw:
            u = u[: cw * ch].reshape(ch, cw)
            v = v[: cw * ch].reshape(ch, cw)
        else:
            u = u[: tile.uv_stride * ch].reshape(ch, tile.uv_stride)[:, :cw]
            v = v[: tile.uv_stride * ch].reshape(ch, tile.uv_stride)[:, :cw]
        self._device.queue.write_texture(
            {"texture": self._u_tex, "origin": origin},
            np.ascontiguousarray(u[:chroma_rows]),
            {"offset": 0, "bytes_per_row": cw},
            (cw, chroma_rows, 1),
        )
        self._device.queue.write_texture(
            {"texture": self._v_tex, "origin": origin},
            np.ascontiguousarray(v[:chroma_rows]),
            {"offset": 0, "bytes_per_row": cw},
            (cw, chroma_rows, 1),
        )

    def content_dims(self) -> tuple[int, int]:
        """Real decoded-frame size in canvas-texel units. Equals the full
        canvas once the encoder fills it; smaller (content pinned to the
        top-left, black padding around) when the host fell back to a
        resolution below what we advertised. Falls back to the full canvas
        before the first tile arrives."""
        cw = self._content_w or self._w
        ch = self._content_h or self._h
        return (min(cw, self._w), min(ch, self._h))

    def set_cursor_image(self, img) -> None:
        """Upload a new cursor pixmap (`_CursorImage`: RGBA8888 + hotspot) as
        the overlay texture. `img is None` clears the overlay. MUST be called
        on the render thread (creates a GPU texture)."""
        if img is None:
            self._cursor_tex = None
            self._cursor_bind_group = None
            return
        w, h = int(img.width), int(img.height)
        if w <= 0 or h <= 0:
            self._cursor_tex = None
            self._cursor_bind_group = None
            return
        tex = self._device.create_texture(
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            size=(w, h, 1),
        )
        rgba = np.frombuffer(img.rgba, dtype=np.uint8)[: w * h * 4].reshape(h, w * 4)
        # wgpu requires the source `bytes_per_row` to be a multiple of 256
        # (COPY_BYTES_PER_ROW_ALIGNMENT). A small cursor (e.g. 17px wide → 68
        # bytes/row) is unaligned, so write_texture throws and leaves the
        # overlay with no texture — the cursor never appears. Pad each row up
        # to the 256-byte stride; the copy still reads only the real w*4 bytes.
        w4 = w * 4
        bpr = ((w4 + 255) // 256) * 256
        if bpr == w4:
            data = np.ascontiguousarray(rgba)
        else:
            data = np.zeros((h, bpr), dtype=np.uint8)
            data[:, :w4] = rgba
        try:
            self._device.queue.write_texture(
                {"texture": tex, "origin": (0, 0, 0)},
                np.ascontiguousarray(data),
                {"offset": 0, "bytes_per_row": bpr},
                (w, h, 1),
            )
        except Exception as e:
            log.warning("cursor write_texture FAILED w=%d h=%d w4=%d bpr=%d "
                        "data=%r: %s", w, h, w4, bpr,
                        getattr(data, "shape", None), e)
            self._cursor_tex = None
            self._cursor_bind_group = None
            return
        self._cursor_tex = tex
        self._cur_w, self._cur_h = w, h
        self._cur_hx, self._cur_hy = int(img.hotspot_x), int(img.hotspot_y)
        self._cursor_bind_group = self._device.create_bind_group(
            layout=self._cursor_bind_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._cursor_uniform_buf,
                                            "offset": 0, "size": 16}},
                {"binding": 1, "resource": tex.create_view()},
                {"binding": 2, "resource": self._cursor_sampler},
            ],
        )

    def set_cursor_pos(self, pos: "tuple[int, int] | None") -> None:
        """Current pointer position in canvas-texel coords (same space as the
        decoded content), or None when the pointer is off the content."""
        self._cursor_pos = pos

    def set_cursor_scale(self, scale: float) -> None:
        """Calibration multiplier for the cursor overlay (default 1.0). The
        sprite's base on-screen size is `pixmap_size * letterbox_scale` (set in
        draw, so 1 sprite pixel = 1 content texel and the cursor tracks the
        video zoom); this multiplier (ISS_CURSOR_SCALE) tunes it further."""
        self._cursor_scale = max(0.1, float(scale))

    def _write_video_uniform(self) -> None:
        """Write the YUV→RGB shader's uv_scale/chroma_scale uniform from the
        current content/canvas ratio. Shared by draw() and render_scene().

        uv_scale crops texture sampling to the decoded-content sub-rect
        (top-left of the textures); padding outside content_dims is never
        sampled. chroma_scale = uv_scale shrunk by the chroma subsample ratio
        (== uv_scale for 4:4:4, half for 4:2:0), so half-res chroma written
        into the top-left of the full-size U/V textures is sampled correctly
        and the bilinear sampler upsamples it — no CPU 4:2:0→4:4:4 upsample."""
        cw, ch = self.content_dims()
        ux = cw / self._w if self._w else 1.0
        uy = ch / self._h if self._h else 1.0
        crx, cry = self._chroma_ratio
        self._device.queue.write_buffer(
            self._uniform_buf, 0,
            np.array([ux, uy, ux * crx, uy * cry], dtype=np.float32).tobytes(),
        )

    def _encode_video(self, rpass, target_w: int, target_h: int) -> None:
        """Encode the video (YUV→RGB) full-screen triangle into `rpass`,
        filling the whole target. Caller writes the uniform first.

        Fills the whole window (pre-rework baseline). The window is
        aspect-locked to the content (see app.py) so a free resize fills with
        no distortion; `uv_scale` (via _write_video_uniform) still crops
        sampling to the decoded sub-rect so texture padding is never shown."""
        if self._mode == "biplanar":
            rpass.set_pipeline(self._pipeline_biplanar)
            rpass.set_bind_group(0, self._bind_biplanar)
        else:
            rpass.set_pipeline(self._pipeline_planar)
            rpass.set_bind_group(0, self._bind_planar)
        rpass.set_viewport(0.0, 0.0, float(target_w), float(target_h), 0.0, 1.0)
        rpass.draw(3)

    def _encode_cursor(self, rpass, target_w: int, target_h: int) -> None:
        """Encode the cursor overlay quad into `rpass`, alpha-blended over
        whatever the pass already holds. No-op until a cursor pixmap AND a
        pointer position are known. Shared by draw() and present_scene().

        Placement: content texels map onto the full window per-axis (fill);
        the sprite is sized by the smaller axis so it stays square (never
        stretched). `_cursor_scale` is the ISS_CURSOR_SCALE calibration."""
        cw, ch = self.content_dims()
        sclx = target_w / cw if cw > 0 else 1.0
        scly = target_h / ch if ch > 0 else 1.0
        _overlay_ok = (
            self._cursor_tex is not None and self._cursor_bind_group is not None
            and self._cursor_pos is not None and target_w > 0 and target_h > 0)
        if _CURSOR_DEBUG:
            self._dbg_n = getattr(self, "_dbg_n", 0) + 1
            if self._dbg_n % 30 == 0:
                log.info(
                    "cursor-overlay draw=%s tex=%s bind=%s pos=%s size=%dx%d "
                    "sprite=%dx%d scale=%.2f",
                    _overlay_ok, self._cursor_tex is not None,
                    self._cursor_bind_group is not None, self._cursor_pos,
                    target_w, target_h, self._cur_w, self._cur_h,
                    self._cursor_scale,
                )
        if not _overlay_ok:
            return
        cx, cy = self._cursor_pos
        d = min(sclx, scly) * self._cursor_scale
        px = cx * sclx
        py = cy * scly
        sw = self._cur_w * d
        sh = self._cur_h * d
        sx = px - self._cur_hx * d
        sy = py - self._cur_hy * d
        x0 = sx / target_w * 2.0 - 1.0
        x1 = (sx + sw) / target_w * 2.0 - 1.0
        y0 = 1.0 - sy / target_h * 2.0
        y1 = 1.0 - (sy + sh) / target_h * 2.0
        self._device.queue.write_buffer(
            self._cursor_uniform_buf, 0,
            np.array([x0, y0, x1, y1], dtype=np.float32).tobytes(),
        )
        rpass.set_pipeline(self._cursor_pipeline)
        rpass.set_bind_group(0, self._cursor_bind_group)
        rpass.set_viewport(0.0, 0.0, float(target_w), float(target_h), 0.0, 1.0)
        rpass.draw(4)

    # ── frame-rate conversion (opt-in; see docs/frame_interpolation_*.md) ──
    def _ensure_frc(self, target_w: int, target_h: int) -> None:
        """Lazily build the blit/interpolate pipeline and (re)allocate the
        2-slot scene ring at the swapchain size. Idempotent; rebuilds the
        scene textures + bind groups on resize."""
        device = self._device
        if self._blit_pipeline is None:
            shader = device.create_shader_module(code=WGSL_BLIT)
            self._blit_sampler = device.create_sampler(
                mag_filter=wgpu.FilterMode.nearest,
                min_filter=wgpu.FilterMode.nearest,
            )
            self._blit_uniform_buf = device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._blit_bind_layout = device.create_bind_group_layout(entries=[
                {"binding": 0, **self._tex_entry},
                {"binding": 1, **self._tex_entry},
                {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
                {"binding": 3, **self._uniform_entry},
            ])
            self._blit_pipeline = device.create_render_pipeline(
                layout=device.create_pipeline_layout(
                    bind_group_layouts=[self._blit_bind_layout]),
                vertex={"module": shader, "entry_point": "vs"},
                fragment={
                    "module": shader, "entry_point": "fs",
                    "targets": [{"format": self._surface_format}],
                },
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            )
        if (self._frc_scene_size == (target_w, target_h)
                and self._frc_scene[0] is not None):
            return
        self._frc_scene = [None, None]
        self._frc_scene_view = [None, None]
        for i in range(2):
            tex = device.create_texture(
                format=self._surface_format,
                usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                       | wgpu.TextureUsage.TEXTURE_BINDING),
                size=(max(1, target_w), max(1, target_h), 1),
            )
            self._frc_scene[i] = tex
            self._frc_scene_view[i] = tex.create_view()
        self._frc_bg = {}
        for a in range(2):
            for b in range(2):
                self._frc_bg[(a, b)] = device.create_bind_group(
                    layout=self._blit_bind_layout,
                    entries=[
                        {"binding": 0, "resource": self._frc_scene_view[a]},
                        {"binding": 1, "resource": self._frc_scene_view[b]},
                        {"binding": 2, "resource": self._blit_sampler},
                        {"binding": 3, "resource": {
                            "buffer": self._blit_uniform_buf,
                            "offset": 0, "size": 16}},
                    ],
                )
        self._frc_scene_size = (target_w, target_h)
        self._frc_ring_idx = 0
        self._frc_have = 0

    def render_scene(self, target_w: int, target_h: int) -> int:
        """FRC: render the current video frame (YUV→RGB, NO cursor) into the
        next scene-ring slot; return its index. The scene holds exactly the
        pixels draw() would put on screen minus the cursor, so present_scene
        can blit it back losslessly or blend two of them. Call once per real
        decoded frame."""
        self._ensure_frc(target_w, target_h)
        idx = self._frc_ring_idx
        self._frc_ring_idx ^= 1
        self._write_video_uniform()
        encoder = self._device.create_command_encoder()
        rpass = encoder.begin_render_pass(color_attachments=[{
            "view": self._frc_scene_view[idx],
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0, 0, 0, 1),
        }])
        self._encode_video(rpass, target_w, target_h)
        rpass.end()
        self._device.queue.submit([encoder.finish()])
        if self._frc_have < 2:
            self._frc_have += 1
        return idx

    def present_scene(
        self, target_view: wgpu.GPUTextureView, target_w: int, target_h: int,
        idx_a: int, idx_b: int, t: float,
    ) -> None:
        """FRC: composite mix(scene[a], scene[b], t) into `target_view`, then
        the cursor overlay on top. t=0 → scene a (lossless passthrough), t=0.5
        → linear blend. Cursor is applied AFTER the blend so it never ghosts."""
        self._ensure_frc(target_w, target_h)
        self._device.queue.write_buffer(
            self._blit_uniform_buf, 0,
            np.array([float(t), 0.0, 0.0, 0.0], dtype=np.float32).tobytes(),
        )
        encoder = self._device.create_command_encoder()
        rpass = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0, 0, 0, 1),
        }])
        rpass.set_pipeline(self._blit_pipeline)
        rpass.set_bind_group(0, self._frc_bg[(idx_a, idx_b)])
        rpass.set_viewport(0.0, 0.0, float(target_w), float(target_h), 0.0, 1.0)
        rpass.draw(3)
        self._encode_cursor(rpass, target_w, target_h)
        rpass.end()
        self._device.queue.submit([encoder.finish()])

    def frc_scenes_ready(self) -> int:
        """Number of valid scenes currently in the ring (0..2)."""
        return self._frc_have

    def draw(
        self, target_view: wgpu.GPUTextureView,
        target_w: int, target_h: int,
    ) -> None:
        """Render one frame into `target_view`, an attachment of size
        `target_w × target_h` surface pixels.

        The decoded content (`content_dims` — the top-left sub-rect of the
        textures the encoder actually filled) is scaled *uniformly* to the
        largest size that fits the target, then centered. Uniform scale
        preserves the decoded aspect ratio, so the image is never stretched;
        a window that matches the content aspect fills completely, and any
        leftover shows as symmetric black bars — letterbox when the window
        is taller than the content aspect, pillarbox when wider — matching
        Apple's viewer. `uv_scale` separately crops sampling to the content
        sub-rect, so black texture padding (host fell back to a frame
        smaller than the advertised canvas) is never sampled."""
        self._write_video_uniform()
        encoder = self._device.create_command_encoder()
        rpass = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0, 0, 0, 1),
        }])
        self._encode_video(rpass, target_w, target_h)
        self._encode_cursor(rpass, target_w, target_h)
        rpass.end()
        self._device.queue.submit([encoder.finish()])


__all__ = ["Renderer", "WGSL"]
