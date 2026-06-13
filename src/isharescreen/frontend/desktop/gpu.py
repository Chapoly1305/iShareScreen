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
// decoded-content sub-region of the textures, cropping out any black
// padding the encoder left when the real frame is smaller than the
// allocated canvas. (1,1) = sample the whole texture.
struct Uniforms {
    uv_scale: vec2<f32>,
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
    let uv = in.uv * U.uv_scale;
    let y  = textureSample(y_tex, samp, uv).r;
    let cb = textureSample(u_tex, samp, uv).r - 0.5;
    let cr = textureSample(v_tex, samp, uv).r - 0.5;
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


class Renderer:
    """Build once, call `upload_tile` per fresh tile, then `draw`."""

    def __init__(
        self, device: wgpu.GPUDevice, surface_format: wgpu.TextureFormat,
        canvas_w: int, canvas_h: int,
    ) -> None:
        self._device = device
        self._w = canvas_w
        self._h = canvas_h
        # Real decoded-content extent within the canvas textures, grown
        # as tiles upload. 0 until the first tile lands (content_dims
        # then falls back to the full canvas). Lets draw() crop the
        # black texture padding and letterbox the real frame.
        self._content_w = 0
        self._content_h = 0

        tex_kw = dict(
            format=wgpu.TextureFormat.r8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            size=(canvas_w, canvas_h, 1),
        )
        self._y_tex = device.create_texture(**tex_kw)
        self._u_tex = device.create_texture(**tex_kw)
        self._v_tex = device.create_texture(**tex_kw)
        # WGPU does not guarantee initialised texture contents. On Mesa
        # i915 we observed unwritten Y/U/V regions sample as bright
        # green. Pre-fill: Y=0 + UV=128 → BT.709 full-range black.
        zeros_y = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        neutral_uv = np.full((canvas_h, canvas_w), 128, dtype=np.uint8)
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
        log.info(
            "renderer: 3x r8unorm %dx%d (%.1f MB total) — pre-cleared to black",
            canvas_w, canvas_h, 3 * canvas_w * canvas_h / 1e6,
        )

        sampler = device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
        )
        # 16-byte uniform: vec2 uv_scale (+8 bytes std140 tail padding).
        # Updated per-draw with the content/canvas ratio.
        self._uniform_buf = device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        device.queue.write_buffer(
            self._uniform_buf, 0,
            np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
        )
        shader = device.create_shader_module(code=WGSL)
        tex_entry = {
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "texture": {"sample_type": wgpu.TextureSampleType.float, "view_dimension": "2d"},
        }
        bind_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, **tex_entry},
            {"binding": 1, **tex_entry},
            {"binding": 2, **tex_entry},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
            {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[bind_layout]),
            vertex={"module": shader, "entry_point": "vs"},
            fragment={
                "module": shader, "entry_point": "fs",
                "targets": [{"format": surface_format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )
        self._bind_group = device.create_bind_group(
            layout=bind_layout,
            entries=[
                {"binding": 0, "resource": self._y_tex.create_view()},
                {"binding": 1, "resource": self._u_tex.create_view()},
                {"binding": 2, "resource": self._v_tex.create_view()},
                {"binding": 3, "resource": sampler},
                {"binding": 4, "resource": {"buffer": self._uniform_buf,
                                            "offset": 0, "size": 16}},
            ],
        )

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
        # Render scale for the cursor sprite = the local display's content
        # scale (Retina factor). A cursor is a UI element drawn at the local
        # display resolution, so its size must NOT scale with the video
        # letterbox (which shrinks the remote canvas into the window).
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
        origin = (0, origin_y, 0)
        # Grow the real decoded-content extent so draw() can crop the
        # texture's black padding and letterbox what the encoder
        # actually filled (≤ canvas when the host fell back to a
        # smaller resolution than we advertised).
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

        if tile.v is None:  # NV-style biplanar — not produced by our decoder path
            return
        cw, ch = tile.chroma_width, tile.chroma_height
        # Same canvas-bound for chroma as for luma so the last tile's
        # padding chroma rows don't get uploaded.
        chroma_rows = min(ch, slot_height, max(0, self._h - origin_y))
        if chroma_rows <= 0:
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
        rgba = np.frombuffer(img.rgba, dtype=np.uint8)[: w * h * 4].reshape(h, w, 4)
        self._device.queue.write_texture(
            {"texture": tex, "origin": (0, 0, 0)},
            np.ascontiguousarray(rgba),
            {"offset": 0, "bytes_per_row": w * 4},
            (w, h, 1),
        )
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
        """Local display content scale (Retina factor) — the cursor sprite is
        drawn at `pixmap_size * scale` surface pixels, matching the size the
        OS would show it, independent of the video letterbox."""
        self._cursor_scale = max(0.1, float(scale))

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
        cw, ch = self.content_dims()
        # uv_scale crops texture sampling to the decoded content sub-rect
        # (top-left of the textures); padding outside content_dims is never
        # sampled. Per-axis: the real frame can be smaller than the canvas
        # on either axis independently.
        ux = cw / self._w if self._w else 1.0
        uy = ch / self._h if self._h else 1.0
        self._device.queue.write_buffer(
            self._uniform_buf, 0,
            np.array([ux, uy, 0.0, 0.0], dtype=np.float32).tobytes(),
        )
        # Aspect-preserving fit: one uniform scale (same factor on both
        # axes) sized to the largest content rect that fits the target,
        # then centered. Preserves the decoded aspect — never stretched —
        # and the leftover falls out as symmetric letter/pillarbox bars.
        if cw > 0 and ch > 0 and target_w > 0 and target_h > 0:
            scale = min(target_w / cw, target_h / ch)
            vw, vh = cw * scale, ch * scale
        else:
            scale = 1.0
            vw, vh = float(target_w), float(target_h)
        viewport = ((target_w - vw) * 0.5, (target_h - vh) * 0.5, vw, vh)
        encoder = self._device.create_command_encoder()
        rpass = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0, 0, 0, 1),
        }])
        rpass.set_pipeline(self._pipeline)
        rpass.set_bind_group(0, self._bind_group)
        rpass.set_viewport(*viewport, 0.0, 1.0)
        rpass.draw(3)

        # Cursor overlay: place the pixmap at the pointer (canvas texels),
        # mapped through the same letterbox transform as the video. The
        # content sub-rect maps onto `viewport` at uniform `scale`, so a
        # canvas coord c → screen px = viewport_offset + c * scale.
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
        if _overlay_ok:
            cx, cy = self._cursor_pos
            vx, vy = viewport[0], viewport[1]
            d = self._cursor_scale
            # Pointer maps through the letterbox; the sprite is sized to the
            # local display (a cursor must not shrink with the video zoom).
            px = vx + cx * scale
            py = vy + cy * scale
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

        rpass.end()
        self._device.queue.submit([encoder.finish()])


__all__ = ["Renderer", "WGSL"]
