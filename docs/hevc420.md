# HEVC 4:2:0 automatic selection

Adapted from upstream PR renegadelink/iShareScreen#31, reworked onto this
branch's `media_config` / decoder-registry design.

## Why

Apple's default stream is HEVC RExt 4:4:4, which only recent GPUs decode in
hardware. Many GPUs that lack 4:4:4 *do* hardware-decode HEVC Main 4:2:0, which
keeps HEVC's efficiency over H.264. Previously Auto went straight from 4:4:4 to
H.264 on those machines, and a manual 4:2:0 choice still used the 4:4:4
decoder's availability gate.

## Auto ladder (desktop frontend)

1. Hardware HEVC 4:4:4 (VideoToolbox, generic libav hwaccel, Intel QSV)
2. Hardware HEVC 4:2:0 (generic libav hwaccel: d3d11va / d3d12va / vaapi / cuda)
3. H.264 4:2:0

When Auto picks rung 2, the CLI sets `ISS_VIDEO_CODEC=hevc` and
`ISS_VIDEO_CHROMA=420`: the same mode-7, 4-tile offer the explicit AMD 4:2:0
path already uses. `--chroma` restricts which HEVC rung Auto may pick, and an
explicit `--codec` or `--decoder` still wins. The browser frontend stays on
H.264.

## Decoders

| name | kind | notes |
|---|---|---|
| `amd-hevc420` | hardware | AMD D3D11, explicit override only, no fallback |
| `libav-hevc420` | hardware | gated on a real 4:2:0 probe; libav's silent SW fallback disabled; 32 extra D3D11 surfaces |
| `libav-hevc420-sw` | software | floor if the hardware context fails to start |
| `vt-hevc444` | hardware | also serves 4:2:0 on macOS |

The probe decodes a 192×192 HEVC Main IDR (shared with the AMD probe) with
software fallback off. 192 px clears NVDEC's 144×144 minimum, unlike the 64×64
sample upstream used. `ISS_HEVC420=0/1` disables or forces the probe result.

## Stream handling

- **No DONL in single-picture mode.** Single-picture HEVC (`ISS_TILES_PER_FRAME=1`)
  carries no DONL fields, and `reassemble_group(has_donl=False)` handles that.
  Multi-tile behavior is unchanged.
- **Parameter sets recovered early.** Parameter sets in AP and single-NAL
  packets are used even before their timestamp group's end-of-frame marker
  arrives.
- **Payload-type logging.** `CODEC-DETECT` logs the RTP payload types
  (100 = HEVC, 123/126 = H.264) and whether DONL is present.
- **Generation reset.** Single-picture HEVC 4:2:0 resets the decoder on every
  new encoder SSRC, like AVC. Tiled HEVC keeps its debounce.
- **Diagnostics.** The log now shows media port announcements, server
  refusals, the answer's selected payload types, and the pixel-format bitmap.

## Media feedback

The heartbeat and RTCP receiver reports now also run while the startup burst
is awaited. Previously they started only after it arrived.

`ISS_MEDIA_FEEDBACK=rr` is **experimental**. It switches to upstream's routing:
an authenticated SRTCP receiver report on the audio leg in place of the PT=101
keepalive, with the video receiver report sent to the video port. The default
stays the PT=101 keepalive, because it matches what Apple's own client sends.

## Not ported

Upstream's mode-9 ("Nearby") offer, which uses direction 2, the `SW:0` feature
and a 4096 bandwidth marker. This branch already decodes HEVC 4:2:0 with the
standard mode-7 offer, so that experimental offer path is not needed.
