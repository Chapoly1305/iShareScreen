"""CUDA kernels for the optional fixed-shape EDSR surface runtime."""
import triton
import triton.language as tl


@triton.jit
def normalize_rgb(X, Y, SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = i // (720 * 1280)
    row = (i // 1280) % 720
    col = i % 1280
    value = tl.load(X + channel * SC + row * SH + col * SW, i < N, 0).to(tl.float32)
    tl.store(Y + i, value / 255.0, i < N)


@triton.jit
def normalize_nv12_709_full(X, Y, SH: tl.constexpr, SW: tl.constexpr,
                           N: tl.constexpr, BLOCK: tl.constexpr):
    """NV12 BT.709 full-range -> normalized RGB, matching decoder truncation.

    X is a (1080,1280) CUDA byte view: 720 luma rows followed by 360 UV rows.
    Color interpretation is explicit; this kernel does not handle limited
    range, BT.601, HDR, other dimensions, or other chroma layouts.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = i // (720 * 1280)
    row = (i // 1280) % 720
    col = i % 1280
    valid = i < N
    y = tl.load(X + row * SH + col * SW, valid, 0).to(tl.float32)
    offset = (720 + row // 2) * SH + (col // 2) * 2 * SW
    u = tl.load(X + offset, valid, 128).to(tl.float32) - 128.0
    v = tl.load(X + offset + SW, valid, 128).to(tl.float32) - 128.0
    red = y + 1.5748 * v
    green = y - (2 * .0722 * (1 - .0722) / (1 - .2126 - .0722)) * u
    green = green - (2 * .2126 * (1 - .2126) / (1 - .2126 - .0722)) * v
    blue = y + 1.8556 * u
    value = tl.where(channel == 0, red, tl.where(channel == 1, green, blue))
    # Existing NVIDIA RGBP conversion clips, then truncates to RGB8. Keep that
    # quantization before normalization; skipping it changes model inputs.
    value = tl.minimum(tl.maximum(value, 0), 255).to(tl.uint8).to(tl.float32)
    tl.store(Y + i, value / 255.0, valid)


@triton.jit
def pack_area(X, Y, SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
              CHANNELS: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
              QUANTIZE_FIRST: tl.constexpr = True):
    # Keep the legacy specialization for reproducible historical benchmarks.
    # The runtime selects False: resize floats before clipping/quantizing RGB8.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = i % CHANNELS
    col = (i // CHANNELS) % 1920
    row = i // (CHANNELS * 1920)
    sx, sy = (col * 4) // 3, (row * 4) // 3
    wx = (3 - col % 3).to(tl.float32) * .25
    wy = (3 - row % 3).to(tl.float32) * .25
    offset = channel * SC + sy * SH + sx * SW
    mask = (i < N) & (channel < 3)
    a = tl.load(X + offset, mask, 0).to(tl.float32)
    b = tl.load(X + offset + SW, mask, 0).to(tl.float32)
    c = tl.load(X + offset + SH, mask, 0).to(tl.float32)
    d = tl.load(X + offset + SH + SW, mask, 0).to(tl.float32)
    a, b, c, d = a * 255, b * 255, c * 255, d * 255
    if QUANTIZE_FIRST:
        a = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(a), 0), 255)
        b = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(b), 0), 255)
        c = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(c), 0), 255)
        d = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(d), 0), 255)
    value = (a * wx + b * (1-wx)) * wy + (c * wx + d * (1-wx)) * (1-wy)
    value = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(value), 0), 255)
    tl.store(Y + i, tl.where(channel == 3, 255, value).to(tl.uint8), i < N)
