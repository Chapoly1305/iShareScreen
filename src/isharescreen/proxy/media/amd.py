"""Windows AMD HEVC Main 4:2:0 capability probe."""
from functools import lru_cache
import logging

# Synthetic 192x192 gray HEVC Main IDR encoded with libx265, no user content.
_SAMPLE = bytes.fromhex(
    "0000000140010c01ffff01600000030090000003000003003c928090000000014201"
    "0101600000030090000003000003003ca01820305964a924cae680800000030080"
    "00000f04000000014401c172b422400000012801af0860f97f5216865ce8cd206c907508000dc8"
)


def hardware_accel(*, defer_download=False):
    from av.codec.hwaccel import HWAccel
    return HWAccel("d3d11va", options={"vendor_id": "0x1002"}, allow_software_fallback=False,
                   is_hw_owned=defer_download)


@lru_cache(maxsize=1)
def available():
    import av
    from .decode_common import context_is_hwaccel
    try:
        ctx = av.CodecContext.create("hevc", "r", hwaccel=hardware_accel())
        frames = list(ctx.decode(av.Packet(_SAMPLE))) + list(ctx.decode(None))
        return bool(frames) and context_is_hwaccel(ctx) is True
    except Exception as error:
        logging.getLogger(__name__).info("AMD HEVC 4:2:0 unavailable: %s", error)
        return False
