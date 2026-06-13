from __future__ import annotations


class Framebuffer:
    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.pixels = bytearray()

    def ensure_size(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        if self.width == width and self.height == height and len(self.pixels) == width * height * 4:
            return
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height * 4)

    def apply_raw_rect(self, x: int, y: int, width: int, height: int, pixels: bytes) -> None:
        if width <= 0 or height <= 0:
            return
        self.ensure_size(max(self.width, x + width), max(self.height, y + height))
        stride = width * 4
        for row in range(height):
            src_off = row * stride
            dst_off = ((y + row) * self.width + x) * 4
            self.pixels[dst_off : dst_off + stride] = pixels[src_off : src_off + stride]

    def snapshot_bgra(self) -> bytes:
        return bytes(self.pixels)
