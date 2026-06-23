from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .timeline import PresentationFrame


def _ppm_rgb_bytes(bgra: bytes) -> bytes:
    rgb = bytearray(len(bgra) // 4 * 3)
    out = 0
    for i in range(0, len(bgra), 4):
        b = bgra[i]
        g = bgra[i + 1]
        r = bgra[i + 2]
        rgb[out : out + 3] = bytes((r, g, b))
        out += 3
    return bytes(rgb)


def write_frame_ledger(path: str | Path, frames: list[PresentationFrame]) -> None:
    write_frame_ledger_with_end(path, frames, None)


def write_frame_ledger_with_end(
    path: str | Path,
    frames: list[PresentationFrame],
    end_pts_epoch: float | None,
) -> None:
    rows = []
    for index, frame in enumerate(frames):
        next_ts = frames[index + 1].pts_epoch if index + 1 < len(frames) else end_pts_epoch
        if next_ts is None:
            next_ts = frame.pts_epoch
        duration_ms = max(0.0, (next_ts - frame.pts_epoch) * 1000.0)
        rows.append(
            {
                "frame_index": frame.frame_index,
                "pts_epoch": frame.pts_epoch,
                "width": frame.width,
                "height": frame.height,
                "duration_ms": duration_ms,
                "source_record_indices": frame.source_record_indices,
            }
        )
    Path(path).write_text(json.dumps(rows, indent=2))


class ExactTimingReplayWriter:
    def __init__(
        self,
        output_path: str | Path,
        ffmpeg_bin: str = "/opt/homebrew/bin/ffmpeg",
    ) -> None:
        self.output_path = Path(output_path)
        self.ffmpeg_bin = ffmpeg_bin
        self._tmp = tempfile.TemporaryDirectory(prefix="ss_replay_")
        self.root = Path(self._tmp.name)
        self.manifest = self.root / "frames.ffconcat"
        self._manifest_lines = ["ffconcat version 1.0"]
        self._pending_path: Path | None = None
        self._pending_pts_epoch: float | None = None
        self._closed = False

    def add_frame(self, frame_index: int, pts_epoch: float, width: int, height: int, bgra: bytes) -> None:
        ppm_path = self.root / f"frame_{frame_index:06d}.ppm"
        rgb = _ppm_rgb_bytes(bgra)
        ppm_path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb)
        if self._pending_path is not None and self._pending_pts_epoch is not None:
            duration = max(0.001, pts_epoch - self._pending_pts_epoch)
            self._manifest_lines.append(f"file '{self._pending_path}'")
            self._manifest_lines.append(f"duration {duration:.9f}")
        self._pending_path = ppm_path
        self._pending_pts_epoch = pts_epoch

    def finish(self, end_pts_epoch: float | None = None) -> None:
        if self._closed:
            return
        if self._pending_path is None:
            raise ValueError("no presentation frames to encode")
        self._manifest_lines.append(f"file '{self._pending_path}'")
        if self._pending_pts_epoch is not None and end_pts_epoch is not None:
            duration = max(0.001, end_pts_epoch - self._pending_pts_epoch)
            self._manifest_lines.append(f"duration {duration:.9f}")
            self._manifest_lines.append(f"file '{self._pending_path}'")
        self.manifest.write_text("\n".join(self._manifest_lines) + "\n")
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(self.manifest),
            "-fps_mode",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(self.output_path),
        ]
        subprocess.run(cmd, check=True)
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._tmp.cleanup()


def encode_matroska(
    frames: list[PresentationFrame],
    output_path: str | Path,
    end_pts_epoch: float | None = None,
    ffmpeg_bin: str = "/opt/homebrew/bin/ffmpeg",
) -> None:
    raise ValueError("encode_matroska is deprecated for replay output; use ExactTimingReplayWriter")
