"""Read-only device telemetry, sampled after timed frame completion."""

import ctypes as c
import time


class DeviceTelemetry:
    def __init__(self):
        self.lib = c.CDLL("libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0
        self.handle = c.c_void_p()
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            c.c_uint,
            c.POINTER(c.c_void_p),
        ]
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0, c.byref(self.handle)) == 0
        self.lib.nvmlDeviceGetClockInfo.argtypes = [
            c.c_void_p,
            c.c_uint,
            c.POINTER(c.c_uint),
        ]
        self.lib.nvmlDeviceGetPowerUsage.argtypes = [c.c_void_p, c.POINTER(c.c_uint)]
        self.lib.nvmlDeviceGetTemperature.argtypes = [
            c.c_void_p,
            c.c_uint,
            c.POINTER(c.c_uint),
        ]
        self.lib.nvmlDeviceGetCurrentClocksEventReasons.argtypes = [
            c.c_void_p,
            c.POINTER(c.c_ulonglong),
        ]

    def sample(self):
        row = dict(monotonic_ns=time.monotonic_ns())
        for name, kind in [("sm_mhz", 1), ("memory_mhz", 2), ("video_mhz", 3)]:
            value = c.c_uint()
            assert (
                self.lib.nvmlDeviceGetClockInfo(self.handle, kind, c.byref(value)) == 0
            )
            row[name] = value.value
        value = c.c_uint()
        assert self.lib.nvmlDeviceGetPowerUsage(self.handle, c.byref(value)) == 0
        row["power_w"] = value.value / 1000
        assert self.lib.nvmlDeviceGetTemperature(self.handle, 0, c.byref(value)) == 0
        row["temperature_c"] = value.value
        reasons = c.c_ulonglong()
        assert (
            self.lib.nvmlDeviceGetCurrentClocksEventReasons(
                self.handle, c.byref(reasons)
            )
            == 0
        )
        row["clock_event_reasons"] = reasons.value
        return row
