from fractions import Fraction
from benchmarks.super_resolution.export_dvr_clips import frame_interval


def test_first_frame_is_not_before_request_and_interval_is20seconds():
    for fps in ["30/1", "60/1"]:
        for start in [0, 103.73666665000002, 1728.5, 1.0]:
            a, b = frame_interval(start, 20, fps)
            rate = Fraction(fps)
            assert Fraction(a, 1) / rate >= Fraction(str(start))
            assert Fraction(a, 1) / rate - Fraction(str(start)) < 1 / rate
            assert (b - a) / rate == 20


def test_aligned_frame_boundary_is_retained():
    assert frame_interval("10.5", 20, "30/1") == (315, 915)
    assert frame_interval("10.5", 20, "60/1") == (630, 1830)
