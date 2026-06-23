"""No-root in-process statistical profiler for the headless client.

py-spy needs root on macOS (SIP). This samples `sys._current_frames()` from a
daemon thread, folds leaf frames, and splits on-CPU work from blocking waits
(recvfrom / queue.get / lock.acquire / select). Run exactly like `iss`:

    echo "$PW" | python tools/profile_headless.py --host H -u U \
        --password-stdin --headless --auto-quit-secs 44

Prints a leaf-function histogram (on-CPU only) + a hot-stack list at exit.
"""
import os
import sys
import time
import threading
import collections

_leaf_cpu = collections.Counter()
_leaf_all = collections.Counter()
_stack = collections.Counter()
_stop = threading.Event()

# Leaves that mean "parked, not burning CPU".
_BLOCK_NAMES = {
    "recvfrom", "recv", "recv_into", "read", "readinto", "get", "wait",
    "acquire", "select", "poll", "sleep", "join", "_worker", "accept",
    "__enter__", "_recv_loop", "run_forever", "_run_once", "epoll",
}
_BLOCK_FILES = {"socket.py", "queue.py", "threading.py", "selectors.py",
                "selector_events.py", "base_events.py"}


def _is_blocking(fn, name):
    return name in _BLOCK_NAMES or fn in _BLOCK_FILES


def _sampler():
    while not _stop.is_set():
        for f in list(sys._current_frames().values()):
            if f is None:
                continue
            fn = os.path.basename(f.f_code.co_filename)
            name = f.f_code.co_name
            _leaf_all[(fn, name)] += 1
            if not _is_blocking(fn, name):
                _leaf_cpu[(fn, name)] += 1
                # fold a short stack for context
                keys = []
                g = f
                d = 0
                while g is not None and d < 6:
                    keys.append(g.f_code.co_name)
                    g = g.f_back
                    d += 1
                _stack[" <- ".join(keys)] += 1
        time.sleep(0.008)  # ~125 Hz


def _dump():
    print("\n@@@@ PROFILE @@@@")
    cpu_total = sum(_leaf_cpu.values()) or 1
    all_total = sum(_leaf_all.values()) or 1
    print(f"@@ on-CPU leaf samples={cpu_total}  all-thread samples={all_total}  "
          f"(on-CPU fraction of sampled frames={100*cpu_total/all_total:.0f}%)")
    print("@@ ---- ON-CPU LEAF FUNCTIONS (top 30) ----")
    for (fn, name), c in _leaf_cpu.most_common(30):
        print(f"@@ {c:7d} {100*c/cpu_total:5.1f}%  {name:34s} {fn}")
    print("@@ ---- HOT ON-CPU STACKS (top 18) ----")
    for s, c in _stack.most_common(18):
        print(f"@@ {c:7d} {100*c/cpu_total:5.1f}%  {s}")
    print("@@@@ END PROFILE @@@@")


def main():
    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    from isharescreen.cli import main as iss_main
    try:
        return iss_main(sys.argv[1:])
    finally:
        _stop.set()
        t.join()
        _dump()


if __name__ == "__main__":
    sys.exit(main())
