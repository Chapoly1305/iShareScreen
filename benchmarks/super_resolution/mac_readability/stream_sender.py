"""Real Mac-to4090 replay over existing SSH authentication with shared pacing.

Frozen actual-Mac HEVC packets follow their recorded timestamps. Exact source
supplements are encoded anew on this Mac after a100ms stability wait. Forward
application framing plus an80byte/fragment overhead allowance is paced at600Kbps;
this is not a claim that every physical TCP/VPN window is below600Kbps.
"""

import argparse
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import queue
import re
import shlex
import struct
import subprocess
import threading
import time

import av
import cv2

try:
    from .component_codec import encode
    from .detail_sender_probe import save, sha
    from .stream_protocol import fragments, KINDS, FRAGMENT, HEADER
except ImportError:
    from component_codec import encode
    from detail_sender_probe import save, sha
    from stream_protocol import fragments, KINDS, FRAGMENT, HEADER


def replay_ready_s(group_index, dts_s, pts_us, idle_tail_s):
    """Keep packet spacing; add idle time only after each three-second state."""
    return (
        group_index * 2 * (3 + idle_tail_s)
        + dts_s
        + (idle_tail_s if pts_us >= 3000000 else 0)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", choices=["bypass", "enhanced"], required=True)
    p.add_argument("--policy", choices=["pixel", "hold", "source"], default="pixel")
    p.add_argument("--source-weight", type=float, choices=[0.25, 1.0], default=1.0)
    p.add_argument("--refresh-source-mix", action="store_true")
    p.add_argument(
        "--idle-tail-s",
        type=float,
        choices=[0.0, 1.0],
        default=0.0,
        help="Auxiliary latency probe only: hold each source after its original3s packet sequence",
    )
    p.add_argument("--groups", nargs="*")
    a = p.parse_args()
    assert re.fullmatch("[a-z0-9_-]+", a.run_name)
    cv2.setNumThreads(1)
    out = a.root / "stream-runs" / a.run_name
    out.mkdir(parents=True, exist_ok=False)
    codec_dir = a.root / "dense-codec025"
    codec = json.loads((codec_dir / "MANIFEST.json").read_text())
    controls = json.loads((a.root / "precodec-v1/MANIFEST.json").read_text())
    assert controls["complete"]
    runs = [
        r
        for r in codec["runs"]
        if r["height"] == 720 and (not a.groups or r["group"] in a.groups)
    ]
    assert runs and (not a.groups or {r["group"] for r in runs} == set(a.groups))
    counter = itertools.count(1)
    events = []
    states = []
    raw_sources = {}
    ordinal = itertools.count()

    def message(kind, generation, payload):
        ident = next(counter)
        return ident, fragments(KINDS[kind], generation, ident, payload)

    def add(ready, priority, kind, generation, payload):
        heapq.heappush(
            events, (ready, priority, next(ordinal), kind, generation, payload)
        )

    for group_index, run in enumerate(runs):
        path = codec_dir / run["path"]
        assert sha(path) == run["sha256"]
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            clip_start = group_index * 2 * (3 + a.idle_tail_s)
            add(
                clip_start,
                0,
                "config",
                0,
                json.dumps(
                    dict(
                        clip=group_index,
                        extradata=base64.b64encode(
                            stream.codec_context.extradata
                        ).decode(),
                    )
                ).encode(),
            )
            for state_index, sid in enumerate(run["state_ids"]):
                generation = group_index * 2 + state_index + 1
                row = next(
                    x
                    for x in controls["images"]
                    if x["state_id"] == sid and x["height"] == 1080
                )
                rp = a.root / "precodec-v1" / row["path"]
                assert sha(rp) == row["sha256"]
                raw_sources[generation] = rp.read_bytes()
                state = dict(
                    generation=generation,
                    state_id=sid,
                    group=run["group"],
                    ready_s=clip_start + state_index * (3 + a.idle_tail_s),
                    raw_sha256=row["sha256"],
                )
                states.append(state)
                add(state["ready_s"], 1, "source", generation, state)
                add(state["ready_s"] + 0.1, 2, "encode", generation, state)
            for packet in container.demux(stream):
                if packet.dts is None or not packet.size:
                    continue
                pts = round(float(packet.pts * packet.time_base) * 1000000)
                ready = replay_ready_s(
                    group_index,
                    float(packet.dts * packet.time_base),
                    pts,
                    a.idle_tail_s,
                )
                generation = group_index * 2 + (1 if pts < 3000000 else 2)
                add(
                    max(clip_start, ready),
                    3,
                    "video",
                    generation,
                    struct.pack(">q", pts) + bytes(packet),
                )
    remote_root = "/home/chen/iShareScreen"
    remote_output = (
        remote_root
        + "/benchmarks/super_resolution/experiments/mac-readability-v2-20261102/stream-runs/"
        + a.run_name
    )
    command = (
        "cd "
        + shlex.quote(remote_root)
        + " && source dev-environment.sh && python -m benchmarks.super_resolution.mac_readability.stream_receiver --project "
        + shlex.quote(remote_root)
        + " --output "
        + shlex.quote(remote_output)
        + " --mode "
        + shlex.quote(a.mode)
        + " --policy "
        + shlex.quote(a.policy)
        + " --source-weight "
        + str(a.source_weight)
    )
    if a.refresh_source_mix:
        command += " --refresh-source-mix"
    stderr = (out / "receiver-stderr.log").open("wb")
    process = subprocess.Popen(
        [
            "ssh",
            "-T",
            "-oBatchMode=yes",
            "-oStrictHostKeyChecking=yes",
            "-oConnectTimeout=8",
            "chen@10.193.124.90",
            command,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        bufsize=0,
    )
    acknowledgments = []
    mailbox = queue.Queue()
    incoming_bytes = 0

    def read_acks():
        nonlocal incoming_bytes
        try:
            while True:
                line = process.stdout.readline()
                received = time.monotonic_ns()
                if not line:
                    raise EOFError("Receiver stdout ended")
                incoming_bytes += len(line)
                record = json.loads(line)
                record["sender_received_ns"] = received
                acknowledgments.append(record)
                mailbox.put(record)
        except BaseException as exc:
            mailbox.put(dict(event="reader_error", error=repr(exc)))

    threading.Thread(target=read_acks, daemon=True).start()
    result = dict(
        complete=False,
        scope=__doc__,
        run_name=a.run_name,
        mode=a.mode,
        policy=a.policy,
        source_weight=a.source_weight,
        refresh_source_mix=a.refresh_source_mix,
        remote_output=remote_output,
        ssh_pid=process.pid,
        idle_tail_s=a.idle_tail_s,
        quality_scope="Auxiliary idle-tail latency probe; do not replace primary confirmationquality"
        if a.idle_tail_s
        else "Original3secondsource campaign",
        codec_manifest_sha256=sha(codec_dir / "MANIFEST.json"),
        controls_sha256=sha(a.root / "precodec-v1/MANIFEST.json"),
        source_sha256=controls["source_sha256"],
        states=states,
        rate_bps=600000,
        overhead_allowance_per_fragment=80,
        source_semantics="Scheduled replay transitions from frozen Mac recordings; not live desktop input",
        script_sha256=sha(Path(__file__)),
        sent=[],
        encodes=[],
        cancelled_detail_fragments=0,
        acknowledgments=acknowledgments,
        limits=[
            "Forward pacing covers application framing plus estimated overhead, not measured physicalwire enforcement",
            "SSH stdout feedback counted separately; no network loss injection",
            "Packet DTS is replay availability; no live VideoToolbox encode scheduling",
        ],
        test_used=False,
    )
    save(out / "RESULTS.json", result)
    encoder = ThreadPoolExecutor(max_workers=1)
    pending_encodes = []
    nettop_stop = threading.Event()
    nettop_thread = None
    nettop_samples = []

    def sample_nettop():
        while not nettop_stop.is_set():
            begin = time.monotonic_ns()
            try:
                # Finite subprocesses exit normally and flush their CSV output.
                # Killing an infinite nettop process lost stdio-buffered samples.
                sample = subprocess.run(
                    [
                        "/usr/bin/nettop",
                        "-P",
                        "-n",
                        "-x",
                        "-L",
                        "2",
                        "-s",
                        "1",
                        "-p",
                        str(process.pid),
                        "-J",
                        "bytes_in,bytes_out",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                nettop_samples.append(
                    dict(
                        start_ns=begin,
                        end_ns=time.monotonic_ns(),
                        returncode=sample.returncode,
                        stdout=sample.stdout,
                        stderr=sample.stderr,
                    )
                )
            except subprocess.TimeoutExpired:
                nettop_samples.append(dict(start_ns=begin, error="nettop timeout"))
            save(out / "NETTOP.json", nettop_samples)

    def wait_event(name, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                item = mailbox.get(
                    timeout=min(1, max(0.01, deadline - time.monotonic()))
                )
            except queue.Empty:
                if process.poll() is not None:
                    raise RuntimeError("Receiver exited " + str(process.returncode))
                continue
            if item["event"] == name:
                return item
            if item["event"] == "reader_error":
                raise RuntimeError(item["error"])
        raise TimeoutError(name)

    def write_chunks(chunks):
        # FileIO.write may perform short writes on a pipe.
        for chunk in chunks:
            view = memoryview(chunk)
            while view:
                n = process.stdin.write(view)
                if not n:
                    raise BrokenPipeError("Short pipe write")
                view = view[n:]

    def ping():
        t1 = time.monotonic_ns()
        ident, chunks = message("ping", 0, struct.pack(">Q", t1))
        write_chunks(chunks)
        answer = wait_event("pong", 10)
        assert answer["id"] == ident
        return answer

    def encode_source(gen):
        begin = time.monotonic_ns()
        packet, counts = encode(raw_sources[gen], 1920, 1080)
        end = time.monotonic_ns()
        return packet, dict(
            generation=gen,
            start_ns=begin,
            finished_ns=end,
            elapsed_ms=(end - begin) / 1e6,
            payload_bytes=len(packet),
            sha256=hashlib.sha256(packet).hexdigest(),
            **counts,
        )

    try:
        result["ready"] = wait_event("ready", 120)
        nettop_thread = threading.Thread(target=sample_nettop, daemon=True)
        nettop_thread.start()
        result["clock_before"] = [ping() for _ in range(9)]
        start = time.monotonic_ns()
        result["replay_start_ns"] = start
        controls_q = deque()
        video_q = deque()
        detail_q = deque()
        current = 0
        next_slot = time.monotonic()
        last_save = 0.0
        sent_budget = 0
        finish = 2 * (3 + a.idle_tail_s) * len(runs)
        while events or controls_q or video_q or detail_q or pending_encodes:
            if process.poll() is not None:
                raise RuntimeError("Receiver died during replay")
            now = (time.monotonic_ns() - start) / 1e9
            if now > finish + 15:
                raise TimeoutError("Replay exceeded bounded drain window")
            while events and events[0][0] <= now:
                ready, priority, order, kind, gen, payload = heapq.heappop(events)
                if kind == "encode":
                    pending_encodes.append((gen, encoder.submit(encode_source, gen)))
                    continue
                if kind == "source":
                    current = gen
                    result["cancelled_detail_fragments"] += len(detail_q)
                    detail_q.clear()
                    payload = json.dumps(
                        dict(payload, source_ns=start + round(ready * 1e9))
                    ).encode()
                ident, chunks = message(kind, gen, payload)
                target = video_q if kind == "video" else controls_q
                target.extend((kind, gen, ident, chunk, ready) for chunk in chunks)
            for gen, future in pending_encodes[:]:
                if future.done():
                    packet, record = future.result()
                    result["encodes"].append(record)
                    pending_encodes.remove((gen, future))
                    if gen == current:
                        ident, chunks = message("detail", gen, packet)
                        detail_q.extend(
                            ("detail", gen, ident, chunk, now) for chunk in chunks
                        )
                    else:
                        record["discarded_before_send"] = True
            target = controls_q or video_q or detail_q
            if target:
                kind, gen, ident, chunk, ready = target.popleft()
                cost = len(chunk) + 80
                # Absolute finish deadlines absorb late/coalesced macOS wakes.
                # Bound accumulated lateness to one maximum fragment, never an
                # unbounded catch-up burst. Idle periods earn no token credit.
                next_slot = (
                    max(
                        next_slot,
                        time.monotonic() - (FRAGMENT + HEADER.size + 80) * 8 / 600000,
                    )
                    + cost * 8 / 600000
                )
                delay = next_slot - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                begin = time.monotonic_ns()
                write_chunks([chunk])
                end = time.monotonic_ns()
                sent_budget += cost
                result["sent"].append(
                    dict(
                        kind=kind,
                        generation=gen,
                        message_id=ident,
                        application_bytes=len(chunk),
                        budget_bytes=cost,
                        ready_s=ready,
                        write_start_ns=begin,
                        write_end_ns=end,
                    )
                )
            else:
                next_slot = max(next_slot, time.monotonic())
                time.sleep(0.001)
            if now - last_save >= 2:
                save(
                    out / "PROGRESS.json",
                    dict(
                        age_s=now,
                        generation=current,
                        sent_fragments=len(result["sent"]),
                        queued_base=len(video_q),
                        queued_detail=len(detail_q),
                        acks=len(acknowledgments),
                    ),
                )
                last_save = now
        # Preserve the planned final static interval even if the network empties early.
        remaining = start / 1e9 + finish - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        ident, chunks = message("finish", 0, b"{}")
        for chunk in chunks:
            time.sleep((len(chunk) + 80) * 8 / 600000)
            write_chunks([chunk])
            result["sent"].append(
                dict(
                    kind="finish",
                    generation=0,
                    message_id=ident,
                    application_bytes=len(chunk),
                    budget_bytes=len(chunk) + 80,
                    write_end_ns=time.monotonic_ns(),
                )
            )
        result["done"] = wait_event("done", 60)
        result["clock_after"] = [ping() for _ in range(9)]
        time.sleep(1.2)
        nettop_stop.set()
        nettop_thread.join(timeout=6)
        _, chunks = message("bye", 0, b"{}")
        write_chunks(chunks)
        process.stdin.close()
        code = process.wait(timeout=15)
        assert code == 0
        result.update(
            complete=True,
            finished_ns=time.monotonic_ns(),
            ssh_returncode=code,
            ack_application_bytes=incoming_bytes,
            paced_application_bytes=sum(x["application_bytes"] for x in result["sent"]),
            paced_budget_bytes=sum(x["budget_bytes"] for x in result["sent"]),
        )
        save(out / "RESULTS.json", result)
        print(
            a.run_name,
            "completed",
            len(states),
            "states",
            len(result["sent"]),
            "fragments",
            flush=True,
        )
    except BaseException as exc:
        result["error"] = repr(exc)
        save(out / "FAILED.json", result)
        if process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
        raise
    finally:
        encoder.shutdown(wait=True)
        nettop_stop.set()
        if nettop_thread is not None:
            nettop_thread.join(timeout=6)
        stderr.close()


if __name__ == "__main__":
    main()
