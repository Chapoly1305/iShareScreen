"""Exercise native launch boundaries, input reuse, cache eviction and failures."""

import torch
import PyNvVideoCodec as nvc
from isharescreen.enhancement.cuda_decoder import NvdecDecoder as Decoder
from isharescreen.enhancement.cuda_native import (
    NativeCudaSurfaceEDSR as CallbackSurfaceEDSR,
)
from isharescreen.enhancement.cuda_surface import SurfaceBusy
from benchmarks.super_resolution.edsr_cuda.common import reference_rgb


def expect(kind, operation):
    try:
        operation()
    except kind:
        return
    raise AssertionError("Expected " + kind.__name__)


def source(path, stream):
    demux = nvc.CreateDemuxer(filename=str(path))
    return demux, iter(demux), Decoder(stream)


def release_input(decoder, stream):
    event = torch.cuda.Event()
    event.record(stream)
    event.synchronize()
    decoder.release(event)


def check_ownership(runtime, path, engine, sha):
    stream = runtime.stream
    torch.cuda.set_stream(stream)
    consumer = torch.cuda.Stream()
    demux, packets, decoder = source(path, stream)
    other_demux, other_packets, other = source(path, stream)
    other_packet = next(other_packets)
    frame, first = runtime.decode_submit(decoder, next(packets))
    held = first.wait_on(consumer)
    with torch.cuda.stream(consumer):
        saved = held.clone()
    consumer.synchronize()
    release_input(decoder, stream)
    expect(RuntimeError, lambda: runtime.submit(frame))
    second_frame, second = runtime.decode_submit(decoder, next(packets))
    assert second_frame.metadata.reserved == 1
    assert not second.ready()
    expect(SurfaceBusy, lambda: runtime.decode_submit(other, other_packet))
    stream.synchronize()
    release_input(decoder, stream)
    pending_packet = next(packets)
    expect(SurfaceBusy, lambda: runtime.decode_submit(decoder, pending_packet))
    assert not decoder.mapped  # Reservation rejection occurs before parsing.
    expect(SurfaceBusy, runtime.close)
    second.discard()
    third_frame, third = runtime.decode_submit(decoder, pending_packet)
    third.discard()
    stream.synchronize()
    assert torch.equal(held, saved)
    release_input(decoder, stream)
    first.release()
    consumer.synchronize()
    assert runtime.collect_completed()
    decoder.close()
    other.close()

    # Keep three different decoder mappings alive so the graph cache must evict
    # an address. Revisit the evicted mapping, then decode through the updated
    # bindings; native code must never retain a stale graph pointer.
    mapped = []
    for _ in range(3):
        dmx, pkt, dec = source(path, stream)
        frame, lease = runtime.decode_submit(dec, next(pkt))
        lease.discard()
        stream.synchronize()
        mapped.append((dmx, pkt, dec, frame))
    assert len({row[3].pointer for row in mapped}) == 3
    assert len(runtime._address_graphs) == 2
    old_builds = runtime.graph_builds
    frame = mapped[0][3]
    lease = runtime.submit(frame)
    lease.discard()
    stream.synchronize()
    assert runtime.graph_builds == old_builds + 2
    for dmx, pkt, dec, old in mapped:
        release_input(dec, stream)
        frame, lease = runtime.decode_submit(dec, next(pkt))
        lease.discard()
        stream.synchronize()
        release_input(dec, stream)
        dec.close()
    assert runtime.collect_completed()

    # Force a Python failure after native CUDA graph launch has succeeded.
    # The runtime must wait for readers, unmap, poison that decoder and recover
    # its reserved output slot without handing a partial lease to a consumer.
    dmx, pkt, dec = source(path, stream)
    frame, lease = runtime.decode_submit(dec, next(pkt))
    lease.discard()
    release_input(dec, stream)
    original = dec.decode_prepared
    launched = []

    def injected(*args):
        frame = original(*args)
        assert frame.metadata.reserved == 1
        launched.append(True)
        raise RuntimeError("Injected failure after successful native launch")

    dec.decode_prepared = injected
    expect(RuntimeError, lambda: runtime.decode_submit(dec, next(pkt)))
    assert launched and dec.poisoned and not dec.mapped
    assert not any(s["busy"] for s in runtime._slots)
    assert stream.query()
    expect(RuntimeError, lambda: dec.decode(next(pkt)))
    dec.close()
    dmx, pkt, dec = source(path, stream)
    frame, lease = runtime.decode_submit(dec, next(pkt))
    lease.discard()
    release_input(dec, stream)
    dec.close()
    assert runtime.collect_completed()

    # A separate graph adds a deterministic delay *before* preprocessing, so
    # an incorrectly reused input-completion event would overwrite Y too soon.
    class DelayedRuntime(CallbackSurfaceEDSR):
        def _normalize_raw(self, frame):
            torch.cuda._sleep(30_000_000)
            super()._normalize_raw(frame)

    delayed = DelayedRuntime(engine, engine_sha256=sha, input_format="nv12_bt709_full")
    ds = delayed.stream
    torch.cuda.set_stream(ds)
    dmx, pkt, dec = source(path, ds)
    ref_dmx, ref_pkt, ref_dec = source(path, ds)
    frame, lease = delayed.decode_submit(dec, next(pkt))
    lease.discard()
    release_input(dec, ds)
    reference_frame = ref_dec.decode(next(ref_pkt))
    release_input(ref_dec, ds)
    reference_frame = ref_dec.decode(next(ref_pkt))
    expected = reference_rgb(reference_frame.as_tensor()).unsqueeze(0).float() / 255
    release_input(ref_dec, ds)
    ref_dec.close()
    frame, lease = delayed.decode_submit(dec, next(pkt))
    assert frame.metadata.reserved == 1 and not lease.ready()
    tensor = frame.as_tensor()
    reuse = torch.cuda.Stream()
    lease.wait_input_reuse_on(reuse)
    with torch.cuda.stream(reuse):
        tensor[:720].fill_(255)
    lease.wait_on(consumer)
    lease.release()
    consumer.synchronize()
    reuse.synchronize()
    assert torch.equal(delayed._input, expected)
    assert torch.all(tensor[:720] == 255).item()
    release_input(dec, ds)
    dec.close()
    assert delayed.collect_completed()
    delayed.close()
    torch.cuda.set_stream(stream)
    return dict(
        native_callback_inflight_rejected=True,
        output_reservation_before_decode=True,
        two_slot_bound=True,
        held_output_preserved=True,
        stale_mapping_rejected=True,
        close_with_live_lease_rejected=True,
        graph_cache_eviction_and_rebind=True,
        partial_launch_failure_cleanup=True,
        recovery_with_new_decoder=True,
        delayed_callback_input_reuse_event=True,
    )
