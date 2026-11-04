"""Prepared ownership, context rejection, cancellation and consumer backpressure."""

import copy
import ctypes as c
import threading
from types import SimpleNamespace

import torch
from benchmarks.super_resolution.edsr_cuda.ownership import (
    source,
    expect,
    release_input,
)
from isharescreen.enhancement.cuda_decoder import WrongContext
from isharescreen.enhancement.cuda_surface import SurfaceBusy


def check_tickets(runtime, path):
    stream = runtime.stream
    torch.cuda.set_stream(stream)
    stream.synchronize()
    demux, packets, decoder = source(path, stream)
    packet = next(packets)
    ticket = runtime.prepare(decoder)
    expect(SurfaceBusy, lambda: runtime.prepare(decoder))
    expect(SurfaceBusy, runtime.close)
    expect(RuntimeError, decoder.close)
    expect(RuntimeError, lambda: decoder.decode(packet))
    expect(RuntimeError, lambda: runtime.submit_prepared(copy.copy(ticket), packet))
    errors = []

    def foreign_thread():
        try:
            ticket.cancel()
        except RuntimeError:
            errors.append(True)

    thread = threading.Thread(target=foreign_thread)
    thread.start()
    thread.join()
    assert errors and ticket.active
    ticket.cancel()
    expect(RuntimeError, ticket.cancel)
    expect(RuntimeError, lambda: ticket.submit(packet))
    assert runtime._ticket is None and decoder._reserved_ticket is None

    ticket = runtime.prepare(decoder)
    expect(
        ValueError,
        lambda: ticket.submit(SimpleNamespace(discardable=True, bDiscontinuity=False)),
    )
    assert not ticket.active and not decoder.poisoned and not decoder.mapped

    # Remove the current context using the driver API, without allowing Torch
    # to lazily restore it. Both preparation and native submission must reject
    # before parsing, and cancellation must require no current CUDA context.
    driver = c.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes = [c.POINTER(c.c_void_p)]
    driver.cuCtxGetCurrent.restype = c.c_int
    driver.cuCtxSetCurrent.argtypes = [c.c_void_p]
    driver.cuCtxSetCurrent.restype = c.c_int
    context = c.c_void_p()
    assert driver.cuCtxGetCurrent(c.byref(context)) == 0 and context.value
    ticket = runtime.prepare(decoder)
    try:
        assert driver.cuCtxSetCurrent(None) == 0
        expect(WrongContext, lambda: ticket.submit(packet))
        assert not ticket.active and not decoder.poisoned and not decoder.mapped
        expect(WrongContext, lambda: runtime.prepare(decoder))
    finally:
        assert driver.cuCtxSetCurrent(context) == 0
    ticket = runtime.prepare(decoder)
    frame, lease = ticket.submit(packet)
    expect(RuntimeError, lambda: ticket.submit(packet))
    lease.discard()
    release_input(decoder, stream)

    # A released output is still unavailable while its consumer is running.
    slow = torch.cuda.Stream()
    frame, first = runtime.decode_submit(decoder, next(packets))
    first.wait_on(slow)
    with torch.cuda.stream(slow):
        torch.cuda._sleep(500_000_000)
    first.release()
    release_input(decoder, stream)
    frame, second = runtime.decode_submit(decoder, next(packets))
    release_input(decoder, stream)
    assert not slow.query()
    expect(SurfaceBusy, lambda: runtime.prepare(decoder))
    slow.synchronize()
    ticket = runtime.prepare(decoder)
    ticket.cancel()
    second.discard()

    # Complete-access-unit decoding has already emitted every frame. EOS must
    # return no output and release its reserved slot without poisoning state.
    eos = SimpleNamespace(
        discardable=False, bDiscontinuity=False, bsl_data=0, bsl=0, pts=0
    )
    ticket = runtime.prepare(decoder)
    assert ticket.submit(eos) is None
    assert not ticket.active and runtime._ticket is None and not decoder.mapped
    assert runtime.collect_completed()
    decoder.close()
    return dict(
        single_reservation_bound=True,
        close_reserved_rejected=True,
        decoder_close_reserved_rejected=True,
        bypass_decode_rejected=True,
        foreign_and_stale_ticket_rejected=True,
        producer_thread_enforced=True,
        cancellation_reclaims_output=True,
        invalid_packet_cancels_without_poison=True,
        wrong_context_rejected_before_native_work=True,
        context_recovery=True,
        delayed_consumer_blocks_reuse=True,
        eos_releases_reservation=True,
    )
