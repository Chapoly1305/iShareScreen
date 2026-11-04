"""Optional fixed-format NVDEC -> EDSR -> GPU-surface runtime.

The preferred decode_submit() path includes reservation in the call. prepare()
can reserve before packet arrival when an application has a suitable idle
interval; its cost and output ownership must still be accounted for.
This component is not registered as a desktop renderer backend.
"""

import collections
import ctypes as c

from .cuda_surface import CudaSurfaceEDSR, SurfaceBusy, SurfaceLease
from .cuda_decoder import (
    NvdecDecoder as Decoder,
    MappedFrame,
    Frame,
    GraphEntry,
    WrongContext,
)


class Driver:
    def __init__(self):
        self.lib = c.CDLL("libcuda.so.1")
        signatures = {
            "cuStreamBeginCapture_v2": [c.c_void_p, c.c_int],
            "cuStreamEndCapture": [c.c_void_p, c.POINTER(c.c_void_p)],
            "cuGraphInstantiateWithFlags": [
                c.POINTER(c.c_void_p),
                c.c_void_p,
                c.c_ulonglong,
            ],
            "cuGraphLaunch": [c.c_void_p, c.c_void_p],
            "cuGraphExecDestroy": [c.c_void_p],
            "cuGraphDestroy": [c.c_void_p],
            "cuEventRecordWithFlags": [c.c_void_p, c.c_void_p, c.c_uint],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = c.c_int

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name} failed: CUDA error {code}")


class DriverGraph:
    def __init__(self, driver, stream, capture):
        self.driver, self.stream = driver, stream
        self.graph, self.executable = c.c_void_p(), c.c_void_p()
        driver.call("cuStreamBeginCapture_v2", stream, 0)
        try:
            capture()
        except BaseException:
            driver.lib.cuStreamEndCapture(stream, c.byref(self.graph))
            if self.graph:
                driver.lib.cuGraphDestroy(self.graph)
            raise
        driver.call("cuStreamEndCapture", stream, c.byref(self.graph))
        try:
            driver.call(
                "cuGraphInstantiateWithFlags", c.byref(self.executable), self.graph, 0
            )
        except BaseException:
            driver.lib.cuGraphDestroy(self.graph)
            raise

    def replay(self):
        self.driver.call("cuGraphLaunch", self.executable, self.stream)

    def close(self):
        if self.executable:
            self.driver.call("cuGraphExecDestroy", self.executable)
            self.executable = c.c_void_p()
        if self.graph:
            self.driver.call("cuGraphDestroy", self.graph)
            self.graph = c.c_void_p()


class Ticket:
    """One reserved output; it must be submitted once or explicitly cancelled."""

    def __init__(self, runtime, decoder, index, slot):
        self.runtime, self.decoder, self.index, self.slot = (
            runtime,
            decoder,
            index,
            slot,
        )
        self.serial, self.active = slot["serial"], True
        self.handle = decoder.handle
        self.metadata = Frame()
        self.metadata_ref = c.byref(self.metadata)
        self.bindings = runtime._bindings
        self.ready_handle = slot["ready"].cuda_event

    def submit(self, packet):
        return self.runtime.submit_prepared(self, packet)

    def cancel(self):
        return self.runtime.cancel(self)


class NativeCudaSurfaceEDSR(CudaSurfaceEDSR):
    """Hash-bound, single-inflight GPU inference with two leased output slots."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("input_format") != "nv12_bt709_full":
            raise ValueError("Explicit NV12 BT.709 full-range input required")
        super().__init__(*args, **kwargs)
        from .cuda_native_kernels import normalize_address

        self._normalize_mapped = normalize_address
        self._driver = Driver()
        self._address_graphs = collections.OrderedDict()
        self._bindings = (GraphEntry * 0)()
        self._ticket = None
        self.graph_builds = self.callback_launches = self.cold_launches = 0
        self.preparations = self.cancellations = 0

    def _normalize_raw(self, frame):
        n = self._input.numel()
        self._normalize_mapped[((n + 255) // 256,)](
            frame.pointer, self._input, frame.pitch, 1, n, 256, enable_fp_fusion=False
        )

    def _graphs_for(self, frame):
        key = (frame.pointer, frame.pitch)
        if key in self._address_graphs:
            return self._address_graphs[key]
        # The caller has established that no earlier inference is running.
        # Compile the pitch specialization outside capture without touching
        # any output slot a consumer might still own.
        self._normalize_raw(frame)
        self.stream.synchronize()
        graphs = []
        try:
            for slot in self._slots:

                def capture(slot=slot):
                    self._normalize_raw(frame)
                    self._driver.call(
                        "cuEventRecordWithFlags",
                        slot["input_consumed"].cuda_event,
                        self.stream.cuda_stream,
                        1,
                    )
                    self._compute(slot["output"])

                graphs.append(
                    DriverGraph(self._driver, self.stream.cuda_stream, capture)
                )
        except BaseException:
            for graph in graphs:
                graph.close()
            raise
        if len(self._address_graphs) == 2:
            _, expired = self._address_graphs.popitem(last=False)
            for graph in expired:
                graph.close()
        self._address_graphs[key] = graphs
        self.graph_builds += len(graphs)
        self._bindings = (GraphEntry * len(self._address_graphs))()
        for row, ((pointer, pitch), pair) in zip(
            self._bindings, self._address_graphs.items()
        ):
            row.pointer, row.pitch = pointer, pitch
            for index, graph in enumerate(pair):
                row.executable[index] = graph.executable.value
        return graphs

    def _check_ticket(self, ticket):
        if (
            self._ticket is not ticket
            or not ticket.active
            or ticket.runtime is not self
            or ticket.decoder._reserved_ticket is not ticket
            or not ticket.slot["busy"]
            or ticket.slot["serial"] != ticket.serial
        ):
            raise RuntimeError("Stale or foreign submission ticket")

    def prepare(self, decoder):
        self._producer()
        if (
            not isinstance(decoder, Decoder)
            or decoder.stream.cuda_stream != self.stream.cuda_stream
        ):
            raise ValueError("Matching decoder and runtime stream required")
        if not decoder.handle or decoder.poisoned or decoder.mapped:
            raise RuntimeError("Decoder must be live with no outstanding input mapping")
        decoder.check_context()
        with self._lock:
            if self._ticket is not None or decoder._reserved_ticket is not None:
                raise SurfaceBusy("A submission ticket is already reserved")
            if self._last_ready is not None and not self._last_ready.query():
                raise SurfaceBusy("Previous inference is running")
            available = [
                (i, s)
                for i, s in enumerate(self._slots)
                if not s["busy"]
                and (s["ready"] if s["discarded"] else s["consumed"]).query()
            ]
            if not available:
                raise SurfaceBusy("Both GPU outputs are owned by consumers")
            index, slot = available[0]
            self._pending = None
            slot.update(busy=True, serial=slot["serial"] + 1, discarded=False)
            try:
                ticket = Ticket(self, decoder, index, slot)
            except BaseException:
                slot.update(busy=False, discarded=True)
                raise
            self._ticket = decoder._reserved_ticket = ticket
            self.preparations += 1
            return ticket

    def _finish_ticket(self, ticket, *, cancelled):
        if cancelled:
            ticket.slot.update(busy=False, discarded=True)
            self.cancellations += 1
        ticket.active = False
        ticket.decoder._reserved_ticket = None
        self._ticket = None

    def cancel(self, ticket):
        self._producer()
        with self._lock:
            self._check_ticket(ticket)
            self._finish_ticket(ticket, cancelled=True)

    def submit_prepared(self, ticket, packet):
        self._producer()
        with self._lock:
            self._check_ticket(ticket)
            decoder, slot = ticket.decoder, ticket.slot
            try:
                decoder.validate_packet(packet)
            except BaseException:
                self._finish_ticket(ticket, cancelled=True)
                raise
            try:
                frame = decoder.decode_prepared(ticket, packet)
                if frame is None:
                    self._finish_ticket(ticket, cancelled=True)
                    return None
                frame.validate()
                if frame.metadata.reserved:
                    self.callback_launches += 1
                else:
                    # Only a new mapped address needs Torch context/stream
                    # setup, compilation and graph construction. The hot path
                    # uses explicit driver streams/events and checks context.
                    torch = self._torch
                    with (
                        torch.cuda.device(self.device),
                        torch.inference_mode(),
                        torch.cuda.stream(self.stream),
                    ):
                        self._graphs_for(frame)[ticket.index].replay()
                        slot["ready"].record(self.stream)
                    self.cold_launches += 1
                self._last_ready = slot["ready"]
                self._pending = frame
                lease = SurfaceLease(self, slot, slot["serial"])
                self._finish_ticket(ticket, cancelled=False)
                return frame, lease
            except WrongContext:
                # Native validation guarantees no work occurred. Cancel using
                # host state only; the caller can restore its original context.
                self._finish_ticket(ticket, cancelled=True)
                raise
            except BaseException:
                decoder.abort()
                self._finish_ticket(ticket, cancelled=True)
                raise

    def decode_submit(self, decoder, packet):
        return self.submit_prepared(self.prepare(decoder), packet)

    def submit(self, frame):
        self._producer()
        if self._ticket is not None:
            raise SurfaceBusy("Cancel or submit the prepared ticket first")
        if not isinstance(frame, MappedFrame):
            raise TypeError("Expected an owned native decoder mapping")
        frame.validate()
        if frame.stream.cuda_stream != self.stream.cuda_stream:
            raise ValueError("Decoder and inference must share this runtime stream")
        torch = self._torch
        with (
            self._lock,
            torch.cuda.device(self.device),
            torch.inference_mode(),
            torch.cuda.stream(self.stream),
        ):
            if self._last_ready is not None and not self._last_ready.query():
                raise SurfaceBusy("Previous inference is running")
            self._pending = None
            available = [
                (i, s)
                for i, s in enumerate(self._slots)
                if not s["busy"]
                and (s["ready"] if s["discarded"] else s["consumed"]).query()
            ]
            if not available:
                raise SurfaceBusy("Both GPU outputs are owned by consumers")
            index, slot = available[0]
            try:
                graphs = self._graphs_for(frame)
                graphs[index].replay()
                # A host-issued record marks completion pending immediately,
                # retaining the proven ready/query/lease behavior.
                slot["ready"].record(self.stream)
            except BaseException:
                self.stream.synchronize()
                raise
            slot.update(busy=True, serial=slot["serial"] + 1, discarded=False)
            self._last_ready = slot["ready"]
            self._pending = frame
            return SurfaceLease(self, slot, slot["serial"])

    def close(self):
        self._producer()
        if self._ticket is not None:
            raise SurfaceBusy("Cancel the prepared ticket before closing")
        if any(s["busy"] for s in self._slots):
            raise SurfaceBusy("Release all output leases before closing")
        self.stream.synchronize()
        for graphs in self._address_graphs.values():
            for graph in graphs:
                graph.close()
        self._address_graphs.clear()
        super().close()
        self._bindings = (GraphEntry * 0)()
