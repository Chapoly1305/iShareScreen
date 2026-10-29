# iShareScreen maintainability review

Review baseline: `myMain` at `16df1ff`, after the protocol configuration, saved-password and Kerberos ports. Reviewed on 2026-10-28.

The project should be cleaned up incrementally. The highest priorities are browser access control, connection lifecycle ownership, and the gap between configurable video formats and local rendering support. After those fixes, extract the large session and frontend components behind explicit interfaces. The new configuration schema and its wire-compatibility tests provide a useful foundation to preserve.

The scope includes the Python runtime, both viewers, the connection launcher, authentication, media and protocol layers, recording and dissector tools, packaging, tests and documentation. The inventory contains 57 runtime Python files (22,641 lines), 21 tool files (5,021 lines), and 32 test files (3,389 lines), plus browser HTML and JavaScript. Sizes include comments and blank lines; they identify places to inspect, not automatic reasons to split files.

Validation on Windows: **219 tests passed, two macOS-native checks skipped**. Installed dependencies pass `pip check`. Additional isolated checks reproduced the bridge deadlock, connection-after-close race, concurrent launcher race, unsupported 10-bit conversion, missing explicit socket cleanup, and CLI routing error described below. These checks used fake sessions, sockets and processes; they did not open a remote session. Native GPU behavior, live Kerberos authentication, and browser interaction across operating systems remain outside this validation.

The bridge access-control and lifecycle code is unchanged from local `main`; these are inherited issues, not reasons to revert the selected ports. The format issue is an existing renderer limitation made more accessible by the new configuration API.

## Findings to address first

P1 means address before broader cleanup. P2 means a concrete functional or resource-management issue. P3 means a smaller behavior or documentation correction.

1. **P1 — The browser bridge exposes screen access and input without authenticating viewers.**

   The QUIC listeners bind to `::` and `0.0.0.0`, and the HTTP bootstrap binds to `0.0.0.0`. Any matching WebTransport CONNECT to `/wt` is accepted without an origin, session-token or client-identity check. A reachable client can become the first viewer, start a Mac connection using the configured credentials, and send keyboard/mouse input. TLS protects transport but does not identify an authorized viewer here. Network reachability depends on the machine's firewall.

   Bind the default viewer to loopback, use an unpredictable per-launch viewer credential, and validate the allowed origin. If LAN viewing is intentional, expose it as an explicit authenticated mode. Keep this boundary separate from the remote Mac authentication backend. Source: [listeners](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:679), [HTTP bootstrap](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:810), [CONNECT dispatch](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:413), [viewer acceptance](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:491).

2. **P1 — Joining an existing browser session deadlocks the bridge.**

   `add_viewer()` holds the non-reentrant `_viewers_lock` and calls `_broadcast_frame()` when cached configuration or keyframes exist. `_broadcast_frame()` acquires the same lock. This runs from the QUIC event loop, so opening a second viewer or a per-monitor tab can stop the entire bridge. An isolated call with a running session and cached configuration remained blocked.

   Take the state snapshot under the lock, release it, then send configuration/keyframes to the joining viewer. Sending the cached configuration to all viewers also resets existing viewers' decoders unnecessarily. Add a regression that joins a second viewer and verifies both continued progress and the recipient of the cached configuration. Source: [join handling](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:826), [broadcast locking](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:1135), [browser configuration reset](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/static/index.html:674).

3. **P2 — A connection can finish after the last viewer has closed it.**

   The startup thread calls `Session.connect()` before publishing the session. Removing the last viewer clears `_session_starting`, but does not cancel or invalidate that startup. The old thread subsequently assigns `_session` and starts its pump even with no viewers. This was reproduced by pausing a fake connection, removing its viewer, and then allowing connection to complete. A new viewer during that interval can also start a second connection.

   Give each connection attempt a generation and cancellation state. Publish a completed session only if its generation is current and viewers still need it; otherwise close it. Track startup/retry tasks and move slow shutdown work out of the viewer-list lock and event loop. Source: [viewer removal](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:859), [startup](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:888), [teardown](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:1144).

4. **P2 — Valid media settings can describe formats the desktop cannot render correctly.**

   `PixelFormat.YUV420_10_VIDEO` passes `MediaOfferConfig.validate()`, but a real PyAV `yuv420p10le` frame passed to `_av_frame_to_tile()` returns `None`. The ordinary planar conversion handles 8-bit formats, and `TileFrame` carries no bit-depth, color-range, matrix or transfer metadata. Both desktop shaders assume full-range BT.709, so limited-range 8-bit planes are also interpreted incorrectly when received unchanged. Encoding a capability in an offer does not provide local playback support.

   Keep raw protocol-field validation separate from frontend/decoder compatibility validation. Reject unsupported explicit runtime combinations with a clear explanation, document the implemented formats, and carry color metadata through `TileFrame` before extending rendering. Preserve the existing default wire fixtures while doing this. Source: [settings validation](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/protocol/media_config.py:263), [frame conversion](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/media/decode_common.py:125), [frame contract](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/media/tiles.py:13), [shader](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/desktop/gpu.py:89).

5. **P2 — Resource cleanup does not consistently cover partial startup and failures.**

   Once `_phase_auth()` returns, a later negotiation exception has no enclosing cleanup for its socket; `Session` receives ownership only after negotiation returns. A fake ClientInit timeout left `close()` uncalled. Separately, `Session._teardown()` places `setsockopt(SO_LINGER)` and `close()` in the same `try`, so failure of the optional linger setting skips explicit close. That path was also reproduced. Garbage collection may eventually release a real socket, but cleanup should not depend on traceback or object lifetime.

   The desktop viewer similarly connects before acquiring audio, window and GPU resources, while its cleanup appears only at the end of the normal loop. Add ownership scopes using `try/finally` or `ExitStack`, transferring resources only after successful setup. Test failure at each acquisition stage. Preserve the documented native shutdown workaround until it is validated separately; `os._exit(0)` currently sits inside the public desktop `run()` function. Source: [negotiation ownership](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/protocol/negotiation.py:748), [socket teardown](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/session.py:1636), [desktop startup](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/desktop/app.py:285), [desktop exit](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/desktop/app.py:1464).

6. **P2 — Launcher operations race over process and dashboard state.**

   Connect and reconnect each start a new thread. `_launch()` updates shared `_PROC`, `_LAST`, `_STATE`, control paths and choice events without serializing the launch lifecycle. Two simultaneous submissions reproduced two child launches while only one remained tracked by `_PROC`. An old process's log pump can also mark a new connection as ended. `_free_bridge_port()` tries to recover ownership by inspecting process command text, which can terminate another legitimate iShareScreen instance on that port.

   Introduce one `SessionLauncher` owner with a serialized launch/stop operation and per-launch generation. Associate log readers, control sockets and modal replies with that generation, choose or reserve an owned bridge port, and terminate only children the launcher owns. Source: [launch state](C:/Users/chenj/iShareScreen/src/isharescreen/gui/connect.py:235), [old process completion](C:/Users/chenj/iShareScreen/src/isharescreen/gui/connect.py:361), [threaded actions](C:/Users/chenj/iShareScreen/src/isharescreen/gui/connect.py:790), [port cleanup](C:/Users/chenj/iShareScreen/src/isharescreen/gui/connect.py:181).

7. **P2 — Decoder fallback stops before decoder initialization has succeeded.**

   `registry.build_best()` falls back only when a constructor raises. `set_params()`, `start()` and initial burst handling happen later in `Session`. QSV opens its codec in `start()`; native VideoToolbox creates its decompression session in `set_params()`. Failure there aborts the connection instead of trying another candidate, even though the registry describes startup fallback. A capability probe cannot guarantee the actual stream will initialize.

   Move construction and stream initialization into one candidate-attempt operation, closing every failed candidate. Define the successful-ready boundary explicitly and test parameter/setup failure with fake decoders. Keep backend-specific recovery semantics: the intentional QSV/VideoToolbox no-op `restart()` behavior must not become an unconditional shared reset. Source: [registry](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/media/registry.py:222), [session initialization](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/session.py:1229), [QSV startup](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/media/qsvhevc.py:163), [VideoToolbox setup](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/media/vtdecode.py:131).

8. **P2 — Capture buffering has no defined behavior when disk writing fails or stalls.**

   The recorder uses an unbounded deque. A write error escapes `_writer_loop()` and ends the writer, while producers continue appending because `_file` remains set. Shutdown waits only two seconds and then writes/closes from another thread, even if the writer still owns a detached batch. A slow or full disk can therefore accumulate memory, lose capture data, or overlap file operations. This finding follows from control-flow inspection; no real disk failure was induced.

   Keep file ownership in one worker, bound queued bytes, expose dropped-packet/write-failure counters, and disable recording cleanly after an I/O error. Add fake-file tests for a stalled writer, a write error, and close during an active batch. Apply the same bounded-buffer review to dashboard subscriber queues and incoming command framing. Source: [recorder state](C:/Users/chenj/iShareScreen/src/isharescreen/util/pcap_recorder.py:204), [writer](C:/Users/chenj/iShareScreen/src/isharescreen/util/pcap_recorder.py:276), [producer](C:/Users/chenj/iShareScreen/src/isharescreen/util/pcap_recorder.py:328), [close](C:/Users/chenj/iShareScreen/src/isharescreen/util/pcap_recorder.py:350).

9. **P3 — CLI routing and profile restoration have drifted from their advertised behavior.**

   The entry point checks for the exact token `--host`; `--host=example.invalid` goes to the GUI and loses its CLI connection arguments. This was reproduced with patched entry functions. The form saves resolution, scale, audio, curtain and decoder preferences but renders defaults for those fields on the next launch. The README still describes the removed terminal UI and its keys.

   Use one argument parser for dispatch, restore every supported persisted field through a typed form model, and update the usage examples. Add behavioral tests for both host argument forms and a complete profile round trip. Source: [entry routing](C:/Users/chenj/iShareScreen/src/isharescreen/tui/entry.py:19), [profile save](C:/Users/chenj/iShareScreen/src/isharescreen/gui/connect.py:342), [form rendering](C:/Users/chenj/iShareScreen/src/isharescreen/gui/connect.py:697), [README](C:/Users/chenj/iShareScreen/README.md:96).

## Structural cleanup priorities

| Area | Maintenance problem | Recommended boundary |
| --- | --- | --- |
| Session orchestration | `session.py` is 4,596 lines. Connection setup, RTP assembly, feedback/recovery, display parsing, clipboard, cursor and metrics share extensive mutable state. | Extract RTP assembly first, then recovery policy and display state. Give each component explicit inputs, outputs and thread ownership; keep `Session` as the coordinator. |
| Decoder contract | `_decoder` is typed as `object`; session logic accesses `_gate`, `_states`, `_keyframe_required`, `_fir_attempts`, `_hw_name` and other implementation details through direct access or `getattr`. | Define a `Decoder` protocol with public capability, initialization, frame, recovery and diagnostics APIs. Preserve distinct backend behavior instead of forcing inheritance. |
| Configuration | The schema snapshots offers, but the CLI still mutates process-wide environment variables and runtime code reads decoder/debug options at different times. Display HDR and media HDR are separate settings. | Resolve CLI, GUI and legacy environment inputs once into validated session/frontend options. Keep mutable builder objects at the boundary and a stable runtime snapshot internally. |
| Desktop viewer | `run()` spans 1,275 lines, including nested callbacks and mutable one-element containers. Primary and secondary windows duplicate input handling. | Introduce `DesktopViewer` and per-window `ViewState`, with shared coordinate/input logic and a cursor controller. Establish deterministic cleanup before moving callbacks. |
| Browser frontend | Transport, decoding, reorder logic, input, audio, resize and diagnostics share one large HTML script. Server code reaches into aioquic private stream fields. | Extract JavaScript modules by responsibility and isolate private aioquic access in a small compatibility adapter with contract tests. Verify packaged assets after extraction. |
| Error handling | The runtime contains 172 `except Exception` handlers. Some correctly isolate optional native features; others conceal broken state or retry indefinitely. | Review catches by boundary. Distinguish transient transport errors, unsupported capabilities, authentication failures and internal faults; give each an explicit retry/stop policy. |
| Protocol tools | The standalone dissector and client implement overlapping SRTP/NAL parsing, with different dependency environments. | Add shared conformance fixtures first. Extract only stable pure wire helpers when useful, retaining the dissector's independent installation and avoiding a new heavy dependency chain. |
| Dead code and documentation | Browser bridge bitrate/FPS/crop attributes are assigned but unused, and its module description still mentions a libx264 encoder. Several comments describe removed TUI or obsolete recovery paths. | Remove unused internal state after checking call sites; deprecate public parameters deliberately. Move historical investigations into docs while retaining comments that explain active protocol/native workarounds. |

Useful extraction anchors: [RTP assembly](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/session.py:2374), [recovery policy](C:/Users/chenj/iShareScreen/src/isharescreen/proxy/session.py:4042), [desktop viewer](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/desktop/app.py:214), [browser script](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/static/index.html:355), [unused bridge state](C:/Users/chenj/iShareScreen/src/isharescreen/frontend/wt/server.py:569).

## Verification needed for safe refactoring

The current tests protect important packet-loss recovery, codec-selection and authentication details. Keep the 40 deterministic media-offer wire fixtures, the concurrent decoder lifecycle checks, the keyring secret-handling checks, and the explicit Kerberos framing tests.

The principal gap is lifecycle coverage across components. Fifteen test files bypass constructors with `__new__`; those tests are useful for focused logic, but cannot establish that startup, ownership and teardown work together. There are no direct tests of the bridge classes, `RenderCanvas` lifecycle or `ControlServer` in the current suite. Introduce fake transport/decoder/process factories so public startup, join, resize, disconnect and failure paths can be exercised without hardware.

Add repository CI for Windows, Linux and macOS, with a minimum-supported-Python check and a primary development version. Start linting with undefined names and unused imports; add type checks at the new interfaces rather than rewriting every module at once. Build and install the wheel in CI, verify browser assets are present, and test the supported dependency range. Keep live GPU/Kerberos/browser checks as a distinct platform-validation layer. The current `pyproject.toml` defines pytest but no lint/type-check configuration, and there is no tracked CI workflow.

## Suggested implementation sequence

| Order | Deliverable | Completion criterion |
| --- | --- | --- |
| 1 | Separate fixes for bridge access, second-viewer locking and cancelled startup | Authorized local viewing works; unauthorized viewers are rejected; join and disconnect-during-connect regression tests pass. |
| 2 | Exception-safe resource ownership, serialized GUI launch, decoder startup fallback, bounded recorder | Fault-injection tests leave no owned resources or obsolete workers active; failed candidates close and fallback succeeds. |
| 3 | Unified configuration resolution and explicit playback capabilities | CLI/GUI/environment precedence is tested; unsupported explicit formats fail clearly; legacy wire fixtures remain stable. |
| 4 | Decoder interface and incremental session extraction | Existing recovery behavior remains covered; consumers stop reaching into decoder private state. |
| 5 | Desktop/window components and browser asset modules | Input mapping and multi-window behavior share tests; wheel installation includes and serves all assets. |
| 6 | Documentation, obsolete-state removal and broader CI | Usage matches the shipped UI; supported platforms run automated checks; cleanup commits remain small enough to review independently. |

Correctness changes should have their own commits before file moves or renames. Preserve working native recovery decisions and protocol bytes while extracting responsibilities. This review adds documentation only; implementation remains a separate step.


## Implemented refactor follow-up

The first refactor addresses resource ownership and extracts independently useful modules:

- `SessionConfig` lives in `proxy/config.py`, with its old import path retained. Decoder environment settings are captured once per session, and a structural decoder interface documents the shared contract.
- Decoder fallback includes parameter setup, startup, and the initial burst. Failed candidates are closed before the next candidate starts.
- Negotiation failures close their socket. Desktop resources use scoped cleanup, including startup exceptions; process exit is explicitly requested only by the CLI.
- Browser session startup is serialized and generation checked. A viewer leaving during connection cannot leave an unpublished session running. Cached frames are sent only to the joining viewer, outside the viewer lock.
- The browser bridge binds loopback and requires a per-launch token for control/session access. Open the complete URL printed by the launcher. The browser script is now a separate `viewer.js` asset.
- Capture buffering is bounded to 16 MiB pending plus one writer batch. Disk write failures stop recording, and only the writer closes the file. Drops are counted on the recorder.

Remaining review work: GUI process ownership/profile restoration, deeper RTP assembly extraction, remaining decoder-private coupling, and native GPU/audio/browser smoke testing. These changes do not replace that platform testing. The combined refactor includes CLI override precedence (omitted flags inherit, explicit auto clears overrides), decoder-name validation, and rejection of empty pixel-format capabilities or AVC offers without 8-bit YUV420.
