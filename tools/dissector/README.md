# Python Dissector

Primary standalone decoder for native and standalone Screen Sharing traces.

This project replaces the Wireshark Lua plugin as the main decoding path for
long post-auth sessions. It reads `pcap` and `pcapng` directly, reconstructs
TCP byte streams, derives the first post-auth CBC transport key from
`EncodeEncryptionInfo`, slices encrypted records, decrypts them, and emits a
JSONL record ledger.

Mode-specific parsing is now split into dedicated modules:

- `libs/mode_rfb.py`: full-quality RFB path (`Raw`/`Zlib`/`ZRLE` and ARD rectangle extensions)
- `libs/mode_adaptive.py`: adaptive AV Conference media path (`0x3f2/0x3f3/0x3ea`, `0x1c MediaStreamOptions`)

## Status

- `working`: conversation discovery from captures
- `working`: TCP payload reassembly per direction
- `working`: first `EncodeEncryptionInfo` rekey extraction from server stream
- `working`: post-auth CBC record slicing and decryption
- `working`: JSONL ledger output with stable fallback names
- `open`: full semantic coverage for every opaque extension family
- `open`: later rekey families beyond the first confirmed transition

## Relationship to `iss --record`

`iss --record` (see `src/isharescreen/util/pcap_recorder.py`) writes a libpcap
file of the live session — the exact wire bytes, still encrypted, wrapped in
synthetic but valid Ethernet/IPv4/TCP|UDP framing. This dissector is the
decode path for those captures: it reassembles the TCP control stream, derives
the post-auth CBC transport key from the cleartext handshake it sees in the
same stream, decrypts the records, and (with the media keys recovered from
`0x1c`) decrypts the UDP/SRTP media. `--record session.pcap` also writes the
post-auth transport key to a sibling `session.key` (same dir, same prefix,
`.key` extension, `0600`). The dissector **auto-loads that sibling key**, so
no `--initial-key-hex` is needed — just point any command at the pcap and it
finds `session.key` next to it. (Pass `--initial-key-hex <hex>` to override;
`list-streams` needs no key at all.)

## Environment

The dissector has heavier, distinct dependencies from the `isharescreen`
client (scapy, pylibsrtp, …) and is intentionally **not** part of the installed
wheel. Install its own requirements into a virtualenv:

```bash
python -m venv tools/dissector/.venv
tools/dissector/.venv/bin/python -m pip install -r tools/dissector/requirements.txt
```

Run all commands below from the repository root.

## Commands

List candidate TCP conversations:

```bash
python tools/dissector/dissector_cli.py \
  list-streams \
  --pcap capture.pcapng
```

Decode one conversation to a JSONL record ledger:

```bash
python tools/dissector/dissector_cli.py \
  decode \
  --pcap capture.pcapng \
  --client-ip 10.0.0.2 \
  --server-ip 10.0.0.1 \
  --server-port 5900 \
  --client-port 50085 \
  --initial-key-hex <32-hex-session-key> \
  --output records.jsonl
```

Summarize an existing ledger:

```bash
python tools/dissector/dissector_cli.py \
  stats \
  --records records.jsonl
```

Fast-scan one session for cadence and update-driving traffic:

```bash
python tools/dissector/dissector_cli.py \
  scan-session \
  --pcap capture.pcapng \
  --client-ip 10.0.0.2 \
  --server-ip 10.0.0.1 \
  --server-port 5900 \
  --client-port 50085 \
  --initial-key-hex <32-hex-session-key> \
  --tail-seconds 3,5
```

Analyze adaptive UDP media streams (RTP shape, payload entropy, sequence gaps),
auto-correlated with the selected TCP adaptive session and latest client `0x1c`:

```bash
python tools/dissector/dissector_cli.py \
  analyze-udp-media \
  --pcap capture.pcapng \
  --client-ip 10.0.0.2 \
  --server-ip 10.0.0.1 \
  --server-port 5900 \
  --client-port 50085 \
  --initial-key-hex <32-hex-session-key> \
  --output udp_media_summary.json
```

If auto port inference fails, pass explicit media ports:

```bash
--media-ports 5900,5901
```

Port inference notes:

- Stage1 `0x3f2` still seeds the base UDP port, but payload-role mapping is not
  assumed fixed (`base=video`, `base+1=audio`).
- `analyze-udp-media` now adds `port_role_inference` based on RTP payload shape
  (PT=100 density + large FU packet rate), which is used by SRTP probe mapping.
- In captures where `base` carries mostly short control packets and `base+1`
  carries large HEVC FU payloads, the tool marks `base+1` as video candidate.

By default `analyze-udp-media` also runs an SRTP candidate probe using parsed
`0x1c` key blocks (requires `pylibsrtp`). Disable it with:

```bash
--disable-srtp-probe
```

Note: native adaptive sessions advertise 46-byte media keys in `0x1c`
(`32-byte master key + 14-byte salt`, AVC suite-5). `pylibsrtp`'s high-level
`Policy` API does not expose direct AES-CM-256 profile selection. The dissector
now falls back to `pylibsrtp` low-level bindings (`_binding.lib`) when
available, and runs real suite-5 unprotect probes (`AES_CM_256_HMAC_SHA1_80`
and `_32`) against captured RTP packets.

Export every client request from one selected session as JSON or Markdown:

```bash
python tools/dissector/dissector_cli.py \
  export-client-requests \
  --pcap capture.pcapng \
  --client-ip 10.0.0.2 \
  --server-ip 10.0.0.1 \
  --server-port 5900 \
  --client-port 50085 \
  --initial-key-hex <32-hex-session-key> \
  --format markdown \
  --output native_requests.md
```

Render an exact-timing replay video:

```bash
python tools/dissector/dissector_cli.py \
  render-replay \
  --pcap capture.pcapng \
  --client-ip 10.0.0.2 \
  --server-ip 10.0.0.1 \
  --server-port 5900 \
  --client-port 50085 \
  --initial-key-hex <32-hex-session-key> \
  --output replay.mkv \
  --emit-frame-ledger replay_frames.json
```

## Output

Each JSONL row represents one decrypted post-auth record and includes:

- `record_index`
- `direction`
- `timestamp_epoch`
- `msg_id`
- `msg_name`
- `record_kind`
- `plain_hex`
- `parsed`

Mode tagging:

- decoded rectangle rows now include `parsed.first_rect.mode_family`
- `SetEncodings` entries include per-encoding `mode_family`
- currently used values:
  - `full_quality_rfb`
  - `adaptive_media`

Replay output:

- `.mkv` is the primary exact-timing artifact
- variable frame spacing is preserved via a concat manifest with per-frame durations
- the optional frame ledger records the presented-frame timestamps and source records
- `scan-session` summarizes logical server update cadence, burst cadence, full-screen refreshes, and client update-driving message families for one selected TCP session
- `analyze-udp-media` correlates the selected TCP adaptive session with UDP media streams, infers media ports from stage1 `0x3f2`, summarizes RTP packet structure/cadence, and reports payload entropy to guide SRTP decrypt work
- `analyze-udp-media` now performs a **real decrypt + depayload of the dynamic
  (HEVC) media** in `udp_media.hevc_decode` (on by default; disable with
  `--disable-media-decode`):
  - self-contained `AES_CM_256_HMAC_SHA1_80` SRTP receive (`libs/srtp.py`,
    pycryptodome only — no libsrtp dependency)
  - for each media port, the 46-byte `0x1c` key block whose HMAC authenticates
    the most packets is selected automatically (`key2` = the proven receive
    direction), so the stream→key mapping is self-validating
  - HEVC depayload with Apple's DONL conventions (`libs/hevc_depay.py`):
    single-NAL / AP (type 48) / FU (type 49) all carry a leading 2-byte DONL,
    FU repeats it in every fragment, and AP has no DOND
  - per-port output: real `outer_type_counts` / `contained_type_counts`,
    `has_param_sets`, `has_idr`, and a per-SSRC `tiles` map (one SSRC == one
    tile) with reassembled NAL-type histograms and `donl_span`. Global decode
    order across tiles is by unwrapped DONL.
  - non-video (audio) streams are authenticated but reported as `kind:
    non-video` with an RTP payload-type histogram instead of bogus NAL types
- per-port UDP analysis also retains the legacy `hevc_pt100_analysis` shape
  heuristic (computed on the *encrypted* PT=100 payload, superseded by
  `hevc_decode`):
  - outer HEVC `nalu_type_counts` (PT=100)
  - FU inner `fu_type_counts`, `fu_start_counts`, `fu_end_counts` for outer type `49`
  - `idr_fu_seq_stats` to expose keyframe-fragment ordering/gaps
  - `likely_keyframe_starvation` / `initial_idr_burst_only` heuristics for
    adaptive bootstrap debugging
- top-level UDP summary now includes `port_role_inference` with per-port
  scoring and inferred `video_candidate_port` / `audio_candidate_port`
- `export-client-requests` emits one ordered row per client request with parsed fields where known and raw hex for unresolved extensions
- adaptive/media sessions now decode `0x1c MediaStreamOptions` with the fixed header fields proven in native `ScreenSharing.framework`:
  - `message_size_be16`
  - `message_version_be16`
  - `message_flags_be32`
  - `audio_offer_len_be16`
  - `video1_offer_len_be16`
  - `video2_offer_len_be16`
  - `session_id_hex`
  - `message_flag_names`
- offer decoding now follows the native fixed wire layout (`36-byte header` including 16-byte session id, then per-stream key pairs `46+46`, then plist offer bytes) and exports deterministic `audio_offer_offset` / `video1_offer_offset` / `video2_offer_offset`
- `0x1c` decode now exports per-stream key blocks (`stream_key_blocks`) so adaptive UDP decrypt experiments can be driven from parsed output
- SRTP probe output now handles 46-byte `0x1c` media keys as AVC suite-5 (`AES_CM_256_HMAC_SHA1_80/32`) and uses low-level `pylibsrtp` bindings when present; otherwise it reports exactly what is missing
- current standalone adaptive negotiation intentionally sets `stream1_supports_60fps` plus `do_not_send_cursor` in `0x1c`
- server-side `0x1c` records now reuse the same field decoder and report whether declared size matches `len(body)-4`
- adaptive/media negotiation is understood through client `0x1c` and server `0x3f2` stage 1 / stage 2
- stage detection in adaptive `0x3f2` parsing now uses `version/type` header
  semantics directly (`1/1=stage1`, `2/2=stage2`) and exports aliases such as
  `media_init_next_udp_port`
- codec payload decode/render for `0x3f3` and `0x3ea` remains open

Known private message families include:

- `MediaStreamOptions` (`0x1c`)
  - built by native `ScreenSharing.framework::_RFBMediaStreamServerConfiguration`
  - fixed header fields precede the plist/config payloads
  - native flag bits now proven statically:
    - bit `0`: `stream1_supports_60fps`
    - bit `1`: `stream2_supports_60fps`
    - bit `2`: `do_not_send_cursor`
    - bit `3`: `apple_remote_desktop_viewer`
  - native session state also carries explicit AVC width/height and per-stream frame-rate setup outside the raw wire dump
  - the standalone dissector now exports those header fields and decoded flag names

- `EncryptedInputEvent` (`0x10`)
  - emitted by native `_RFBPostX11KeyAndMouseCore`
  - fixed 18-byte record
  - byte `0` = `0x10`
  - byte `1` = subtype
  - bytes `2..17` = AES-ECB-encrypted in place before send
  - the standalone dissector decrypts that 16-byte block with the current client transport key and exports:
    - two leading event/flag bytes
    - two big-endian 32-bit fields
    - two tail state/button bytes
    - trailing big-endian `x` / `y` coordinates
  - exact per-field semantics are still partially unresolved, but the payload is no longer opaque
- `server_extension_0x96`

Other unresolved families are still emitted with explicit fallback names such as:

- `server_extension_0x96`

## Notes

- The current auto-discovery path derives the first CBC key/IV from the first
  cleartext `EncodeEncryptionInfo` (`0x44f`) seen in the selected server
  conversation.
- The decoder first tries to lock the client CBC handoff onto the native
  encrypted preface shape (`0x1d` then `0x02`) and falls back to the first
  short client `SetEncryptionMessage` (`0x12`, 8-byte form) after rekey if
  that probe fails.
- If automatic client or server start offsets prove wrong for a trace, the CLI
  supports explicit `--client-cbc-start-offset` and `--server-cbc-start-offset`
  overrides for investigation work.
- When multiple sessions share the same client/server IP pair, pass
  `--client-port <ephemeral_port>` to lock decode to one TCP session.
