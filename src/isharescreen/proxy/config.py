"""Public session configuration, independent of the session implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .media.options import DecoderOptions
from .protocol.media_config import MediaOfferConfig
from .protocol.negotiation import AdvertiseDims

@dataclass
class SessionConfig:
    """Connection parameters. Defaults are conservative; the CLI
    overrides whatever the user customised."""

    host: str
    username: str
    password: str
    port: int = 5900
    auth_mode: Literal["srp", "nonsrp", "kerberos"] = "srp"
    advertise: Optional[AdvertiseDims] = None
    hdr: bool = False
    audio: bool = True
    # HiDPI mode for the host's virtual display, resolved to a backing:point
    # ratio by the frontend (which knows the window size):
    #   "on"   → always 2× (Retina): crisp, but ~4× the pixels = more
    #            bandwidth, and UI renders half-size on a non-Retina client.
    #   "off"  → always 1× (flat, backing == logical): far less bandwidth,
    #            correctly-sized UI on non-Retina (Linux/Windows) clients.
    #   "auto" → match the LOCAL display: 2× on a Retina client, 1× on a
    #            non-Retina one (so the stream maps 1:1 to the client's pixels);
    #            2× is downgraded to 1× when it wouldn't fit the host's
    #            3840×2160 cap (window logical > 1920×1080).
    # The frontend passes the resolved scale into send_dynamic_resolution and
    # the initial AdvertiseDims; the proxy itself only carries the mode.
    hidpi: str = "auto"

    # When auth user differs from the console user, ask the console user
    # to share their existing session (Apple's "Ask to share" choice in
    # the Screen Sharing.app prompt). On accept, the viewer joins the
    # console session in observe-only mode rather than starting an
    # alt-user virtual display. False (default) = no SessionSelect exchange.
    share_console: bool = False

    # When auth user differs from the console user, log them in to a
    # fresh virtual display via Apple's cmd=2 SessionSelect path
    # (encrypted creds at body+0x48 / +0x88; daemon spawns user2's
    # vdisplay; iss replays Apple Screen Sharing's canned 308-byte SDC so the
    # encoder targets the new vdisplay rather than the console user's
    # screen). Mutually exclusive with `share_console`.
    alt_session: bool = False

    # Optional callback () -> "share_console" | "alt_session". Invoked during
    # negotiation when a different user is at the console and neither flag was
    # forced — lets a launcher (the GUI) prompt the user at connect time.
    on_session_choice: "Optional[Callable[[str], str]]" = None

    # Curtain mode: send the SetDisplayConfiguration message during
    # negotiation so the daemon spins up a SkyLight virtual display
    # and blanks the host's physical screen while we view. When False,
    # we skip that message entirely — the daemon then encodes whatever
    # is on the physical display, no curtain. Default on for privacy
    # (the host's screen isn't broadcast to anyone walking past). The
    # cmd=2 alt-session path ignores this flag and always sends Apple's
    # canned SDC instead.
    curtain: bool = True

    # Which 4-SSRC quality tier to subscribe to (0 = highest, ascending = lower).
    quality_tier: int = 0

    # AAC-ELD backend override (None → platform default).
    aac_backend: Optional[str] = None

    # Whether to perform Apple's two-TCP warmup before the real session.
    # Required for surviving lock-screen → login → desktop transitions
    # without the daemon closing our TCP. Defaults to True.
    warmup_tcp: bool = True

    # UDP port overrides — None ⇒ port + 0/1 (Apple's layout: audio
    # rtcp-muxes onto ctrl_port; video rtcp-muxes onto video_port).
    # Custom values let a NAT relay route the streams.
    udp_ctrl_port: Optional[int] = None
    udp_video_port: Optional[int] = None
    udp_bind_host: str = ""  # "" = INADDR_ANY

    # Local control socket: when set, the Session opens a ControlServer at
    # this path (UDS on POSIX, sidecar `.port` file on Windows) so an
    # external TUI / monitor can subscribe to snapshots and (later) issue
    # commands. None = no control server.
    control_socket: Optional[str] = None

    # Packet capture: when set, every byte that crosses the TCP control
    # socket and the two UDP media sockets is written — exactly as it goes
    # on the wire (still enc1103/SRTP encrypted) — to a classic `.pcap` at
    # this path, with synthetic Ethernet/IPv4/TCP|UDP framing. The result is
    # byte-identical to a packet-capture dump and feeds straight into the
    # bundled Python dissector (which derives the keys from the cleartext
    # handshake it sees) or Wireshark. None ⇒ no capture; zero overhead.
    record_pcap: Optional[str] = None

    # Dynamic resolution: the frontend issues a mid-session resize (via
    # Session.send_dynamic_resolution) whenever the viewer window changes
    # size, so the host's virtual display re-renders sharp at the new size
    # instead of stretching a fixed canvas. This is an IN-BAND change on
    # the existing connection — 0x1d → server 0x451 → 0x1c re-offer →
    # encoder restart — no reconnect. Purely a frontend concern; the
    # protocol layer doesn't read this flag, it rides along with the rest
    # of the config to the desktop viewer. Off ⇒ classic fixed canvas
    # (the window just stretches the stream on resize).
    dynamic_resolution: bool = False
    # Editable startup offer. None uses the existing ISS_* environment choices.
    # Session snapshots it at construction; this is not a live control API.
    media_offer: Optional[MediaOfferConfig] = field(default=None, kw_only=True)
    decoder_options: Optional[DecoderOptions] = field(default=None, kw_only=True)

    # Credential persistence is supplied by the launcher, after successful
    # negotiation. These remain last to preserve existing positional callers.
    on_authenticated: Optional[Callable[[], None]] = None
    krb_realm: str = ""
    krb_principal: str = ""
    krb_service: str = ""

