"""One-time local Mac login; save only after Screen Sharing authenticates.

Run in a Mac terminal. The password is read without echo, stays on the Mac,
and is stored through the project's native Keychain backend after success.
No screen curtain, audio, recording, or keyboard/mouse injection is requested.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
from pathlib import Path
import sys
import time

from isharescreen.credentials import set_password
from isharescreen.proxy.config import SessionConfig
from isharescreen.proxy.protocol.media_config import MediaOfferConfig
from isharescreen.proxy.session import Session


def main():
    if sys.platform != "darwin" or not sys.stdin.isatty():
        raise SystemExit("Run this command in a terminal on the Mac.")
    logging.basicConfig(level=logging.ERROR)
    host, username = "127.0.0.1", getpass.getuser()
    print("iShareScreen local capture setup")
    print(
        "Authenticate to this Mac's Screen Sharing service; save in Mac Keychain after success."
    )
    password = getpass.getpass(f"Mac login password for {username}: ")
    saved = []

    def authenticated():
        set_password(host, 5900, username, config.password)
        saved.append(True)

    media = MediaOfferConfig.from_environment()
    media.video.codec = "hevc"
    media.set_video_bitrate(600_000)
    media.validate()
    config = SessionConfig(
        host=host,
        username=username,
        password=password,
        curtain=False,
        audio=False,
        dynamic_resolution=False,
        media_offer=media,
        on_authenticated=authenticated,
    )
    del password
    try:
        with Session(config) as session:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and session.is_connected:
                if session.wait_for_fresh_tile(timeout=0.25):
                    break
            result = {
                "authenticated": bool(saved),
                "connected": session.is_connected,
                "canvas": list(session.canvas_dims),
                "decoder": session.hw_accel,
                "checked_at_unix": time.time(),
            }
        if not saved:
            raise RuntimeError("Screen Sharing authentication did not complete")
        marker = Path(__file__).with_name("MAC-AUTHORIZED.json")
        marker.write_text(json.dumps(result, indent=2) + "\n")
        os.chmod(marker, 0o600)
        print("Local Screen Sharing login succeeded. You can return to Codex.")
    except Exception as error:
        # Authentication exceptions may contain protocol details. Do not print
        # arbitrary exception contents or a traceback containing credential state.
        print(
            f"Local login failed ({type(error).__name__}); password was not printed.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
