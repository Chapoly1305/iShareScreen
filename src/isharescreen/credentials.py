"""Passwords in the OS credential store, scoped to an iss endpoint/account.

Select native backends explicitly: never fall back to a plaintext keyring or
an arbitrary user-configured plugin. Import lazily so manual login works on
machines without an available credential store.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Callable

log = logging.getLogger("iss.credentials")


class CredentialStoreError(Exception):
    """The system credential store is unavailable or access was denied."""


def _backend():
    try:
        if sys.platform == "darwin":
            from keyring.backends.macOS import Keyring
        elif sys.platform == "win32":
            from keyring.backends.Windows import WinVaultKeyring as Keyring
        else:
            from keyring.backends.SecretService import Keyring
        backend = Keyring()
        if backend.priority <= 0:
            raise RuntimeError("backend unavailable")
        return backend
    except Exception:
        raise CredentialStoreError(
            "system credential store unavailable; install keyring and unlock "
            "Keychain / Credential Manager / Secret Service"
        ) from None


def _service(host: str, port: int) -> str:
    # JSON avoids ambiguous host/port concatenation, including IPv6 hosts.
    return "isharescreen:" + json.dumps(
        [host.strip().lower().rstrip("."), int(port)], separators=(",", ":")
    )


def get_password(host: str, port: int, username: str) -> str | None:
    try:
        return _backend().get_password(_service(host, port), username)
    except CredentialStoreError:
        raise
    except Exception:
        # Backend exceptions can include credential data; do not expose them.
        raise CredentialStoreError("could not read system credential store") from None


def set_password(host: str, port: int, username: str, password: str) -> None:
    if not password:
        raise CredentialStoreError("cannot save an empty password")
    try:
        _backend().set_password(_service(host, port), username, password)
    except CredentialStoreError:
        raise
    except Exception:
        raise CredentialStoreError("could not save password in system credential store") from None


def delete_password(host: str, port: int, username: str) -> None:
    try:
        backend = _backend()
        service = _service(host, port)
        if backend.get_password(service, username) is not None:
            backend.delete_password(service, username)
    except CredentialStoreError:
        raise
    except Exception:
        raise CredentialStoreError("could not delete saved password") from None


def save_after_auth(host: str, port: int, username: str, password: str) -> Callable[[], None]:
    """One-shot callback: save only after successful remote authentication."""
    pending = True

    def save() -> None:
        nonlocal pending
        if not pending:
            return
        pending = False
        try:
            set_password(host, port, username, password)
        except CredentialStoreError as error:
            log.warning("Connected, but password was not saved: %s", error)
        else:
            log.info("Password saved in system credential store")

    return save
