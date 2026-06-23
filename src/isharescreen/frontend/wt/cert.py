"""Self-signed certificate generation for WebTransport.

WebTransport requires HTTPS / HTTP/3, which means TLS. For a LAN tool
the user can't reasonably obtain a real CA-signed cert, so we generate
an ephemeral self-signed cert at startup and pass its SHA-256
fingerprint to the browser via the URL query string. The browser then
uses `serverCertificateHashes` in the WebTransport constructor to
short-circuit normal CA validation — Chrome/Edge/Firefox all support
this for non-root EC certs ≤ 14 days old.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
from pathlib import Path
from typing import Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


log = logging.getLogger(__name__)

# Spec cap: serverCertificateHashes accepts certs only if NotAfter is
# within 14 days. We set 13 days for safety. Tools that re-launch every
# session generate a fresh cert each time, so this is never visible to
# users.
_CERT_LIFETIME_DAYS = 13


def make_self_signed(common_name: str = "localhost") -> tuple[bytes, bytes, str]:
    """Generate a fresh ECDSA-P256 self-signed cert valid for 13 days.

    Returns ``(cert_pem, key_pem, sha256_hex)`` — the PEMs to feed
    aioquic's `load_cert_chain`, plus the cert's DER SHA-256 hex digest
    for the browser-side `serverCertificateHashes` array."""
    import ipaddress
    key = ec.generate_private_key(ec.SECP256R1())
    now = _dt.datetime.now(_dt.timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=1))
        .not_valid_after(now + _dt.timedelta(days=_CERT_LIFETIME_DAYS))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("iss-bridge"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv6Address("::1")),
            ]),
            critical=False,
        )
        # Chrome's `serverCertificateHashes` rejects certs without
        # these extensions — silently, with no handshake attempt
        # reaching the server. KeyUsage + EKU are both load-bearing.
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    sha256_hex = hashlib.sha256(cert_der).hexdigest()
    log.info("generated self-signed cert (sha256 %s…)", sha256_hex[:16])
    return cert_pem, key_pem, sha256_hex


def write_cert(
    cache_dir: Optional[Path] = None,
    *, common_name: str = "iss-bridge",
) -> tuple[Path, Path, str]:
    """Generate (or reuse) cert + key files on disk. Returns paths +
    sha256 hex. If `cache_dir` is given and contains a still-valid
    pair, reuse it; otherwise mint fresh."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cert_p = cache_dir / "wt.crt"
        key_p = cache_dir / "wt.key"
        if cert_p.exists() and key_p.exists():
            try:
                cert_pem = cert_p.read_bytes()
                cert = x509.load_pem_x509_certificate(cert_pem)
                if cert.not_valid_after_utc > _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=1):
                    cert_der = cert.public_bytes(serialization.Encoding.DER)
                    sha256_hex = hashlib.sha256(cert_der).hexdigest()
                    return cert_p, key_p, sha256_hex
            except Exception as e:
                log.debug("cached cert unusable, regenerating: %s", e)
    cert_pem, key_pem, sha256_hex = make_self_signed(common_name)
    if cache_dir is not None:
        cert_p.write_bytes(cert_pem)
        key_p.write_bytes(key_pem)
        return cert_p, key_p, sha256_hex
    # No cache requested — write to a temp location.
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="iss-wt-"))
    cert_p = tmp / "wt.crt"
    key_p = tmp / "wt.key"
    cert_p.write_bytes(cert_pem)
    key_p.write_bytes(key_pem)
    return cert_p, key_p, sha256_hex


__all__ = ["make_self_signed", "write_cert"]
