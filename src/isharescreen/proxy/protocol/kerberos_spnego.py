"""Experimental Windows SSPI backend for Apple type-35 Kerberos."""
from .auth import AuthError
from .lkdc import discover_lkdc_realm


class SpnegoGSS:
    def __init__(self):
        try:
            import spnego
        except ImportError:
            raise AuthError("Windows Kerberos requires pyspnego; reinstall isharescreen dependencies") from None
        self.api = spnego
        self.context = None
        self.required = (spnego.ContextReq.mutual_auth | spnego.ContextReq.sequence_detect
                         | spnego.ContextReq.confidentiality | spnego.ContextReq.integrity)

    def start(self, host, username, password, realm, principal, service):
        identity = principal or username
        if not realm:
            if "@" in identity:
                realm = identity.rsplit("@", 1)[1]
            elif service and "/" in service and "@" in service:
                realm = service.rsplit("@", 1)[1]
            else:
                realm = discover_lkdc_realm(host)
        client = principal or (username if "@" in username else f"{username}@{realm}")
        target = service or (f"vnc/{realm}@{realm}" if realm.startswith("LKDC:") else f"vnc/{host}")
        if "/" in target:
            service_name, hostname = target.split("/", 1)
        elif "@" in target:
            service_name, hostname = target.split("@", 1)
        else:
            raise AuthError("Kerberos service must be service/hostname or service@hostname")
        try:
            self.context = self.api.client(
                username=client, password=password or None, hostname=hostname,
                service=service_name, protocol="kerberos", context_req=self.required,
                options=self.api.NegotiateOptions.use_sspi,
            )
            return self.step()
        except AuthError:
            raise
        except Exception:
            # Do not expose library exception text: it may contain credential data.
            raise AuthError("Windows Kerberos could not acquire credentials; check the account and realm configuration") from None

    def step(self, token=None):
        try:
            output = self.context.step(token)
        except self.api.exceptions.NoCredentialError:
            raise AuthError("Windows Kerberos has no usable credentials. Enter the Mac account password; an LKDC realm may also require Windows KDC configuration.") from None
        except Exception:
            raise AuthError("Windows Kerberos ticket exchange failed; SSPI may need a configured KDC mapping for the Mac's LKDC realm. Use SRP until that setup is verified.") from None
        if token is not None:
            if not self.context.complete or self.context.negotiated_protocol != "kerberos":
                raise AuthError("Windows Kerberos context did not complete")
            if self.context.context_attr & self.required != self.required:
                raise AuthError("Windows Kerberos context lacks required security flags")
        return output or b""

    def unwrap(self, token):
        try:
            result = self.context.unwrap(token)
            return result.data, result.encrypted
        except Exception:
            raise AuthError("Windows Kerberos could not unwrap the session key") from None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # pyspnego/sspilib native handles are released with the context object.
        self.context = None
