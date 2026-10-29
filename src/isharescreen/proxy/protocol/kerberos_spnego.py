"""Experimental Windows SSPI backend for Apple type-35 Kerberos."""
from .auth import AuthError
from .lkdc import discover_lkdc_realm


def _diagnostic_code(error):
    """Expose numeric SSPI status only, never native exception text or secrets."""
    native = getattr(error, "base_error", None)
    code = getattr(native, "winerror", None)
    if not isinstance(code, int):
        code = getattr(error, "winerror", None)
    return f" (SSPI 0x{code & 0xffffffff:08X})" if isinstance(code, int) else ""


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
        except Exception as error:
            # Do not expose library exception text: it may contain credential data.
            raise AuthError("Windows Kerberos could not acquire credentials" + _diagnostic_code(error) + "; check the account and realm configuration") from None

    def step(self, token=None):
        try:
            output = self.context.step(token)
        except self.api.exceptions.NoCredentialError as error:
            raise AuthError("Windows Kerberos has no usable credentials" + _diagnostic_code(error) + ". Enter the Mac account password; an LKDC realm may also require Windows KDC configuration.") from None
        except Exception as error:
            raise AuthError("Windows Kerberos ticket exchange failed" + _diagnostic_code(error) + "; check KDC routing, account credentials and service identity. The status code is needed to distinguish these causes.") from None
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
