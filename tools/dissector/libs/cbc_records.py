from __future__ import annotations

from Crypto.Cipher import AES  # type: ignore[import-untyped]

from .message_parser import (
    guess_client_message_type,
    guess_server_message_type,
    parse_client_message,
    parse_server_message,
)
from .types import DecodedRecord, RekeyInfo, ReassembledStream


def _hex(data: bytes) -> str:
    return data.hex()


def decrypt_ecb_block(key_hex: str, block: bytes) -> bytes:
    cipher = AES.new(bytes.fromhex(key_hex), AES.MODE_ECB)
    return cipher.decrypt(block)


def decrypt_cbc_block(key_hex: str, iv_hex: str, ciphertext: bytes) -> bytes:
    cipher = AES.new(bytes.fromhex(key_hex), AES.MODE_CBC, iv=bytes.fromhex(iv_hex))
    return cipher.decrypt(ciphertext)


def find_rekey_info(server_stream: ReassembledStream, initial_key_hex: str) -> RekeyInfo | None:
    rekeys = find_all_rekey_info(server_stream, initial_key_hex)
    return rekeys[0] if rekeys else None


def find_all_rekey_info(server_stream: ReassembledStream, initial_key_hex: str) -> list[RekeyInfo]:
    rekeys = []
    data = server_stream.data
    for i in range(len(data) - 51):
        if data[i] != 0:
            continue
        if int.from_bytes(data[i + 12:i + 16], "big") != 0x44F:
            continue
        counter = int.from_bytes(data[i + 16:i + 20], "big")
        block1 = data[i + 20:i + 36]
        block2 = data[i + 36:i + 52]
        next_key = decrypt_ecb_block(initial_key_hex, block1)
        next_iv = decrypt_ecb_block(initial_key_hex, block2)

        ts, fn = None, None
        for span in server_stream.spans:
            if span.start <= i < span.end:
                ts, fn = span.timestamp_epoch, span.frame_number
                break

        rekeys.append(RekeyInfo(
            server_offset=i,
            frame_number=fn,
            timestamp_epoch=ts,
            counter=counter,
            wrap_key_hex=initial_key_hex,
            next_key_hex=_hex(next_key),
            next_iv_hex=_hex(next_iv),
        ))
    return rekeys


def find_client_cbc_start(
    client_stream: ReassembledStream,
    rekey_timestamp: float,
) -> int | None:
    for span in client_stream.spans:
        if span.timestamp_epoch < rekey_timestamp:
            continue
        segment = client_stream.data[span.start : span.end]
        if len(segment) >= 8 and segment[0] == 0x12:
            return span.start + 8
    return None


def find_client_cbc_start_by_preface(
    client_stream: ReassembledStream,
    rekey_timestamp: float,
    key_hex: str,
    iv_hex: str,
) -> int | None:
    data = client_stream.data

    def valid_record_at(offset: int, expected_msg_id: int | None = None) -> bool:
        if offset < 0 or offset + 18 > len(data):
            return False
        cipher_len = int.from_bytes(data[offset : offset + 2], "big")
        if cipher_len == 0 or cipher_len % 16 != 0 or offset + 2 + cipher_len > len(data):
            return False
        ciphertext = data[offset + 2 : offset + 2 + cipher_len]
        plain = decrypt_cbc_block(key_hex, iv_hex, ciphertext)
        if len(plain) < 2:
            return False
        body_len = int.from_bytes(plain[0:2], "big")
        if 2 + body_len > len(plain):
            return False
        body = plain[2 : 2 + body_len]
        if not body:
            return False
        if expected_msg_id is not None and body[0] != expected_msg_id:
            return False
        return True

    for span in client_stream.spans:
        if span.timestamp_epoch < rekey_timestamp:
            continue
        candidates = [span.start]
        if len(data) >= span.start + 8 and data[span.start] == 0x12:
            candidates.append(span.start + 8)
        for candidate in candidates:
            if not valid_record_at(candidate, 0x1D):
                continue
            cipher_len = int.from_bytes(data[candidate : candidate + 2], "big")
            second = candidate + 2 + cipher_len
            if valid_record_at(second, 0x02):
                return candidate
            return candidate
    return None


def decode_cbc_records(
    stream_id: str,
    direction: str,
    stream: ReassembledStream,
    start_offset: int,
    key_hex: str,
    iv_hex: str,
    base_seq: int | None,
    stop_after_epoch: float | None = None,
    include_plain_hex: bool = True,
    include_parsed: bool = True,
    rekey_offsets: list[tuple[int, str, str]] | None = None,
) -> list[DecodedRecord]:
    return list(
        iter_cbc_records(
            stream_id=stream_id,
            direction=direction,
            stream=stream,
            start_offset=start_offset,
            key_hex=key_hex,
            iv_hex=iv_hex,
            base_seq=base_seq,
            stop_after_epoch=stop_after_epoch,
            include_plain_hex=include_plain_hex,
            include_parsed=include_parsed,
            rekey_offsets=rekey_offsets,
        )
    )


def iter_cbc_records(
    stream_id: str,
    direction: str,
    stream: ReassembledStream,
    start_offset: int,
    key_hex: str,
    iv_hex: str,
    base_seq: int | None,
    stop_after_epoch: float | None = None,
    include_plain_hex: bool = True,
    include_parsed: bool = True,
    rekey_offsets: list[tuple[int, str, str]] | None = None,
):
    data = stream.data[start_offset:]
    pos = 0
    record_index = 0
    current_key_hex = key_hex
    current_iv_hex = iv_hex
    spans = stream.spans
    span_idx = 0
    rekey_idx = 0

    def current_meta(offset: int) -> tuple[float | None, int | None]:
        nonlocal span_idx
        while span_idx + 1 < len(spans) and offset >= spans[span_idx].end:
            span_idx += 1
        if spans and spans[span_idx].start <= offset < spans[span_idx].end:
            span = spans[span_idx]
            return span.timestamp_epoch, span.frame_number
        if spans:
            span = spans[-1]
            return span.timestamp_epoch, span.frame_number
        return None, None

    while pos + 2 <= len(data):
        # Check for rekey at current position
        if rekey_offsets and rekey_idx < len(rekey_offsets):
            rekey_offset, rekey_key, rekey_iv = rekey_offsets[rekey_idx]
            if start_offset + pos >= rekey_offset:
                current_key_hex = rekey_key
                current_iv_hex = rekey_iv
                rekey_idx += 1

        cipher_len = int.from_bytes(data[pos : pos + 2], "big")
        if cipher_len == 0:
            break

        timestamp_epoch, frame_number = current_meta(start_offset + pos)
        if cipher_len % 16 != 0:
            yield DecodedRecord(
                stream_id=stream_id,
                record_index=record_index + 1,
                direction=direction,
                timestamp_epoch=timestamp_epoch,
                frame_number=frame_number,
                tcp_offset=start_offset + pos,
                cipher_len=cipher_len,
                msg_id=None,
                msg_name=f"{direction}_malformed_cipher_len",
                record_kind="malformed",
                plain_len=0,
                plain_hex="",
                body_bytes=b"",
                warnings=[f"cipher_len {cipher_len} is not 16-byte aligned"],
            )
            break

        if pos + 2 + cipher_len > len(data):
            yield DecodedRecord(
                stream_id=stream_id,
                record_index=record_index + 1,
                direction=direction,
                timestamp_epoch=timestamp_epoch,
                frame_number=frame_number,
                tcp_offset=start_offset + pos,
                cipher_len=cipher_len,
                msg_id=None,
                msg_name=f"{direction}_partial_record",
                record_kind="partial",
                plain_len=0,
                plain_hex="",
                body_bytes=b"",
                warnings=["truncated record at end of stream"],
            )
            break

        record_index += 1
        if stop_after_epoch is not None and timestamp_epoch is not None and timestamp_epoch > stop_after_epoch:
            break

        ciphertext = data[pos + 2 : pos + 2 + cipher_len]
        plain = decrypt_cbc_block(current_key_hex, current_iv_hex, ciphertext)
        current_iv_hex = _hex(ciphertext[-16:])

        if len(plain) < 2:
            yield DecodedRecord(
                stream_id=stream_id,
                record_index=record_index,
                direction=direction,
                timestamp_epoch=timestamp_epoch,
                frame_number=frame_number,
                tcp_offset=start_offset + pos,
                cipher_len=cipher_len,
                msg_id=None,
                msg_name=f"{direction}_short_plaintext",
                record_kind="malformed",
                plain_len=len(plain),
                plain_hex=_hex(plain),
                body_bytes=b"",
                warnings=["missing inner plaintext length"],
            )
            break

        body_len = int.from_bytes(plain[0:2], "big")
        body_end = 2 + body_len
        if body_end > len(plain):
            yield DecodedRecord(
                stream_id=stream_id,
                record_index=record_index,
                direction=direction,
                timestamp_epoch=timestamp_epoch,
                frame_number=frame_number,
                tcp_offset=start_offset + pos,
                cipher_len=cipher_len,
                msg_id=None,
                msg_name=f"{direction}_declared_body_too_long",
                record_kind="malformed",
                plain_len=len(plain),
                plain_hex=_hex(plain),
                body_bytes=b"",
                warnings=[f"declared body_len={body_len} exceeds plaintext len={len(plain)}"],
            )
            break

        body = plain[2:body_end]
        tail_len = len(plain) - body_end
        filler_hex = ""
        sha1_hex = ""
        if tail_len >= 20 and include_plain_hex:
            filler = plain[body_end:-20]
            trailing_sha1 = plain[-20:]
            filler_hex = _hex(filler)
            sha1_hex = _hex(trailing_sha1)

        msg_id = body[0] if body else None
        if include_parsed:
            if direction == "client":
                msg_name, parsed = parse_client_message(body, ecb_key_hex=current_key_hex)
            else:
                msg_name, parsed = parse_server_message(body)
        else:
            msg_name = guess_client_message_type(msg_id) if direction == "client" else guess_server_message_type(msg_id)
            parsed = {}

        yield DecodedRecord(
            stream_id=stream_id,
            record_index=record_index,
            direction=direction,
            timestamp_epoch=timestamp_epoch,
            frame_number=frame_number,
            tcp_offset=start_offset + pos,
            cipher_len=cipher_len,
            msg_id=msg_id,
            msg_name=msg_name,
            record_kind=msg_name,
            plain_len=len(plain),
            plain_hex=_hex(body) if include_plain_hex else "",
            body_bytes=body,
            filler_hex=filler_hex,
            trailing_sha1_hex=sha1_hex,
            parsed=parsed,
        )
        pos += 2 + cipher_len
