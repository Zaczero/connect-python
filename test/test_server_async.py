from __future__ import annotations

import asyncio
import contextlib
import json
import struct
from types import SimpleNamespace

import pytest

from connectrpc.codec import Codec
from connectrpc.method import IdempotencyLevel, MethodInfo
from connectrpc.server import ConnectASGIApplication, Endpoint

_PATH = "/test.Service/Call"
_SHAPES = ("unary", "client_stream", "server_stream", "bidi_stream")
_PROTOCOLS = ("connect", "grpc", "grpc-web")


class _BytesCodec(Codec[bytes, bytes]):
    def name(self) -> str:
        return "proto"

    def encode(self, message: bytes) -> bytes:
        return message

    def decode(self, data: bytes | bytearray, message_class: type[bytes]) -> bytes:
        return message_class(data)


class _Application(ConnectASGIApplication[object]):
    @property
    def path(self) -> str:
        return "/test.Service"


def _application(shape, handler, *, interceptors=(), codec=None):
    method = MethodInfo(
        name="Call",
        service_name="test.Service",
        input=bytes,
        output=bytes,
        idempotency_level=IdempotencyLevel.UNKNOWN,
    )
    endpoint = getattr(Endpoint, shape)(method, handler)
    return _Application(
        service=object(),
        endpoints=lambda _: {_PATH: endpoint},
        codecs=[_BytesCodec() if codec is None else codec],
        compressions=[],
        interceptors=interceptors,
    )


def _scope(protocol, shape, *, timeout_ms="1000"):
    if protocol == "connect":
        content_type = (
            "application/proto" if shape == "unary" else "application/connect+proto"
        )
        timeout_header = b"connect-timeout-ms"
    else:
        content_type = f"application/{protocol}+proto"
        timeout_header = b"grpc-timeout"
        timeout_ms += "m"
    return {
        "type": "http",
        "http_version": "2",
        "method": "POST",
        "scheme": "http",
        "path": _PATH,
        "raw_path": _PATH.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", content_type.encode()),
            (timeout_header, timeout_ms.encode()),
        ],
        "extensions": {"http.response.trailers": {}},
    }


async def _invoke(app, protocol, shape, *, timeout_ms="1000", send=None):
    request = b"request"
    if protocol != "connect" or shape != "unary":
        request = struct.pack(">BI", 0, len(request)) + request
    incoming = asyncio.Queue()
    incoming.put_nowait({"type": "http.request", "body": request, "more_body": False})
    events = []

    async def record(message):
        events.append(message)
        if send is not None:
            await send(message)

    await app(_scope(protocol, shape, timeout_ms=timeout_ms), incoming.get, record)
    return events


def _terminal_status(events, protocol, shape):
    if protocol == "grpc":
        return [
            value.decode()
            for message in events
            for key, value in message.get("headers", ())
            if key == b"grpc-status"
        ]
    body = b"".join(
        message.get("body", b"")
        for message in events
        if message["type"] == "http.response.body"
    )
    if protocol == "connect" and shape == "unary":
        return [json.loads(body)["code"]]
    offset = 0
    statuses = []
    while offset < len(body):
        flag, size = struct.unpack(">BI", body[offset : offset + 5])
        payload = body[offset + 5 : offset + 5 + size]
        offset += 5 + size
        if protocol == "grpc-web" and flag == 0x80:
            statuses.extend(
                line.split(b":", 1)[1].strip().decode()
                for line in payload.split(b"\r\n")
                if line.startswith(b"grpc-status:")
            )
        elif protocol == "connect" and flag == 2:
            statuses.append(json.loads(payload).get("error", {}).get("code", "ok"))
    return statuses


@pytest.fixture
def clock(monkeypatch):
    now = SimpleNamespace(value=100.0)
    # Keep synchronous expiry separate from the event loop's timer delivery.
    monkeypatch.setattr(
        "connectrpc.request.time", SimpleNamespace(monotonic=lambda: now.value)
    )
    return now


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_expired_deadline_does_not_start_metadata_or_handler(shape, protocol):
    called = []

    class Metadata:
        async def on_start(self, _ctx):
            called.append("metadata")

        async def on_end(self, _token, _ctx, _error):
            called.append("metadata end")

    async def unary(_request, _ctx):
        called.append("handler")
        return b"response"

    async def streaming(_request, _ctx):
        called.append("handler")
        yield b"response"

    app = _application(
        shape,
        streaming if shape in ("server_stream", "bidi_stream") else unary,
        interceptors=[Metadata()],
    )
    events = await _invoke(app, protocol, shape, timeout_ms="0")
    assert called == []
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("completion", ["synchronous", "suppressed_cancellation"])
async def test_metadata_expiry_does_not_invoke_handler(
    shape, protocol, completion, clock, monkeypatch
):
    if completion == "suppressed_cancellation":
        monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    called = []

    class Metadata:
        async def on_start(self, _ctx):
            clock.value += 2
            if completion == "suppressed_cancellation":
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.Event().wait()

        async def on_end(self, _token, _ctx, _error):
            called.append("metadata end")

    async def unary(_request, _ctx):
        called.append("handler")
        return b"response"

    async def streaming(_request, _ctx):
        called.append("handler")
        yield b"response"

    events = await _invoke(
        _application(
            shape,
            streaming if shape in ("server_stream", "bidi_stream") else unary,
            interceptors=[Metadata()],
        ),
        protocol,
        shape,
    )
    assert called == ["metadata end"]
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["unary", "server_stream"])
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_decoding_expiry_does_not_invoke_handler(shape, protocol, clock):
    called = []

    class ExpiringCodec(_BytesCodec):
        def decode(self, data, message_class):
            clock.value += 2
            return super().decode(data, message_class)

    async def unary(_request, _ctx):
        called.append(True)
        return b"response"

    async def streaming(_request, _ctx):
        called.append(True)
        yield b"response"

    events = await _invoke(
        _application(
            shape,
            streaming if shape == "server_stream" else unary,
            codec=ExpiringCodec(),
        ),
        protocol,
        shape,
    )
    assert called == []
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("stage", ["handler", "serialization", "metadata_end"])
async def test_synchronous_expiry_does_not_send_success(shape, protocol, stage, clock):
    class ExpiringCodec(_BytesCodec):
        def encode(self, message):
            if stage == "serialization":
                clock.value += 2
            return message

    class Metadata:
        async def on_start(self, _ctx):
            return None

        async def on_end(self, _token, _ctx, _error):
            if stage == "metadata_end":
                clock.value += 2

    async def unary(_request, _ctx):
        if stage == "handler":
            clock.value += 2
        return b"response"

    async def streaming(_request, _ctx):
        if stage == "handler":
            clock.value += 2
        yield b"response"

    events = await _invoke(
        _application(
            shape,
            streaming if shape in ("server_stream", "bidi_stream") else unary,
            codec=ExpiringCodec(),
            interceptors=[Metadata()],
        ),
        protocol,
        shape,
    )
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("completion", ["synchronous", "suppressed_cancellation"])
async def test_expired_metadata_does_not_admit_another_interceptor(
    shape, protocol, completion, clock, monkeypatch
):
    if completion == "suppressed_cancellation":
        monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    called = []

    class First:
        async def on_start(self, _ctx):
            called.append("first")
            clock.value += 2
            if completion == "suppressed_cancellation":
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.Event().wait()

        async def on_end(self, _token, _ctx, _error):
            called.append("first end")

    class Second(First):
        async def on_start(self, _ctx):
            called.append("second")

    async def unary(_request, _ctx):
        called.append("handler")
        return b"response"

    async def streaming(_request, _ctx):
        called.append("handler")
        yield b"response"

    events = await _invoke(
        _application(
            shape,
            streaming if shape in ("server_stream", "bidi_stream") else unary,
            interceptors=[First(), Second()],
        ),
        protocol,
        shape,
    )
    assert called == ["first", "first end"]
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_known_expiry_is_reported_before_successful_metadata_end(
    shape, protocol, clock
):
    errors = []

    class Metadata:
        async def on_start(self, _ctx):
            return None

        async def on_end(self, _token, _ctx, error):
            errors.append(error)

    async def unary(_request, _ctx):
        clock.value += 2
        return b"response"

    async def exhausted(_request, _ctx):
        clock.value += 2
        for response in ():
            yield response

    events = await _invoke(
        _application(
            shape,
            exhausted if shape in ("server_stream", "bidi_stream") else unary,
            interceptors=[Metadata()],
        ),
        protocol,
        shape,
    )
    assert len(errors) == 1
    assert errors[0] is not None
    assert errors[0].code.value == "deadline_exceeded"
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["server_stream", "bidi_stream"])
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("completion", ["synchronous", "suppressed_cancellation"])
async def test_expired_message_send_does_not_advance_response_iterator(
    shape, protocol, completion, clock, monkeypatch
):
    if completion == "suppressed_cancellation":
        monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    advanced = []
    closed = []

    async def handler(_request, _ctx):
        try:
            yield b"first"
            advanced.append(True)
            yield b"second"
        finally:
            closed.append(True)

    async def send(message):
        if message.get("more_body"):
            clock.value += 2
            if completion == "suppressed_cancellation":
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.Event().wait()

    events = await _invoke(_application(shape, handler), protocol, shape, send=send)
    assert advanced == []
    assert closed == [True]
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_suspended_handler_deadline_sends_protocol_error(
    shape, protocol, clock, monkeypatch
):
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    cleaned = []

    async def unary(_request, _ctx):
        try:
            clock.value += 2
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    async def streaming(_request, _ctx):
        await unary(_request, _ctx)
        yield b"response"

    events = await _invoke(
        _application(
            shape, streaming if shape in ("server_stream", "bidi_stream") else unary
        ),
        protocol,
        shape,
    )
    assert cleaned == [True]
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]
