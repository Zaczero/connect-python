from __future__ import annotations

import asyncio
import contextlib
import json
import struct
from types import SimpleNamespace

import pytest

from connectrpc.code import Code
from connectrpc.codec import Codec
from connectrpc.errors import ConnectError
from connectrpc.method import IdempotencyLevel, MethodInfo
from connectrpc.server import ConnectASGIApplication, Endpoint

pytestmark = pytest.mark.timeout(10)

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
        if send is not None:
            await send(message)
        events.append(message)

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


def _assert_successful_response(events, protocol, shape):
    unary_connect = (protocol, shape) == ("connect", "unary")
    assert events[0]["type"] == "http.response.start"
    assert events[0]["status"] == 200
    assert events[1]["body"] == (
        b"response" if unary_connect else b"\x00\x00\x00\x00\x08response"
    )
    if not unary_connect:
        assert _terminal_status(events, protocol, shape) == [
            "ok" if protocol == "connect" else "0"
        ]
    if protocol == "grpc":
        assert events[-1]["type"] == "http.response.trailers"
        assert events[-1]["more_trailers"] is False
    else:
        assert events[-1]["type"] == "http.response.body"
        assert events[-1]["more_body"] is False


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


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_deadline_interrupts_successful_metadata_cleanup(
    shape, protocol, clock, monkeypatch
):
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    cleaned = []

    class Metadata:
        async def on_start(self, _ctx):
            return None

        async def on_end(self, _token, _ctx, _error):
            try:
                clock.value += 2
                await asyncio.Event().wait()
            finally:
                cleaned.append(True)

    async def unary(_request, _ctx):
        return b"response"

    async def streaming(_request, _ctx):
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
    assert cleaned == [True]
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("stage", ["headers", "body"])
@pytest.mark.parametrize("failure", ["deadline", "cancellation", "send_error"])
@pytest.mark.parametrize("acceptance", ["before_failure", "not_accepted"])
async def test_send_failure_stops_output(
    shape, protocol, stage, failure, acceptance, clock, monkeypatch
):
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    blocked = asyncio.Event()
    attempts = []
    accepted = []
    closed = []
    send_error = BrokenPipeError("connection closed")

    async def handler(_request, _ctx):
        return b"response"

    async def streaming(_request, _ctx):
        try:
            yield b"response"
        finally:
            closed.append(True)

    async def send(message):
        attempts.append(message)
        target = (
            message["type"] == "http.response.start"
            if stage == "headers"
            else message["type"] == "http.response.body"
        )
        if target:
            if acceptance == "before_failure":
                accepted.append(message)
            blocked.set()
            if failure == "send_error":
                raise send_error
            await asyncio.Event().wait()
        accepted.append(message)

    task = asyncio.create_task(
        _invoke(
            _application(
                shape,
                streaming if shape in ("server_stream", "bidi_stream") else handler,
            ),
            protocol,
            shape,
            send=send,
        )
    )
    await blocked.wait()
    if failure == "deadline":
        clock.value += 2
        with pytest.raises(ConnectError) as exc_info:
            await task
        assert exc_info.value.code == Code.DEADLINE_EXCEEDED
    elif failure == "cancellation":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(BrokenPipeError) as exc_info:
            await task
        assert exc_info.value is send_error
    assert [message["type"] for message in attempts] == (
        ["http.response.start"]
        if stage == "headers"
        else ["http.response.start", "http.response.body"]
    )
    assert len(accepted) == (stage == "body") + (acceptance == "before_failure")
    assert closed == ([True] if shape in ("server_stream", "bidi_stream") else [])


@pytest.mark.asyncio
async def test_unary_expiry_after_headers_does_not_restart_response(clock):
    sent = []

    async def handler(_request, _ctx):
        return b"response"

    async def send(message):
        sent.append(message)
        clock.value += 2

    with pytest.raises(ConnectError) as exc_info:
        await _invoke(_application("unary", handler), "connect", "unary", send=send)
    assert exc_info.value.code == Code.DEADLINE_EXCEEDED
    assert [message["type"] for message in sent] == ["http.response.start"]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["server_stream", "bidi_stream"])
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_empty_stream_header_send_failure_stops_output(
    shape, protocol, clock, monkeypatch
):
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    blocked = asyncio.Event()
    attempts = []

    async def handler(_request, _ctx):
        for response in ():
            yield response

    async def send(message):
        attempts.append(message)
        if message["type"] == "http.response.start":
            blocked.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(
        _invoke(_application(shape, handler), protocol, shape, send=send)
    )
    await blocked.wait()
    clock.value += 2
    with pytest.raises(ConnectError) as exc_info:
        await task
    assert exc_info.value.code == Code.DEADLINE_EXCEEDED
    assert [message["type"] for message in attempts] == ["http.response.start"]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize(
    ("protocol", "terminal_type"),
    [
        ("connect", "http.response.body"),
        ("grpc", "http.response.body"),
        ("grpc", "http.response.trailers"),
        ("grpc-web", "http.response.body"),
    ],
)
@pytest.mark.parametrize(
    "failure", ["deadline", "suppressed_deadline", "cancellation", "send_error"]
)
@pytest.mark.parametrize("acceptance", ["before_failure", "not_accepted"])
async def test_terminal_send_is_not_retried(
    shape, protocol, terminal_type, failure, acceptance, clock, monkeypatch
):
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    blocked = asyncio.Event()
    sent = []
    accepted = []
    terminal_attempts = []
    send_error = BrokenPipeError("connection closed")

    async def unary(_request, _ctx):
        return b"response"

    async def streaming(_request, _ctx):
        yield b"response"

    async def send(message):
        sent.append(message)
        if message["type"] != terminal_type or message.get("more_body"):
            accepted.append(message)
            return
        terminal_attempts.append(message)
        if acceptance == "before_failure":
            accepted.append(message)
        blocked.set()
        if failure == "send_error":
            raise send_error
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if failure != "suppressed_deadline":
                raise
        if acceptance == "not_accepted":
            accepted.append(message)

    task = asyncio.create_task(
        _invoke(
            _application(
                shape, streaming if shape in ("server_stream", "bidi_stream") else unary
            ),
            protocol,
            shape,
            send=send,
        )
    )
    await blocked.wait()
    if failure in ("deadline", "suppressed_deadline"):
        clock.value += 2
        if failure == "suppressed_deadline":
            events = await task
            if protocol == "grpc" and terminal_type == "http.response.body":
                assert _terminal_status(events, protocol, shape) == ["4"]
                assert (
                    sum(
                        message["type"] == "http.response.body"
                        and not message.get("more_body")
                        for message in accepted
                    )
                    == 1
                )
                assert (
                    sum(
                        message["type"] == "http.response.trailers"
                        for message in accepted
                    )
                    == 1
                )
            else:
                _assert_successful_response(events, protocol, shape)
        else:
            with pytest.raises(ConnectError) as exc_info:
                await task
            assert exc_info.value.code == Code.DEADLINE_EXCEEDED
    elif failure == "cancellation":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(BrokenPipeError) as exc_info:
            await task
        assert exc_info.value is send_error
    assert len(terminal_attempts) == 1
    if failure != "suppressed_deadline":
        expected_attempts = (
            2
            if (protocol, shape) == ("connect", "unary")
            else 4
            if terminal_type == "http.response.trailers"
            else 3
        )
        assert len(sent) == expected_attempts
        assert len(accepted) == expected_attempts - (acceptance == "not_accepted")
    if (
        protocol == "grpc"
        and terminal_type == "http.response.body"
        and failure != "suppressed_deadline"
    ):
        assert all(message["type"] != "http.response.trailers" for message in sent)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_final_send_completion_is_not_reclassified(shape, protocol, clock):
    completed = []

    async def unary(_request, _ctx):
        return b"response"

    async def streaming(_request, _ctx):
        yield b"response"

    async def send(message):
        terminal = (
            message["type"] == "http.response.trailers"
            if protocol == "grpc"
            else message["type"] == "http.response.body"
            and not message.get("more_body")
        )
        if terminal:
            completed.append(message)
            clock.value += 2

    events = await _invoke(
        _application(
            shape, streaming if shape in ("server_stream", "bidi_stream") else unary
        ),
        protocol,
        shape,
        send=send,
    )
    assert len(completed) == 1
    _assert_successful_response(events, protocol, shape)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
async def test_grpc_body_completion_expiry_sends_first_error_trailers(shape, clock):
    async def unary(_request, _ctx):
        return b"response"

    async def streaming(_request, _ctx):
        yield b"response"

    async def send(message):
        if message["type"] == "http.response.body" and not message.get("more_body"):
            clock.value += 2

    events = await _invoke(
        _application(
            shape, streaming if shape in ("server_stream", "bidi_stream") else unary
        ),
        "grpc",
        shape,
        send=send,
    )
    assert [message["type"] for message in events] == [
        "http.response.start",
        "http.response.body",
        "http.response.body",
        "http.response.trailers",
    ]
    assert events[-2]["body"] == b""
    assert events[-2]["more_body"] is False
    assert _terminal_status(events, "grpc", shape) == ["4"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol", "shape"),
    [
        (protocol, shape)
        for protocol in _PROTOCOLS
        for shape in _SHAPES
        if (protocol, shape) != ("connect", "unary")
    ],
)
async def test_stream_header_completion_expiry_does_not_encode_message(
    protocol, shape, clock
):
    class FailingCodec(_BytesCodec):
        def encode(self, message):
            pytest.fail(f"An expired response entered its encoder: {message!r}")

    async def unary(_request, _ctx):
        return b"response"

    async def streaming(_request, _ctx):
        yield b"response"

    async def send(message):
        if message["type"] == "http.response.start":
            clock.value += 2

    events = await _invoke(
        _application(
            shape,
            streaming if shape in ("server_stream", "bidi_stream") else unary,
            codec=FailingCodec(),
        ),
        protocol,
        shape,
        send=send,
    )
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]
    assert sum(message["type"] == "http.response.start" for message in events) == 1
    assert all(
        not message.get("more_body")
        for message in events
        if message["type"] == "http.response.body"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_deadline_unwinds_all_metadata_in_request_task(
    shape, protocol, clock, monkeypatch
):
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    cleanup = []
    tasks = []

    class Outer:
        async def on_start(self, _ctx):
            tasks.append(asyncio.current_task())

        async def on_end(self, _token, _ctx, error):
            tasks.append(asyncio.current_task())
            cleanup.append(("outer", error.code))

    class Inner:
        async def on_start(self, _ctx):
            tasks.append(asyncio.current_task())

        async def on_end(self, _token, _ctx, _error):
            tasks.append(asyncio.current_task())
            try:
                clock.value += 2
                await asyncio.Event().wait()
            finally:
                cleanup.append("inner closed")

    async def unary(_request, _ctx):
        tasks.append(asyncio.current_task())
        return b"response"

    async def streaming(_request, _ctx):
        await unary(_request, _ctx)
        yield b"response"

    events = await _invoke(
        _application(
            shape,
            streaming if shape in ("server_stream", "bidi_stream") else unary,
            interceptors=[Outer(), Inner()],
        ),
        protocol,
        shape,
    )
    assert cleanup == ["inner closed", ("outer", Code.CANCELED)]
    assert all(task is asyncio.current_task() for task in tasks)
    assert _terminal_status(events, protocol, shape) == [
        "deadline_exceeded" if protocol == "connect" else "4"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["server_stream", "bidi_stream"])
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("failure", ["deadline", "cancellation", "send_error"])
async def test_send_failure_survives_iterator_and_metadata_cleanup_errors(
    shape, protocol, failure, clock, monkeypatch
):
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.value)
    entered = asyncio.Event()
    cleanup = []
    attempts = []
    send_error = BrokenPipeError("connection closed")
    metadata_error = LookupError("metadata cleanup failed")
    iterator_error = ValueError("iterator cleanup failed")

    class Metadata:
        async def on_start(self, _ctx):
            return None

        async def on_end(self, _token, _ctx, _error):
            cleanup.append("metadata")
            raise metadata_error

    async def handler(_request, _ctx):
        try:
            yield b"response"
        finally:
            cleanup.append("iterator")
            raise iterator_error

    async def send(message):
        attempts.append(message)
        if message.get("more_body"):
            entered.set()
            if failure == "send_error":
                raise send_error
            await asyncio.Event().wait()

    task = asyncio.create_task(
        _invoke(
            _application(shape, handler, interceptors=[Metadata()]),
            protocol,
            shape,
            send=send,
        )
    )
    await entered.wait()
    if failure == "deadline":
        clock.value += 2
        with pytest.raises(ConnectError) as exc_info:
            await task
        assert exc_info.value.code == Code.DEADLINE_EXCEEDED
    elif failure == "cancellation":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(BrokenPipeError) as exc_info:
            await task
        assert exc_info.value is send_error
    assert [message["type"] for message in attempts] == [
        "http.response.start",
        "http.response.body",
    ]
    assert cleanup == ["iterator", "metadata"]
