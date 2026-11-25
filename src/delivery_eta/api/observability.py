"""Logs, metrics, and the checks that wrap every request."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id: ContextVar[str] = ContextVar("request_id", default="-")
log = logging.getLogger("delivery_eta")

_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
}
PAGE_POLICY = "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"


class JsonLines(logging.Formatter):
    """One JSON object per line, which log services can search."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "severity": record.levelname,
            "message": record.getMessage(),
            "request_id": request_id.get(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            line["error"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLines())
    logging.basicConfig(handlers=[handler], level=level, force=True)


class Metrics:
    """Kept per app rather than global, so tests and reloads don't clash."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "eta_requests_total",
            "Requests handled.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "eta_request_seconds",
            "Time to answer a request.",
            ["route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
            registry=self.registry,
        )
        self.minutes = Histogram(
            "eta_predicted_minutes",
            "Predicted delivery times. A shift here means orders have changed.",
            buckets=(10, 15, 20, 25, 30, 35, 40, 45, 50, 60),
            registry=self.registry,
        )
        self.missing = Counter(
            "eta_missing_inputs_total",
            "Inputs left out of requests.",
            ["input"],
            registry=self.registry,
        )
        self.unusual = Counter(
            "eta_unusual_orders_total",
            "Orders outside what the model learned from.",
            registry=self.registry,
        )
        self.model = Gauge(
            "eta_model_info", "The loaded model.", ["version"], registry=self.registry
        )


class RequestGuard:
    """Gives each request an id, refuses huge bodies, adds security headers, and logs the result."""

    def __init__(self, app: ASGIApp, metrics: Metrics, max_body_bytes: int) -> None:
        self.app = app
        self.metrics = metrics
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        given = headers.get(b"x-request-id", b"").decode("latin-1")
        rid = given if _SAFE_ID.fullmatch(given) else uuid.uuid4().hex
        token = request_id.set(rid)
        start = time.perf_counter()
        status = 500
        responded = False
        received = 0
        too_large = f"The request is larger than {self.max_body_bytes:,} bytes."

        async def guarded_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # FastAPI passes this one through; any other error would become a 400.
                    raise HTTPException(413, too_large)
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal status, responded
            if message["type"] == "http.response.start":
                status, responded = message["status"], True
                extra = {**SECURITY_HEADERS, "x-request-id": rid}
                path = scope["path"]
                if path == "/" or path.startswith("/static"):
                    extra["content-security-policy"] = PAGE_POLICY
                message["headers"] = [
                    *message["headers"],
                    *((k.encode(), v.encode()) for k, v in extra.items()),
                ]
            await send(message)

        async def refuse(code: int, detail: str) -> None:
            body = json.dumps({"detail": detail, "request_id": rid}).encode()
            await guarded_send(
                {
                    "type": "http.response.start",
                    "status": code,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await guarded_send({"type": "http.response.body", "body": body})

        try:
            declared = headers.get(b"content-length", b"0")
            if declared.isdigit() and int(declared) > self.max_body_bytes:
                await refuse(413, too_large)
            else:
                await self.app(scope, guarded_receive, guarded_send)
        except Exception:
            log.exception("Unhandled error")
            if not responded:
                await refuse(500, "Something went wrong on our side.")
        finally:
            seconds = time.perf_counter() - start
            route = getattr(scope.get("route"), "path", "unmatched")
            self.metrics.requests.labels(scope["method"], route, str(status)).inc()
            self.metrics.latency.labels(route).observe(seconds)
            if route not in ("/health", "/ready"):
                log.info(
                    "%s %s %s",
                    scope["method"],
                    scope["path"],
                    status,
                    extra={"fields": {"status": status, "ms": round(seconds * 1000, 1)}},
                )
            request_id.reset(token)
