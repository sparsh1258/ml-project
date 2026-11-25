"""Checks that a running copy really works. Needs only Python.

python scripts/smoke_test.py http://localhost:8000 [--key KEY] [--wait 60]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def call(url: str, body: Any = None, key: str = "") -> tuple[int, str]:
    headers = {"Content-Type": "application/json", **({"X-API-Key": key} if key else {})}
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data, headers)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=15) as reply:  # noqa: S310
            return reply.status, reply.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--key", default="", help="an API key, if the server needs one")
    parser.add_argument("--wait", type=int, default=0, help="seconds to wait for it to start")
    args = parser.parse_args()
    if not args.url.startswith(("http://", "https://")):
        parser.error("the address must start with http:// or https://")
    url = args.url.rstrip("/")

    deadline = time.monotonic() + args.wait
    while True:
        try:
            status, text = call(f"{url}/ready")
            if status == 200:
                break
        except OSError as e:
            text = str(e)
        if time.monotonic() >= deadline:
            print(f"FAIL  not ready: {text[:200]}")
            return 1
        time.sleep(1)

    order = {"distance_km": 8, "ordered_at": "2026-01-05T20:00:00", "traffic": "jam"}
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {name}{'' if ok else ': ' + detail[:200]}")

    status, text = call(f"{url}/api/v1/predict?explain=true", order, args.key)
    reply = json.loads(text) if status == 200 else {}
    sensible = bool(reply) and 5 < reply["low"] <= reply["minutes"] <= reply["high"] < 90
    check("predicts an order", sensible, text)
    status, text = call(f"{url}/api/v1/predict/batch", {"orders": [order, order]}, args.key)
    check("predicts a batch", status == 200 and len(json.loads(text)["estimates"]) == 2, text)
    status, text = call(f"{url}/api/v1/predict", {"distance_km": -1}, args.key)
    check("refuses a bad order", status == 422, text)
    status, text = call(f"{url}/")
    check("serves the page", status == 200 and "Delivery ETA" in text, text)
    status, text = call(f"{url}/api/v1/model")
    check("describes its model", status == 200, text)
    if args.key:
        status, text = call(f"{url}/api/v1/predict", order)
        check("refuses callers without a key", status == 401, text)
    print(f"model {reply.get('model_version', 'unknown')}, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
