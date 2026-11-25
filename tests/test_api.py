from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from delivery_eta.api.app import create_app
from delivery_eta.api.observability import JsonLines
from delivery_eta.api.schemas import OrderIn
from delivery_eta.api.settings import Settings
from delivery_eta.artifact import ArtifactError
from delivery_eta.features import LOCAL_TIME

from .conftest import a_request

MakeClient = Callable[..., TestClient]
KEY = "a-long-enough-test-key"


def predict(client: TestClient, body: dict[str, Any], **kwargs: Any):
    return client.post("/api/v1/predict", json=body, **kwargs)


# --- Predictions ------------------------------------------------------------------------


def test_a_full_order(client: TestClient) -> None:
    response = predict(client, a_request())
    assert response.status_code == 200
    body = response.json()
    assert body["low"] <= body["minutes"] <= body["high"]
    assert body["missing"] == []
    assert body["warnings"] == []
    assert body["factors"] is None
    assert body["distance_km"] == 6
    assert body["model_version"] == client.get("/ready").json()["model_version"]


def test_the_least_an_order_needs(client: TestClient) -> None:
    response = predict(client, {"distance_km": 5})
    assert response.status_code == 200
    assert len(response.json()["missing"]) == 10


def test_places_and_distance_give_the_same_answer(client: TestClient) -> None:
    places = {
        "restaurant": {"lat": 22.745, "lon": 75.892},
        "customer": {"lat": 22.765, "lon": 75.912},
    }
    by_place = predict(client, a_request(distance_km=None, **places)).json()
    by_distance = predict(client, a_request(distance_km=by_place["distance_km"])).json()
    assert by_place["minutes"] == pytest.approx(by_distance["minutes"], abs=0.2)


def test_explanations(client: TestClient) -> None:
    factors = predict(client, a_request(), params={"explain": True}).json()["factors"]
    assert len(factors) == 12
    assert set(factors[0]) == {"input", "minutes"}


def test_batches_keep_their_order(client: TestClient) -> None:
    orders = [a_request(distance_km=d) for d in (12, 2, 7)]
    response = client.post("/api/v1/predict/batch", json={"orders": orders})
    assert response.status_code == 200
    assert [e["distance_km"] for e in response.json()["estimates"]] == [12, 2, 7]
    single = [predict(client, o).json()["minutes"] for o in orders]
    assert [e["minutes"] for e in response.json()["estimates"]] == single


def test_batch_limits(make_client: MakeClient) -> None:
    client = make_client(max_batch=2)
    too_many = client.post("/api/v1/predict/batch", json={"orders": [a_request()] * 3})
    assert too_many.status_code == 413
    assert "limit is 2" in too_many.json()["detail"]
    assert client.post("/api/v1/predict/batch", json={"orders": []}).status_code == 422


@pytest.mark.parametrize(
    ("given", "hour"),
    [
        ("2026-11-25T20:30:00", 20),  # no offset: Indian time
        ("2026-11-25T15:00:00Z", 20),  # 20:30 in India
        ("2026-11-25T23:45:00+05:30", 23),
    ],
)
def test_times_are_read_as_indian_time(given: str, hour: int) -> None:
    order = OrderIn(distance_km=5, ordered_at=given)  # type: ignore[arg-type]
    assert order.to_order(datetime.now(LOCAL_TIME))["order_hour"] == hour


def test_no_time_means_now() -> None:
    now = datetime(2026, 11, 25, 13, 5, tzinfo=LOCAL_TIME)
    assert OrderIn(distance_km=5).to_order(now)["order_hour"] == 13


# --- Bad input --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "where", "words"),
    [
        ({"distance_km": None}, "", "both restaurant and customer"),
        ({"distance_km": -1}, "distance_km", "greater than 0"),
        ({"distance_km": 51}, "distance_km", "less than or equal to 50"),
        ({"restaurant": {"lat": 22, "lon": 75}}, "", "not both"),
        ({"weather": "raining"}, "weather", "Input should be"),
        ({"rider_rating": 6}, "rider_rating", "less than or equal to 5"),
        ({"rider_age": 12}, "rider_age", "greater than or equal to 18"),
        ({"trafic": "jam"}, "trafic", "Extra inputs"),
        (
            {
                "distance_km": None,
                "restaurant": {"lat": 28.6, "lon": 77.2},
                "customer": {"lat": 19.1, "lon": 72.9},
            },
            "",
            "the limit is 50 km",
        ),
        (
            {
                "distance_km": None,
                "restaurant": {"lat": 51.5, "lon": -0.1},
                "customer": {"lat": 51.6, "lon": -0.1},
            },
            "restaurant.lat",
            "less than or equal to 38",
        ),
    ],
)
def test_bad_orders_are_explained(
    client: TestClient, change: dict[str, Any], where: str, words: str
) -> None:
    response = predict(client, a_request(**change))
    assert response.status_code == 422
    problems = response.json()["problems"]
    assert any(p["where"] == where and words in p["message"] for p in problems), problems


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_not_a_number_is_refused(client: TestClient, value: str) -> None:
    response = client.post(
        "/api/v1/predict",
        content=f'{{"distance_km": {value}}}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_broken_json_is_refused_without_echoing_it(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        content='{"secret-text": ',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert "secret-text" not in response.text


def test_huge_bodies_are_refused(make_client: MakeClient) -> None:
    client = make_client(max_body_bytes=2000)
    assert predict(client, {"distance_km": 5, "x": "y" * 3000}).status_code == 413

    def chunks() -> Iterator[bytes]:  # no Content-Length, so only counting can catch it
        for _ in range(30):
            yield b"y" * 100

    response = client.post(
        "/api/v1/predict", content=chunks(), headers={"content-type": "application/json"}
    )
    assert response.status_code == 413


# --- Access -----------------------------------------------------------------------------


def test_keys_protect_predictions_and_metrics(make_client: MakeClient) -> None:
    client = make_client(api_keys=f"{KEY}, another-long-test-key")
    assert predict(client, a_request()).status_code == 401
    assert predict(client, a_request(), headers={"X-API-Key": "wrong"}).status_code == 401
    assert predict(client, a_request(), headers={"X-API-Key": KEY}).status_code == 200
    assert client.post("/api/v1/predict/batch", json={"orders": [a_request()]}).status_code == 401
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": KEY}).status_code == 200
    # Health checks come from the host, which has no key.
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/api/v1/model").json()["key_required"] is True


def test_short_keys_are_not_accepted() -> None:
    with pytest.raises(ValidationError, match="at least 16"):
        Settings(api_keys="short")  # type: ignore[arg-type]


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ETA_API_KEYS", f"{KEY}, ")
    monkeypatch.setenv("ETA_CORS_ORIGINS", "https://a.example,https://b.example")
    monkeypatch.setenv("ETA_DOCS", "false")
    settings = Settings()
    assert settings.api_keys == {KEY}
    assert settings.cors_origins == {"https://a.example", "https://b.example"}
    assert settings.docs is False


def test_other_websites_are_kept_out_unless_listed(make_client: MakeClient) -> None:
    origin = {"Origin": "https://shop.example"}
    assert "access-control-allow-origin" not in make_client().get("/health", headers=origin).headers
    listed = make_client(cors_origins="https://shop.example")
    allowed = listed.get("/health", headers=origin).headers["access-control-allow-origin"]
    assert allowed == "https://shop.example"
    other = listed.get("/health", headers={"Origin": "https://evil.example"}).headers
    assert "access-control-allow-origin" not in other


def test_docs_and_metrics_can_be_switched_off(make_client: MakeClient) -> None:
    client = make_client(docs=False, metrics=False)
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/metrics").status_code == 404


# --- Operations -------------------------------------------------------------------------


def test_every_reply_has_an_id_and_safe_headers(client: TestClient) -> None:
    for path in ("/health", "/", "/nothing-here"):
        headers = client.get(path).headers
        assert len(headers["x-request-id"]) == 32
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
    assert (
        client.get("/health", headers={"X-Request-ID": "abc-1"}).headers["x-request-id"] == "abc-1"
    )
    unsafe = client.get("/health", headers={"X-Request-ID": "a b\tc"}).headers["x-request-id"]
    assert unsafe != "a b\tc"


def test_metrics_count_what_happened(client: TestClient) -> None:
    predict(client, {"distance_km": 45})
    predict(client, {"distance_km": -1})
    text = client.get("/metrics").text
    assert 'eta_requests_total{method="POST",route="/api/v1/predict",status="200"} 1.0' in text
    assert 'eta_requests_total{method="POST",route="/api/v1/predict",status="422"} 1.0' in text
    assert "eta_predicted_minutes_count 1.0" in text
    assert 'eta_missing_inputs_total{input="weather"} 1.0' in text
    assert "eta_unusual_orders_total 1.0" in text


def test_paths_that_do_not_exist_share_one_metric(client: TestClient) -> None:
    for i in range(3):
        client.get(f"/nothing/{i}")
    text = client.get("/metrics").text
    assert 'eta_requests_total{method="GET",route="unmatched",status="404"} 3.0' in text


def test_the_server_will_not_start_without_a_model(tmp_path: Path) -> None:
    app = create_app(Settings(model_dir=tmp_path / "none"))
    with pytest.raises(ArtifactError, match="No trained model"), TestClient(app):
        pass


def test_unexpected_errors_hide_their_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("internal detail")

    monkeypatch.setattr(client.app.state.predictor, "predict", boom)  # type: ignore[attr-defined]
    response = predict(client, a_request())
    assert response.status_code == 500
    assert "internal detail" not in response.text
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert 'status="500"' in client.get("/metrics").text


def test_logs_are_json_lines() -> None:
    record = logging.LogRecord("x", logging.INFO, "f", 1, "hello %s", ("you",), None)
    record.fields = {"ms": 1.5}
    line = json.loads(JsonLines().format(record))
    assert line["message"] == "hello you"
    assert line["severity"] == "INFO"
    assert line["ms"] == 1.5


# --- The page ---------------------------------------------------------------------------


def test_the_page_and_its_files_are_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "default-src 'self'" in page.headers["content-security-policy"]
    for name in ("app.js", "style.css", "favicon.svg"):
        assert f"/static/{name}" in page.text
        assert client.get(f"/static/{name}").status_code == 200
    # The policy blocks inline code, so the page must not rely on any.
    assert "<style" not in page.text
    assert "onclick" not in page.text


def test_the_page_offers_exactly_what_the_api_accepts(client: TestClient) -> None:
    page = client.get("/").text
    schema = client.get("/openapi.json").json()["components"]["schemas"]["OrderIn"]["properties"]
    for name in ("vehicle", "weather", "traffic", "city", "order_type"):
        (choices,) = (option["enum"] for option in schema[name]["anyOf"] if "enum" in option)
        for choice in choices:
            assert f'value="{choice}"' in page, (name, choice)
