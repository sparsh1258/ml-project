"""The web service: predictions, health checks, metrics, and a page to try it."""

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from delivery_eta import __version__
from delivery_eta.api.observability import Metrics, RequestGuard, log, request_id, setup_logging
from delivery_eta.api.schemas import BatchIn, BatchOut, EstimateOut, ModelOut, OrderIn
from delivery_eta.api.settings import Settings
from delivery_eta.artifact import ArtifactError
from delivery_eta.features import LOCAL_TIME
from delivery_eta.predictor import Predictor

STATIC = Path(__file__).parent / "static"


def get_predictor(request: Request) -> Predictor:
    predictor: Predictor | None = request.app.state.predictor
    if predictor is None:
        raise HTTPException(503, "The model is not loaded.")
    return predictor


Ready = Annotated[Predictor, Depends(get_predictor)]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    setup_logging(settings.log_level)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # A missing or damaged model stops the server from starting: better than serving nothing.
        try:
            app.state.predictor = Predictor.load(settings.model_dir)
        except ArtifactError as e:
            log.error("Can't start: %s", e)
            raise
        version = app.state.predictor.model.version
        metrics.model.labels(version).set(1)
        log.info("Ready", extra={"fields": {"model_version": version}})
        yield
        app.state.predictor = None

    app = FastAPI(
        title="Delivery ETA",
        version=__version__,
        summary="Predicts food delivery time in minutes, with a likely range.",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs else None,
    )
    app.state.predictor = None

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=sorted(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )
    # Added last, so it is outermost and sees every request.
    app.add_middleware(RequestGuard, metrics=metrics, max_body_bytes=settings.max_body_bytes)

    def check_key(x_api_key: Annotated[str, Header()] = "") -> None:
        if not settings.api_keys:
            return
        given = x_api_key.encode()
        # Compare with every key, so the time taken doesn't hint at which one is close.
        matches = [hmac.compare_digest(given, key.encode()) for key in settings.api_keys]
        if not any(matches):
            raise HTTPException(401, "A valid X-API-Key header is needed.")

    keyed = [Depends(check_key)]

    @app.exception_handler(RequestValidationError)
    async def explain_bad_input(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The default reply repeats the whole input back; ours says only what is wrong and where.
        problems = [
            {
                "where": ".".join(str(part) for part in error["loc"] if part != "body"),
                "message": error["msg"].removeprefix("Value error, "),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            jsonable_encoder(
                {
                    "detail": "Some inputs are not valid.",
                    "problems": problems,
                    "request_id": request_id.get(),
                }
            ),
            status_code=422,
        )

    def estimate(predictor: Predictor, orders: list[OrderIn], explain: bool) -> list[EstimateOut]:
        now = datetime.now(LOCAL_TIME)
        inputs = [order.to_order(now) for order in orders]
        estimates = predictor.predict(inputs, explain=explain)
        for e in estimates:
            metrics.minutes.observe(e.minutes)
            if e.warnings:
                metrics.unusual.inc()
            for name in e.missing:
                metrics.missing.labels(name).inc()
        version = predictor.model.version
        return [
            EstimateOut.build(e, order, version) for e, order in zip(estimates, inputs, strict=True)
        ]

    @app.post("/api/v1/predict", dependencies=keyed, tags=["predictions"])
    def predict(order: OrderIn, predictor: Ready, explain: bool = False) -> EstimateOut:
        """Delivery time for one order. `?explain=true` adds what moved the estimate."""
        return estimate(predictor, [order], explain)[0]

    @app.post("/api/v1/predict/batch", dependencies=keyed, tags=["predictions"])
    def predict_batch(batch: BatchIn, predictor: Ready) -> BatchOut:
        """Delivery times for many orders, in the same order."""
        if len(batch.orders) > settings.max_batch:
            raise HTTPException(
                413, f"{len(batch.orders)} orders sent; the limit is {settings.max_batch}."
            )
        return BatchOut(estimates=estimate(predictor, batch.orders, explain=False))

    @app.get("/api/v1/model", tags=["about"])
    def model(predictor: Ready) -> ModelOut:
        """Which model is running and how well it did on days it never saw."""
        info = predictor.model.info
        report = info["report"]
        test = report["test"]
        return ModelOut(
            version=info["version"],
            trained_on_orders=info["trained_on_orders"],
            first_day=report["data"]["first_day"],
            last_day=report["data"]["last_day"],
            test_orders=test["orders"],
            average_error_minutes=test["overall"]["mae"],
            guess_the_average_error_minutes=test["guess_the_average_mae"],
            within_5_minutes=test["overall"]["within_5_min"],
            range_coverage=test["overall"]["coverage"],
            seen=info["seen"],
            key_required=bool(settings.api_keys),
        )

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        """The process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"])
    def ready(predictor: Ready) -> dict[str, str]:
        """The model is loaded and giving answers."""
        return {"status": "ready", "model_version": predictor.model.version}

    if settings.metrics:

        @app.get("/metrics", dependencies=keyed, tags=["operations"], include_in_schema=False)
        def prometheus() -> Response:
            return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
