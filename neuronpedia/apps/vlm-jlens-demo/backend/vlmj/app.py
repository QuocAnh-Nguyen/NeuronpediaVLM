# SPDX-License-Identifier: Apache-2.0
"""FastAPI application for the J-Lens VLM demo backend.

Endpoints mirror the frozen JSON contract (snake_case, no aliasing): sessions are
encoded synchronously, every model-touching action returns a ``job_id`` that is
polled through ``GET /api/jobs/{job_id}``. The model and lens are built lazily on
the first engine-touching request, so the process always boots; when the lens is
unavailable ``/api/meta`` says so and lens-dependent endpoints answer 409.

Run it with ``python -m vlmj.app`` (see ``README.md`` for the environment
variables and the three ways to get a lens).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from vlmj.engine import (
    TINY_DEMO_CONFIG,
    TINY_MODEL_NAME,
    ApiError,
    Config,
    Engine,
    capabilities_payload,
    dtype_name,
    gpu_info,
    grid_for,
    job_attribution,
    job_generate,
    job_knockout,
    job_lens,
    job_steer,
    resolve_device,
    resolve_dtype,
    sample_images,
)
from vlmj.jobs import JobManager
from vlmj.schemas import (
    AttributionRequest,
    GenerateRequest,
    KnockoutRequest,
    LensRequest,
    SessionRequest,
    SteerRequest,
)

logger = logging.getLogger("vlmj")

#: Where a built frontend is mounted when ``../frontend/dist`` exists.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class Runtime:
    """Lazily-built engine, job queue and cached sample images."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.jobs = JobManager(wrapper=self._execute_job)
        self._engine: Engine | None = None
        self._error: str | None = None
        self._lock = threading.Lock()
        self._samples: list[dict[str, str]] | None = None

    # ------------------------------------------------------------------ engine
    def engine(self) -> Engine | None:
        """Engine, building it on first use; ``None`` when the model cannot load."""
        with self._lock:
            if self._engine is None and self._error is None:
                try:
                    self._engine = Engine(self.config)
                except Exception as exc:  # noqa: BLE001 - reported through /api/meta
                    self._error = f"{type(exc).__name__}: {exc}"
                    logger.exception("failed to initialise the model")
            return self._engine

    def require_engine(self) -> Engine:
        engine = self.engine()
        if engine is None:
            raise ApiError(503, f"model backend failed to load: {self._error}")
        return engine

    def _execute_job(self, ctx: Any, fn: Any) -> Any:
        """Job wrapper: the engine's global lock plus inference mode."""
        return self.require_engine().execute(ctx, fn)

    def samples(self) -> list[dict[str, str]]:
        with self._lock:
            if self._samples is None:
                self._samples = sample_images(int(TINY_DEMO_CONFIG["image_size"]))
            return self._samples

    # ------------------------------------------------------------------ meta
    def meta(self) -> dict[str, Any]:
        engine = self.engine()
        if engine is None:
            return self._degraded_meta()
        return {
            "mode": engine.config.backend,
            "device": str(engine.device),
            "dtype": dtype_name(engine.dtype),
            "model": engine.model_info(),
            "lens": engine.lens_info(),
            "capabilities": engine.capabilities(),
            "jobs": self.jobs.counts(),
            "gpu": gpu_info(),
        }

    def _degraded_meta(self) -> dict[str, Any]:
        """``/api/meta`` when the model itself failed to load (still JSON-shaped)."""
        backend = self.config.backend
        device = str(resolve_device(backend, self.config.device))
        try:
            dtype = dtype_name(resolve_dtype(backend, self.config.dtype))
        except ValueError:
            dtype = "unknown"
        if backend == "tiny":
            image_seq_length = int(TINY_DEMO_CONFIG["image_size"] // TINY_DEMO_CONFIG["patch_size"]) ** 2
            model = {
                "name": TINY_MODEL_NAME,
                "n_layers": int(TINY_DEMO_CONFIG["n_layers"]),
                "d_model": int(TINY_DEMO_CONFIG["d_model"]),
                "image_seq_length": image_seq_length,
                "grid": grid_for(image_seq_length),
                "vocab_size": int(TINY_DEMO_CONFIG["vocab_size"]),
            }
        else:
            model = {
                "name": self.config.model_id,
                "n_layers": 0,
                "d_model": 0,
                "image_seq_length": 0,
                "grid": [0, 0],
                "vocab_size": 0,
            }
        lens_dir = str(self.config.lens_dir) if self.config.lens_dir is not None else None
        error = self._error or "model backend failed to load"
        return {
            "mode": backend,
            "device": device,
            "dtype": dtype,
            "model": model,
            "lens": {
                "available": False,
                "dir": lens_dir,
                "source_layers": [],
                "masks": [],
                "n_prompts": {},
                "mock": False,
                "notes": None,
                "error": error,
            },
            "capabilities": capabilities_payload(),
            "jobs": self.jobs.counts(),
            "gpu": gpu_info(),
        }


# --------------------------------------------------------------------------- app


def _format_validation_error(exc: RequestValidationError) -> str:
    """Single-string ``detail`` for 422s (the frontend renders one message)."""
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        parts.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "validation failed: " + "; ".join(parts) if parts else "validation failed"


def create_app(config: Config | None = None) -> FastAPI:
    """Build the FastAPI app; ``config`` defaults to the ``VLMJ_*`` environment."""
    resolved = config if config is not None else Config.from_env()
    runtime = Runtime(resolved)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        runtime.jobs.shutdown()

    app = FastAPI(title="vlm-jlens-demo backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _format_validation_error(exc)})

    # ------------------------------------------------------------------ meta
    @app.get("/api/meta")
    def read_meta() -> dict[str, Any]:
        return runtime.meta()

    @app.get("/api/samples")
    def read_samples() -> dict[str, Any]:
        return {"samples": runtime.samples()}

    # ------------------------------------------------------------------ sessions
    @app.post("/api/session")
    def create_session(body: SessionRequest) -> dict[str, Any]:
        engine = runtime.require_engine()
        session = engine.create_session(
            image_b64=body.image_b64, image_path=body.image_path, prompt=body.prompt, variant=body.variant
        )
        return engine.session_payload(session)

    @app.get("/api/session/{session_id}")
    def read_session(session_id: str) -> dict[str, Any]:
        engine = runtime.require_engine()
        return engine.session_info(engine.get_session(session_id))

    # ------------------------------------------------------------------ jobs
    @app.post("/api/generate")
    def post_generate(body: GenerateRequest) -> dict[str, str]:
        engine = runtime.require_engine()
        session = engine.get_session(body.session_id)
        job = partial(job_generate, session=session, max_new_tokens=body.max_new_tokens)
        return {"job_id": runtime.jobs.submit("generate", job)}

    @app.post("/api/lens")
    def post_lens(body: LensRequest) -> dict[str, str]:
        engine = runtime.require_engine()
        engine.require_lens()
        session = engine.get_session(body.session_id)
        engine.resolve_layers(body.layers)
        baseline = session.baseline
        for target in body.targets:
            if target.kind == "prompt":
                engine.check_position(target.i, session.batch.seq_len, "prompt position")
            elif target.kind == "patch":
                engine.check_position(target.i, session.image_count, "patch index")
            elif target.kind == "gen":
                if baseline is None:
                    raise ApiError(422, "no generated caption yet; run POST /api/generate first")
                engine.check_position(target.i, len(baseline["token_ids"]), "generated-token index")
        for token in body.track or []:
            engine.resolve_token(token)
        job = partial(job_lens, session=session, request=body)
        return {"job_id": runtime.jobs.submit("lens", job)}

    @app.post("/api/attribution")
    def post_attribution(body: AttributionRequest) -> dict[str, str]:
        engine = runtime.require_engine()
        engine.require_lens()
        session = engine.get_session(body.session_id)
        engine.require_image_session(session)
        engine.validate_readout_layer(body.layer)
        if body.metric in ("lens_prob", "lens_logit"):
            engine.resolve_token(body.token)
        job = partial(job_attribution, session=session, request=body)
        return {"job_id": runtime.jobs.submit("attribution", job)}

    @app.post("/api/knockout")
    def post_knockout(body: KnockoutRequest) -> dict[str, str]:
        engine = runtime.require_engine()
        engine.require_lens()
        session = engine.get_session(body.session_id)
        engine.require_image_session(session)
        for patch in body.patches:
            engine.check_position(patch, session.image_count, "patch index")
        job = partial(job_knockout, session=session, request=body)
        return {"job_id": runtime.jobs.submit("knockout", job)}

    @app.post("/api/steer")
    def post_steer(body: SteerRequest) -> dict[str, str]:
        engine = runtime.require_engine()
        engine.require_lens()
        session = engine.get_session(body.session_id)
        engine.validate_steer_layer(body.layer)
        if body.mode in ("add", "ablate") and not body.token:
            raise ApiError(422, f"steering mode {body.mode!r} needs a target token")
        if body.mode == "swap" and (not body.token or not body.source_token):
            raise ApiError(422, "steering mode 'swap' needs both token and source_token")
        if body.token:
            engine.resolve_token(body.token)
        if body.source_token:
            engine.resolve_token(body.source_token)
        job = partial(job_steer, session=session, request=body)
        return {"job_id": runtime.jobs.submit("steer", job)}

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str) -> dict[str, Any]:
        snapshot = runtime.jobs.snapshot(job_id)
        if snapshot is None:
            raise ApiError(404, f"unknown job_id: {job_id}")
        return snapshot

    # ------------------------------------------------------------------ frontend
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
        logger.info("serving the built frontend from %s", FRONTEND_DIST)
    else:

        @app.get("/")
        def read_root() -> dict[str, Any]:
            return {
                "service": "vlm-jlens-demo backend",
                "frontend": f"not built ({FRONTEND_DIST})",
                "docs": "/docs",
            }

    return app


app = create_app()


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m vlmj.app`` entry point."""
    parser = argparse.ArgumentParser(description="Run the vlm-jlens-demo backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="reload on source changes")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import uvicorn

    if args.reload:
        uvicorn.run(
            "vlmj.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
