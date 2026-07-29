import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select, text

from app.api import router
from app.config import get_settings
from app.database import Base, get_engine, session_factory
from app.embeddings import EmbeddingError
from app.llm import ModelError
from app.middleware import SecurityAndMetricsMiddleware
from app.models import KnowledgeSource

settings = get_settings()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            separators=(",", ":"),
        )


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=settings.log_level.upper(), handlers=[handler], force=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if get_engine().dialect.name == "sqlite":
        Base.metadata.create_all(get_engine())
    yield


app = FastAPI(
    title="Anlu Health API",
    version="0.1.0",
    docs_url=None if settings.app_env == "production" else "/docs",
    redoc_url=None,
    lifespan=lifespan,
)
app.add_middleware(SecurityAndMetricsMiddleware, settings=settings)
app.include_router(router)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.exception_handler(ModelError)
@app.exception_handler(EmbeddingError)
async def dependency_unavailable(_, exc: Exception) -> JSONResponse:  # type: ignore[no-untyped-def]
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health/live", include_in_schema=False)
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
def ready() -> dict[str, object]:
    try:
        with session_factory()() as db:
            db.execute(text("SELECT 1"))
            approved_sources = db.scalar(
                select(func.count(KnowledgeSource.id)).where(KnowledgeSource.approved.is_(True))
            )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is not ready") from exc
    if settings.app_env == "production" and not approved_sources:
        raise HTTPException(status_code=503, detail="No approved knowledge sources are indexed")
    local_experimental = (
        settings.app_env != "production"
        and settings.model_provider == "mock"
        and settings.embedding_provider == "mock"
    )
    return {
        "status": "ready",
        "approved_sources": approved_sources or 0,
        "runtime_mode": "local_experimental" if local_experimental else "hosted",
        "model_runtime": (
            "deterministic_mock"
            if settings.model_provider == "mock"
            else "private_openai_compatible_endpoint"
        ),
        "embedding_runtime": (
            "deterministic_mock"
            if settings.embedding_provider == "mock"
            else settings.embedding_provider
        ),
        "history_storage": "enabled" if settings.save_chat_history else "disabled",
    }


@app.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    if settings.metrics_token:
        provided = request.headers.get("Authorization", "")
        expected = f"Bearer {settings.metrics_token}"
        if not secrets.compare_digest(provided, expected):
            raise HTTPException(status_code=404, detail="Not found")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
