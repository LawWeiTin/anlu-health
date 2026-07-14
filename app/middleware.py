import logging
import time
import uuid

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import Settings

logger = logging.getLogger("healthbridge.http")
REQUESTS = Counter(
    "healthbridge_http_requests_total",
    "HTTP requests",
    ("method", "path", "status"),
)
LATENCY = Histogram(
    "healthbridge_http_request_duration_seconds",
    "HTTP request latency",
    ("method", "path"),
)


class SecurityAndMetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:80]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        path = request.url.path
        REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        LATENCY.labels(request.method, path).observe(elapsed)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'; "
            "img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'"
        )
        response.headers["Cache-Control"] = "no-store" if path.startswith("/api/") else "no-cache"
        if self.settings.cookie_secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        logger.info(
            "request id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            path,
            response.status_code,
            elapsed * 1000,
        )
        return response
