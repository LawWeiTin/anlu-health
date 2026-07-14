import threading
import time
from collections import defaultdict, deque
from functools import lru_cache

import redis

from app.config import get_settings


class RateLimitExceeded(Exception):
    pass


class RateLimiter:
    def __init__(self, redis_url: str | None = None) -> None:
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True) if redis_url else None
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> None:
        if self._redis is not None:
            bucket = int(time.time() // window_seconds)
            redis_key = f"anlu:rate:{key}:{bucket}"
            try:
                with self._redis.pipeline() as pipe:
                    pipe.incr(redis_key)
                    pipe.expire(redis_key, window_seconds + 5)
                    count, _ = pipe.execute()
                if int(count) > limit:
                    raise RateLimitExceeded
                return
            except redis.RedisError:
                # Availability of the health app should not depend on Redis. Fall back locally.
                pass

        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= limit:
                raise RateLimitExceeded
            events.append(now)


@lru_cache
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_settings().redis_url)
