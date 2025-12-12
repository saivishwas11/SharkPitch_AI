import time
import random
import logging
from functools import wraps
from typing import Callable, TypeVar, Optional
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RateLimitConfig:
    rate_limit_per_minute: int = 30
    max_retries: int = 5
    initial_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.1


class RateLimiter:
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.last_called = 0
        self.lock = Lock()

    def _calculate_delay(self):
        with self.lock:
            now = time.time()
            min_interval = 60 / self.config.rate_limit_per_minute
            delta = now - self.last_called
            if delta < min_interval:
                delay = min_interval - delta
                delay += delay * self.config.jitter * random.uniform(-1, 1)
                return max(delay, 0)
            return 0

    def _update_last_called(self):
        with self.lock:
            self.last_called = time.time()

    def __call__(self, func: Callable[..., T]):
        @wraps(func)
        def wrapper(*a, **kw):
            delay = self._calculate_delay()
            if delay > 0:
                time.sleep(delay)

            last_exc = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    result = func(*a, **kw)
                    self._update_last_called()
                    return result
                except Exception as e:
                    last_exc = e
                    backoff = min(self.config.initial_delay * (2 ** attempt), self.config.max_delay)
                    backoff += backoff * self.config.jitter * random.random()
                    time.sleep(backoff)

            raise RuntimeError(f"Failed after retries: {last_exc}") from last_exc

        return wrapper


groq_rate_limiter = RateLimiter()
