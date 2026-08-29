#!/usr/bin/env python3
"""
Phase 4 - the plumbing every ingestion source sits on.

WHY THIS TREE IS OUTSIDE stark-core/

CLAUDE.md: "STARK holds no capital. It places no orders, holds no keys, and
makes no network calls. Execution lives outside this tree." Ingestion touches
the network and holds credentials, so it lives here instead, and it hands STARK
what STARK accepts: plain arrays with provenance attached. The boundary is the
point - a bug in a crawler must not be able to reach a risk limit.

WHAT THIS FILE PROVIDES

  Provenance    every record says where it came from, when, and the hash of the
                bytes it was parsed from. A number whose origin you cannot
                reconstruct is not data, it is a rumour.
  RateLimiter   token bucket on an injectable clock, so politeness is testable
  Cache         content-addressed on disk; a re-run costs nothing and a source
                that goes down does not erase yesterday's work
  HttpClient    timeouts, bounded retries with exponential backoff, Retry-After
                honoured, a real User-Agent, and an offline mode that refuses to
                touch the network at all

BEING A GOOD CITIZEN IS NOT OPTIONAL. Every source declares a rate limit and a
contact string. Public APIs (arXiv, ECB, FRED) are used through their documented
interfaces. A source whose terms forbid automated access does not get a client
written for it, and none is written here.

stdlib only - no new dependency for the privilege of making an HTTP request.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol

USER_AGENT = "STARK-MK18-research/1.0 (+contact: set CONTACT in your config)"
DEFAULT_TIMEOUT = 30.0


class FetchError(RuntimeError):
    """A fetch failed after exhausting its retries. Carries what was tried."""


@dataclass(frozen=True)
class Provenance:
    """Where a number came from. Mandatory, not decorative."""

    source: str
    url: str
    fetched_at: float
    content_sha256: str
    note: str = ""

    def __str__(self) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(self.fetched_at))
        return f"{self.source} <- {self.url} @ {stamp} [{self.content_sha256[:12]}]"


@dataclass(frozen=True)
class Record:
    """One normalised item from any source. Payload shape is the source's business."""

    source: str
    key: str
    timestamp: float
    payload: Mapping[str, object]
    provenance: Provenance

    def to_json(self) -> str:
        return json.dumps({
            "source": self.source, "key": self.key, "timestamp": self.timestamp,
            "payload": dict(self.payload),
            "provenance": {
                "url": self.provenance.url,
                "fetched_at": self.provenance.fetched_at,
                "sha256": self.provenance.content_sha256,
            },
        }, sort_keys=True)


class Source(Protocol):
    """Anything that can produce records. The only contract ingestion has."""

    name: str

    def fetch(self) -> list[Record]:
        ...


# ── politeness ────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token bucket. `min_interval` seconds between requests, burst allowed.

    The clock and the sleep function are injected, so a test can prove the
    limiter waits without actually waiting. A rate limiter you cannot test is a
    rate limiter you will quietly break.
    """

    def __init__(
        self,
        min_interval: float,
        burst: int = 1,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval cannot be negative")
        self.min_interval = float(min_interval)
        self.burst = max(int(burst), 1)
        self.clock = clock
        self.sleeper = sleeper
        self._allowance = float(self.burst)
        self._last = clock()

    def acquire(self) -> float:
        """Block until a request is allowed. Returns how long it waited."""
        now = self.clock()
        if self.min_interval > 0:
            self._allowance = min(float(self.burst),
                                  self._allowance + (now - self._last) / self.min_interval)
        else:
            self._allowance = float(self.burst)
        self._last = now
        if self._allowance >= 1.0:
            self._allowance -= 1.0
            return 0.0
        wait = (1.0 - self._allowance) * self.min_interval
        self.sleeper(wait)
        self._last = self.clock()
        self._allowance = 0.0
        return wait


# ── cache ─────────────────────────────────────────────────────────────────────

class Cache:
    """
    Content-addressed response cache keyed by URL.

    Two jobs. It keeps you from hammering someone else's server during
    development, and it keeps a record of the exact bytes a number was derived
    from - so when a series is revised under you, the old bytes are still there
    to prove what you saw.
    """

    def __init__(self, directory: str | Path, ttl_seconds: float = 86_400.0) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl = float(ttl_seconds)

    def _path(self, url: str) -> Path:
        return self.directory / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".bin")

    def get(self, url: str) -> bytes | None:
        path = self._path(url)
        if not path.exists():
            return None
        if self.ttl > 0 and (time.time() - path.stat().st_mtime) > self.ttl:
            return None
        return path.read_bytes()

    def put(self, url: str, content: bytes) -> None:
        self._path(url).write_bytes(content)

    def age(self, url: str) -> float | None:
        path = self._path(url)
        return None if not path.exists() else time.time() - path.stat().st_mtime


# ── transport ─────────────────────────────────────────────────────────────────

Transport = Callable[[str, Mapping[str, str], float], tuple[int, bytes, Mapping[str, str]]]


def urllib_transport(url: str, headers: Mapping[str, str], timeout: float):
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as e:                      # 4xx/5xx still carry a body
        return e.code, e.read() if e.fp else b"", dict(e.headers or {})


class HttpClient:
    """
    A polite, cached, retrying HTTP GET. That is the whole feature list.

    `offline=True` refuses to touch the network and serves cache only - which is
    what tests use, and what you want when re-deriving a result months later
    without silently picking up revised data.
    """

    RETRYABLE = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        cache: Cache | None = None,
        transport: Transport = urllib_transport,
        user_agent: str = USER_AGENT,
        max_retries: int = 4,
        timeout: float = DEFAULT_TIMEOUT,
        offline: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate_limiter = rate_limiter or RateLimiter(min_interval=1.0)
        self.cache = cache
        self.transport = transport
        self.user_agent = user_agent
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.offline = bool(offline)
        self.sleeper = sleeper
        self.request_count = 0

    def get(self, url: str, headers: Mapping[str, str] | None = None,
            use_cache: bool = True) -> tuple[bytes, Provenance]:
        if use_cache and self.cache is not None:
            cached = self.cache.get(url)
            if cached is not None:
                return cached, Provenance("cache", url, time.time(),
                                          hashlib.sha256(cached).hexdigest(), "served from cache")
        if self.offline:
            raise FetchError(f"offline mode and nothing cached for {url}")

        merged = {"User-Agent": self.user_agent, "Accept-Encoding": "identity"}
        merged.update(headers or {})

        last_problem = "no attempt made"
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.acquire()
            self.request_count += 1
            try:
                status, body, response_headers = self.transport(url, merged, self.timeout)
            except Exception as e:                            # transport-level failure
                last_problem = f"{type(e).__name__}: {e}"
                status, body, response_headers = 0, b"", {}

            if 200 <= status < 300:
                if self.cache is not None:
                    self.cache.put(url, body)
                return body, Provenance("http", url, time.time(), hashlib.sha256(body).hexdigest())

            if status and status not in self.RETRYABLE:
                raise FetchError(f"{url} returned {status} (not retryable)")
            if status:
                last_problem = f"HTTP {status}"

            if attempt < self.max_retries:
                # Retry-After is the server telling you exactly how long to wait.
                # Ignoring it is how a polite client becomes an abusive one.
                explicit = _retry_after_seconds(response_headers)
                self.sleeper(explicit if explicit is not None else 2.0 ** attempt)

        raise FetchError(f"{url} failed after {self.max_retries + 1} attempts ({last_problem})")


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                return None
    return None


def write_jsonl(records: Iterable[Record], path: str | Path) -> int:
    """Append records to a JSONL file. The handoff format into STARK."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
            written += 1
    return written


if __name__ == "__main__":
    print("STARK MK18 - ingestion plumbing\n")

    class FakeClock:
        """A clock that only moves when something sleeps. Exactly like a busy loop."""

        def __init__(self) -> None:
            self.now = 0.0
            self.slept = 0.0

        def __call__(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds
            self.slept += seconds

    clock = FakeClock()
    limiter = RateLimiter(min_interval=1.0, burst=2, clock=clock, sleeper=clock.sleep)
    print("rate limiter (1 req/sec, burst 2), five requests back to back:")
    for i in range(5):
        print(f"  request {i} at t={clock.now:.2f}s: waited {limiter.acquire():.2f}s")
    print(f"  total sleep {clock.slept:.2f}s - and the test itself never slept\n")

    calls: list[str] = []

    def flaky(url, headers, timeout):
        calls.append(url)
        if len(calls) < 3:
            return 503, b"", {"Retry-After": "0"}
        return 200, b'{"ok": true}', {}

    client = HttpClient(transport=flaky, rate_limiter=RateLimiter(0.0), sleeper=lambda s: None)
    body, provenance = client.get("https://example.invalid/data")
    print(f"retry/backoff: recovered after {len(calls)} attempts")
    print(f"  provenance: {provenance}")
