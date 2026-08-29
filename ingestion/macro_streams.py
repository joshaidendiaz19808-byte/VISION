#!/usr/bin/env python3
"""
Phase 4 - alternative and macro data streams.

THE ONLY THING THAT REALLY MATTERS IN THIS FILE

Macro data is revised. US GDP is published, then revised twice in the following
two months, then re-benchmarked years later. Payrolls are revised almost every
month. The series you download today is NOT the series anyone could see back
then, and a backtest run on it is reading tomorrow's newspaper.

This is the same failure VISION exists to catch, wearing a suit. It does not
look like cheating - you did not peek, you just downloaded a CSV - and it will
make a mediocre strategy look excellent, because the revised series is closer to
the truth than the number traders actually reacted to.

So the core type here is not a time series. It is a POINT-IN-TIME series: every
value carries the date it referred to AND the date it became knowable, and
`as_of(t)` returns only what a person could have known at t. Where a source does
not publish release dates, `PointInTimeSeries` says so and refuses to pretend -
`is_point_in_time` is False and every downstream consumer can see it.

Sources: ECB Statistical Data Warehouse (open, no key) and FRED (needs a free
API key, read from the environment - no credential is ever written into this
tree). A generic CSV loader covers everything else. Sources whose terms forbid
automated access do not get a client here.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from base import Cache, FetchError, HttpClient, Provenance, RateLimiter, Record

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
ECB_API = "https://data-api.ecb.europa.eu/service/data"

DAY = 86_400.0

# Series that are heavily revised. Using the latest vintage of any of these in a
# backtest is look-ahead, full stop. The list is short and deliberately blunt.
HEAVILY_REVISED = frozenset({
    "GDP", "GDPC1", "PAYEMS", "INDPRO", "PCE", "PCEPI", "RSAFS", "HOUST", "BUSINV",
})


@dataclass(frozen=True)
class Observation:
    """One number, and the two dates that make it usable."""

    period: float        # what the number is ABOUT
    value: float
    released: float | None = None   # when it became knowable; None = unknown

    @property
    def lag_days(self) -> float | None:
        return None if self.released is None else (self.released - self.period) / DAY


@dataclass
class PointInTimeSeries:
    """
    A macro series that knows when each of its numbers became public.

    `as_of(t)` is the whole point: it returns the series as it stood at time t,
    which is the only version any decision could have been made from.
    """

    name: str
    observations: list[Observation]
    units: str = ""
    frequency: str = ""
    provenance: Provenance | None = None
    release_lag_days: float | None = None    # assumed lag when releases are unknown

    @property
    def is_point_in_time(self) -> bool:
        """True only if EVERY observation knows when it was released."""
        return bool(self.observations) and all(o.released is not None for o in self.observations)

    @property
    def revision_risk(self) -> bool:
        return self.name.upper() in HEAVILY_REVISED

    def as_of(self, when: float) -> list[Observation]:
        """
        What was knowable at `when`.

        If release dates are missing, an assumed lag is applied and the result is
        an ESTIMATE of point-in-time - better than nothing, worse than the real
        thing, and `is_point_in_time` stays False so nobody forgets which they
        are holding.
        """
        out = []
        for observation in self.observations:
            released = observation.released
            if released is None:
                if self.release_lag_days is None:
                    continue          # unknowable: refuse to guess
                released = observation.period + self.release_lag_days * DAY
            if released <= when:
                out.append(observation)
        return sorted(out, key=lambda o: o.period)

    def latest_as_of(self, when: float) -> Observation | None:
        window = self.as_of(when)
        return window[-1] if window else None

    def to_arrays(self, when: float | None = None) -> tuple[list[float], list[float]]:
        """Handoff into STARK: plain lists, point-in-time if `when` is given."""
        window = self.observations if when is None else self.as_of(when)
        window = sorted(window, key=lambda o: o.period)
        return [o.period for o in window], [o.value for o in window]

    def warnings(self) -> list[str]:
        """Everything a consumer of this series needs to be told, in plain English."""
        notes: list[str] = []
        if not self.is_point_in_time:
            if self.release_lag_days is None:
                notes.append(
                    f"'{self.name}' carries NO release dates. Any backtest using it is "
                    "reading numbers before they existed. Supply release_lag_days, or "
                    "get a vintage series."
                )
            else:
                notes.append(
                    f"'{self.name}' has no real release dates; a flat "
                    f"{self.release_lag_days:.0f}-day lag is assumed. That is an estimate."
                )
        if self.revision_risk:
            notes.append(
                f"'{self.name}' is heavily revised after first publication. The values here "
                "are the CURRENT vintage, not what was printed at the time."
            )
        return notes

    def __str__(self) -> str:
        pit = "point-in-time" if self.is_point_in_time else "NOT point-in-time"
        span = ""
        if self.observations:
            first = time.strftime("%Y-%m-%d", time.gmtime(self.observations[0].period))
            last = time.strftime("%Y-%m-%d", time.gmtime(self.observations[-1].period))
            span = f" | {first} to {last}"
        return f"{self.name}: {len(self.observations)} obs{span} | {pit}"


# ── clients ───────────────────────────────────────────────────────────────────

class FredClient:
    """
    FRED series. Needs a free API key, read from FRED_API_KEY in the environment.

    The key is never written to disk by this tree and never appears in a log line.
    FRED's own observations endpoint can return real vintages via
    `realtime_start`/`realtime_end`, which is what makes point-in-time possible
    here at all - use it.
    """

    name = "fred"

    def __init__(self, client: HttpClient | None = None, api_key: str | None = None,
                 cache_dir: str | None = None, offline: bool = False) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.client = client or HttpClient(
            rate_limiter=RateLimiter(min_interval=0.6),
            cache=Cache(cache_dir) if cache_dir else None, offline=offline)

    def url(self, series_id: str, realtime_start: str | None = None) -> str:
        params = {"series_id": series_id, "api_key": self.api_key or "", "file_type": "json"}
        if realtime_start:                       # ask for the vintage, not the revision
            params["realtime_start"] = realtime_start
            params["realtime_end"] = realtime_start
        return FRED_API + "?" + urllib.parse.urlencode(params)

    def fetch(self, series_id: str, realtime_start: str | None = None) -> PointInTimeSeries:
        if not self.api_key:
            raise FetchError(
                "FRED needs an API key. Get a free one at fred.stlouisfed.org and export "
                "FRED_API_KEY. No key is stored in this repository."
            )
        body, provenance = self.client.get(self.url(series_id, realtime_start))
        return parse_fred(series_id, body, provenance)


def parse_fred(series_id: str, body: bytes, provenance: Provenance | None = None) -> PointInTimeSeries:
    """Parse FRED's observations JSON. Missing values are dropped, never zero-filled."""
    payload = json.loads(body.decode("utf-8"))
    observations = []
    for row in payload.get("observations", []):
        value = row.get("value")
        if value in (None, ".", ""):
            continue                              # FRED marks missing with "."
        try:
            observations.append(Observation(
                period=_parse_date(row["date"]),
                value=float(value),
                released=_parse_date(row["realtime_start"]) if row.get("realtime_start") else None,
            ))
        except (KeyError, ValueError):
            continue
    return PointInTimeSeries(series_id, sorted(observations, key=lambda o: o.period),
                             units=payload.get("units", ""), provenance=provenance)


class EcbClient:
    """ECB Statistical Data Warehouse. Open, documented, no key required."""

    name = "ecb"

    def __init__(self, client: HttpClient | None = None, cache_dir: str | None = None,
                 offline: bool = False) -> None:
        self.client = client or HttpClient(
            rate_limiter=RateLimiter(min_interval=1.0),
            cache=Cache(cache_dir) if cache_dir else None, offline=offline)

    def url(self, flow: str, key: str) -> str:
        return f"{ECB_API}/{flow}/{key}?format=csvdata"

    def fetch(self, flow: str, key: str, release_lag_days: float = 1.0) -> PointInTimeSeries:
        body, provenance = self.client.get(self.url(flow, key))
        series = parse_ecb_csv(f"{flow}/{key}", body, provenance)
        series.release_lag_days = release_lag_days
        return series


def parse_ecb_csv(name: str, body: bytes, provenance: Provenance | None = None) -> PointInTimeSeries:
    """
    Parse ECB CSV. It carries no release dates, so the result is NOT point-in-time
    and the object says so.
    """
    reader = csv.DictReader(io.StringIO(body.decode("utf-8", errors="replace")))
    observations = []
    for row in reader:
        period, value = row.get("TIME_PERIOD"), row.get("OBS_VALUE")
        if not period or value in (None, "", "NaN"):
            continue
        try:
            observations.append(Observation(period=_parse_date(period), value=float(value)))
        except ValueError:
            continue
    return PointInTimeSeries(name, sorted(observations, key=lambda o: o.period),
                             provenance=provenance)


def load_csv(name: str, text: str, period_column: str = "date",
             value_column: str = "value", release_column: str | None = None,
             release_lag_days: float | None = None) -> PointInTimeSeries:
    """Generic CSV loader. Pass `release_column` if your file knows when things dropped."""
    reader = csv.DictReader(io.StringIO(text))
    observations = []
    for row in reader:
        try:
            observations.append(Observation(
                period=_parse_date(row[period_column]),
                value=float(row[value_column]),
                released=_parse_date(row[release_column]) if release_column and row.get(release_column) else None,
            ))
        except (KeyError, ValueError):
            continue
    return PointInTimeSeries(name, sorted(observations, key=lambda o: o.period),
                             release_lag_days=release_lag_days)


# ── alignment ─────────────────────────────────────────────────────────────────

def align_as_of(series: Sequence[PointInTimeSeries], when: float) -> dict[str, float]:
    """
    The latest value of each series that was public at `when`.

    The correct way to build a feature row for a moment in the past: ask each
    series what it knew then, not what it knows now. A series with nothing
    released by `when` is absent from the result rather than forward-filled from
    the future.
    """
    out: dict[str, float] = {}
    for one in series:
        latest = one.latest_as_of(when)
        if latest is not None:
            out[one.name] = latest.value
    return out


def to_records(series: PointInTimeSeries) -> list[Record]:
    return [
        Record("macro", series.name, o.period,
               {"value": o.value, "released": o.released, "units": series.units,
                "point_in_time": series.is_point_in_time},
               series.provenance or Provenance("macro", "local", time.time(), ""))
        for o in series.observations
    ]


def _parse_date(text: str) -> float:
    text = text.strip()
    for pattern in ("%Y-%m-%d", "%Y-%m", "%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return time.mktime(time.strptime(text, pattern))
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {text!r}")


if __name__ == "__main__":
    print("STARK MK18 - macro streams: the point-in-time problem\n")

    # A series published with a 30-day lag, then revised upward a month later.
    csv_text = (
        "date,value,released\n"
        "2024-01-01,100.0,2024-02-01\n"
        "2024-02-01,102.0,2024-03-01\n"
        "2024-03-01,104.0,2024-04-01\n"
        "2024-04-01,106.0,2024-05-01\n"
    )
    honest = load_csv("INDPRO", csv_text, release_column="released")
    print(f"  {honest}")

    march = _parse_date("2024-03-15")
    print(f"\n  Standing on 2024-03-15, what could you actually see?")
    for observation in honest.as_of(march):
        period = time.strftime("%Y-%m", time.gmtime(observation.period))
        released = time.strftime("%Y-%m-%d", time.gmtime(observation.released))
        print(f"    {period}: {observation.value:6.1f}  (released {released})")
    print(f"    The March and April numbers do not appear. They did not exist yet.")

    naive = load_csv("INDPRO", "date,value\n" + "\n".join(
        line.rsplit(",", 1)[0] for line in csv_text.splitlines()[1:]))
    print(f"\n  The same file WITHOUT release dates:")
    print(f"    {naive}")
    for warning in naive.warnings():
        print(f"    ! {warning}")
    print(f"    as_of(2024-03-15) returns {len(naive.as_of(march))} observations - it refuses")
    print(f"    to guess rather than quietly handing you the future.")

    naive.release_lag_days = 30.0
    print(f"\n  With an assumed 30-day lag it returns {len(naive.as_of(march))}, and stays")
    print(f"  flagged NOT point-in-time, because an assumption is not a release date.")

    print("\n  A feature row for that moment:")
    print(f"    {align_as_of([honest], march)}")
    print("\nThis is look-ahead bias in its most respectable disguise: you did not peek,")
    print("you just downloaded a CSV. VISION cannot catch it from a trade log - only")
    print("point-in-time data can prevent it, which is why it lives at the ingestion layer.")
