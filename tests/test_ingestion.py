#!/usr/bin/env python3
"""
Phase 4 ingestion tests.

No test here touches the network. Transports are injected, clocks are fake, and
the parsers are fed bytes. A test suite that needs the internet is a test suite
that fails when someone else's server has a bad afternoon.

Run: pytest tests/test_ingestion.py -q
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from ingestion.base import (Cache, FetchError, HttpClient, Provenance, RateLimiter,
                            Record, write_jsonl)
from ingestion.macro_streams import (Observation, PointInTimeSeries, align_as_of,
                                     load_csv, parse_ecb_csv, parse_fred)
from ingestion.preprints import (NoveltyIndex, Preprint, parse_arxiv_atom, revisions,
                                 triage)
from ingestion.telemetry.ais import (AisDecodeError, Position, Track, bits_to_sixbit,
                                     build_tracks, decode_position, decode_stream,
                                     encode_position, find_gaps, find_rendezvous,
                                     haversine_nm, nmea_checksum, parse_aivdm,
                                     sixbit_to_bits)
from ingestion.telemetry.sar_tanks import (SarLook, Tank, aggregate_inventory,
                                           estimate_fill, resolvable_fill_step,
                                           shadow_pixels_for_fill)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds


# ── plumbing ──────────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_it_actually_throttles(self):
        clock = FakeClock()
        limiter = RateLimiter(1.0, burst=1, clock=clock, sleeper=clock.sleep)
        for _ in range(5):
            limiter.acquire()
        assert clock.slept == pytest.approx(4.0), "five requests at 1/s must take 4s of waiting"

    def test_burst_is_allowed_then_throttled(self):
        clock = FakeClock()
        limiter = RateLimiter(1.0, burst=3, clock=clock, sleeper=clock.sleep)
        assert all(limiter.acquire() == 0.0 for _ in range(3))
        assert limiter.acquire() > 0.0

    def test_zero_interval_never_waits(self):
        clock = FakeClock()
        limiter = RateLimiter(0.0, clock=clock, sleeper=clock.sleep)
        for _ in range(50):
            limiter.acquire()
        assert clock.slept == 0.0

    def test_negative_interval_is_rejected(self):
        with pytest.raises(ValueError):
            RateLimiter(-1.0)


class TestHttpClient:
    def _client(self, transport, **kw):
        return HttpClient(transport=transport, rate_limiter=RateLimiter(0.0),
                          sleeper=lambda s: None, **kw)

    def test_retries_then_succeeds(self):
        attempts = []

        def flaky(url, headers, timeout):
            attempts.append(url)
            return (503, b"", {}) if len(attempts) < 3 else (200, b"ok", {})

        body, provenance = self._client(flaky).get("https://example.invalid/x")
        assert body == b"ok" and len(attempts) == 3
        assert provenance.content_sha256

    def test_gives_up_and_says_so(self):
        with pytest.raises(FetchError, match="after"):
            self._client(lambda u, h, t: (503, b"", {})).get("https://example.invalid/x")

    def test_a_404_is_not_retried(self):
        attempts = []

        def missing(url, headers, timeout):
            attempts.append(url)
            return 404, b"", {}

        with pytest.raises(FetchError, match="not retryable"):
            self._client(missing).get("https://example.invalid/x")
        assert len(attempts) == 1, "retrying a 404 is just rudeness"

    def test_retry_after_is_honoured(self):
        waits = []
        attempts = []

        def throttled(url, headers, timeout):
            attempts.append(url)
            return (429, b"", {"Retry-After": "7"}) if len(attempts) < 2 else (200, b"ok", {})

        client = HttpClient(transport=throttled, rate_limiter=RateLimiter(0.0),
                            sleeper=waits.append)
        client.get("https://example.invalid/x")
        assert 7.0 in waits, "the server said 7 seconds and we must wait 7 seconds"

    def test_a_transport_exception_is_retried_not_raised(self):
        attempts = []

        def broken(url, headers, timeout):
            attempts.append(url)
            if len(attempts) < 2:
                raise OSError("connection reset")
            return 200, b"ok", {}

        assert self._client(broken).get("https://example.invalid/x")[0] == b"ok"

    def test_user_agent_is_always_sent(self):
        seen = {}

        def capture(url, headers, timeout):
            seen.update(headers)
            return 200, b"ok", {}

        self._client(capture).get("https://example.invalid/x")
        assert "User-Agent" in seen and seen["User-Agent"]

    def test_cache_serves_without_a_second_request(self, tmp_path):
        attempts = []

        def counted(url, headers, timeout):
            attempts.append(url)
            return 200, b"payload", {}

        client = self._client(counted, cache=Cache(tmp_path))
        client.get("https://example.invalid/x")
        client.get("https://example.invalid/x")
        assert len(attempts) == 1

    def test_offline_refuses_rather_than_dialling_out(self, tmp_path):
        def forbidden(url, headers, timeout):
            raise AssertionError("offline mode reached the network")

        client = HttpClient(transport=forbidden, cache=Cache(tmp_path), offline=True,
                            rate_limiter=RateLimiter(0.0))
        with pytest.raises(FetchError, match="offline"):
            client.get("https://example.invalid/x")

    def test_offline_still_serves_the_cache(self, tmp_path):
        cache = Cache(tmp_path)
        cache.put("https://example.invalid/x", b"yesterday")
        client = HttpClient(transport=lambda *a: (_ for _ in ()).throw(AssertionError()),
                            cache=cache, offline=True, rate_limiter=RateLimiter(0.0))
        assert client.get("https://example.invalid/x")[0] == b"yesterday"


def test_records_carry_provenance(tmp_path):
    record = Record("s", "k", 1.0, {"v": 2},
                    Provenance("s", "https://example.invalid", 1.0, "abc"))
    assert "provenance" in record.to_json()
    assert write_jsonl([record], tmp_path / "out.jsonl") == 1


# ── macro: the point-in-time problem ──────────────────────────────────────────

CSV_WITH_RELEASES = (
    "date,value,released\n"
    "2024-01-01,100.0,2024-02-01\n"
    "2024-02-01,102.0,2024-03-01\n"
    "2024-03-01,104.0,2024-04-01\n"
)


def march_15() -> float:
    return time.mktime(time.strptime("2024-03-15", "%Y-%m-%d"))


class TestPointInTime:
    def test_as_of_hides_the_future(self):
        series = load_csv("X", CSV_WITH_RELEASES, release_column="released")
        assert [o.value for o in series.as_of(march_15())] == [100.0, 102.0]

    def test_a_series_with_releases_is_point_in_time(self):
        assert load_csv("X", CSV_WITH_RELEASES, release_column="released").is_point_in_time

    def test_a_series_without_releases_is_not_and_says_so(self):
        series = load_csv("X", "date,value\n2024-01-01,100.0\n2024-02-01,102.0\n")
        assert not series.is_point_in_time
        assert any("NO release dates" in w for w in series.warnings())

    def test_without_releases_it_refuses_rather_than_guessing(self):
        """The failure mode this whole module exists to prevent."""
        series = load_csv("X", "date,value\n2024-01-01,100.0\n2024-03-01,104.0\n")
        assert series.as_of(march_15()) == [], "handed out data with no known release date"

    def test_an_assumed_lag_works_but_stays_flagged(self):
        series = load_csv("X", "date,value\n2024-01-01,100.0\n2024-02-01,102.0\n",
                          release_lag_days=30.0)
        assert len(series.as_of(march_15())) == 2
        assert not series.is_point_in_time, "an assumption is not a release date"
        assert any("assumed" in w for w in series.warnings())

    def test_heavily_revised_series_are_called_out(self):
        assert load_csv("GDP", CSV_WITH_RELEASES, release_column="released").revision_risk
        assert not load_csv("DGS10", CSV_WITH_RELEASES, release_column="released").revision_risk

    def test_align_builds_an_honest_feature_row(self):
        series = load_csv("X", CSV_WITH_RELEASES, release_column="released")
        assert align_as_of([series], march_15()) == {"X": 102.0}

    def test_a_series_with_nothing_released_is_absent_not_zero(self):
        early = time.mktime(time.strptime("2024-01-05", "%Y-%m-%d"))
        series = load_csv("X", CSV_WITH_RELEASES, release_column="released")
        assert align_as_of([series], early) == {}, "forward-filled a value from the future"

    def test_to_arrays_respects_as_of(self):
        series = load_csv("X", CSV_WITH_RELEASES, release_column="released")
        assert len(series.to_arrays(march_15())[1]) == 2
        assert len(series.to_arrays()[1]) == 3


class TestMacroParsers:
    def test_fred_json_including_missing_values(self):
        body = b"""{"units":"lin","observations":[
            {"date":"2024-01-01","value":"100.0","realtime_start":"2024-02-01"},
            {"date":"2024-02-01","value":".","realtime_start":"2024-03-01"},
            {"date":"2024-03-01","value":"104.0","realtime_start":"2024-04-01"}]}"""
        series = parse_fred("TEST", body)
        assert len(series.observations) == 2, "FRED's '.' must be dropped, never zero-filled"
        assert series.is_point_in_time

    def test_ecb_csv_is_parsed_and_flagged_not_point_in_time(self):
        body = b"KEY,TIME_PERIOD,OBS_VALUE\nA,2024-01,1.5\nA,2024-02,1.6\nA,2024-03,\n"
        series = parse_ecb_csv("ecb/test", body)
        assert len(series.observations) == 2
        assert not series.is_point_in_time

    def test_unparseable_rows_are_skipped_not_guessed(self):
        series = load_csv("X", "date,value\nnot-a-date,1.0\n2024-01-01,2.0\n2024-02-01,oops\n")
        assert [o.value for o in series.observations] == [2.0]


# ── preprints ─────────────────────────────────────────────────────────────────

ATOM_SAMPLE = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <title>A Study of Market Microstructure</title>
    <summary>We examine liquidity and spread formation in limit order books.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <updated>2024-02-01T00:00:00Z</updated>
    <author><name>A Researcher</name></author>
    <category term="q-fin.TR"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Topological Quantum Approaches to Portfolio Selection</title>
    <summary>We introduce anyonic braiding statistics for allocation problems.</summary>
    <published>2024-01-02T00:00:00Z</published>
    <updated>2024-01-02T00:00:00Z</updated>
    <author><name>B Researcher</name></author>
    <category term="q-fin.PM"/>
  </entry>
</feed>"""


class TestPreprints:
    def test_atom_parsing_extracts_the_fields(self):
        papers = parse_arxiv_atom(ATOM_SAMPLE)
        assert len(papers) == 2
        assert papers[0].arxiv_id == "2401.00001" and papers[0].version == 2
        assert papers[0].is_revision and not papers[1].is_revision
        assert papers[0].authors == ("A Researcher",)

    def test_malformed_entries_are_skipped(self):
        assert parse_arxiv_atom(b'<feed xmlns="http://www.w3.org/2005/Atom"><entry/></feed>') == []

    def test_novelty_ranks_the_unusual_paper_higher(self):
        papers = parse_arxiv_atom(ATOM_SAMPLE)
        corpus = [Preprint(f"c{i}", 1, "Market Microstructure and Liquidity",
                           "We examine liquidity and spread formation in limit order books.",
                           (), (), 0.0, 0.0, "") for i in range(30)]
        ranked = triage(papers, corpus=corpus, k=2)
        assert "Quantum" in ranked[0][0].title, "the odd one out did not rank first"

    def test_scores_actually_discriminate(self):
        """Regression: averaging only the rarest terms gave every paper the same score."""
        corpus = [Preprint(f"c{i}", 1, "Common market study",
                           "liquidity spread order book market", (), (), 0.0, 0.0, "")
                  for i in range(40)]
        index = NoveltyIndex().fit(corpus)
        ordinary = Preprint("x", 1, "Common market study", "liquidity spread order book",
                            (), (), 0.0, 0.0, "")
        strange = Preprint("y", 1, "Anyonic braiding", "topological quantum anyon braiding",
                           (), (), 0.0, 0.0, "")
        assert index.score(strange) > index.score(ordinary) + 0.5

    def test_stopwords_do_not_drive_the_score(self):
        index = NoveltyIndex().fit([Preprint("c", 1, "the and of", "we are in it", (), (), 0, 0, "")])
        assert "the" not in Preprint("x", 1, "the market", "", (), (), 0, 0, "").terms()

    def test_revisions_are_detected(self):
        first = parse_arxiv_atom(ATOM_SAMPLE)
        later = [Preprint(p.arxiv_id, p.version + 1, p.title, p.abstract, p.authors,
                          p.categories, p.published, p.updated, p.url) for p in first]
        assert len(revisions(first, later)) == 2
        assert revisions(first, first) == []


# ── SAR tank gauging ──────────────────────────────────────────────────────────

class TestSarTanks:
    def test_the_geometry_round_trips(self):
        tank = Tank("T", radius_m=40.0, wall_height_m=20.0)
        look = SarLook(incidence_deg=35.0, pixel_spacing_m=1.0)
        for truth in (0.1, 0.35, 0.7, 0.95):
            pixels = shadow_pixels_for_fill(tank, look, truth)
            assert estimate_fill(tank, look, pixels).fill_fraction == pytest.approx(truth, abs=1e-6)

    def test_slant_range_conversion_matters(self):
        tank = Tank("T", radius_m=40.0, wall_height_m=20.0)
        ground = SarLook(incidence_deg=35.0, pixel_spacing_m=1.0, slant_range=False)
        slant = SarLook(incidence_deg=35.0, pixel_spacing_m=1.0, slant_range=True)
        assert estimate_fill(tank, ground, 4.0).roof_depth_m != \
               pytest.approx(estimate_fill(tank, slant, 4.0).roof_depth_m)

    def test_a_coarse_sensor_admits_it_cannot_resolve_a_tank(self):
        tank = Tank("T", radius_m=40.0, wall_height_m=20.0)
        coarse = estimate_fill(tank, SarLook(35.0, 10.0),
                               shadow_pixels_for_fill(tank, SarLook(35.0, 10.0), 0.7))
        fine = estimate_fill(tank, SarLook(35.0, 1.0),
                             shadow_pixels_for_fill(tank, SarLook(35.0, 1.0), 0.7))
        assert not coarse.usable and fine.usable
        # Just under the 10x pixel-size ratio, and that shortfall is real physics:
        # incidence-angle uncertainty is a floor BOTH sensors share, so it does not
        # shrink with resolution. Buying finer pixels stops helping once the angle
        # error dominates.
        ratio = coarse.fill_sigma / fine.fill_sigma
        assert 8.0 < ratio < 10.0, f"sigma ratio {ratio:.2f} does not match the geometry"

    def test_resolvable_step_scales_with_pixel_size(self):
        tank = Tank("T", radius_m=40.0, wall_height_m=20.0)
        assert resolvable_fill_step(tank, SarLook(35.0, 10.0)) == pytest.approx(
            10 * resolvable_fill_step(tank, SarLook(35.0, 1.0)), rel=1e-9)

    def test_shallower_look_angles_give_longer_shadows(self):
        """More oblique looks buy precision. That is why look angle is chosen, not accepted."""
        tank = Tank("T", radius_m=40.0, wall_height_m=20.0)
        steep = resolvable_fill_step(tank, SarLook(20.0, 1.0))
        shallow = resolvable_fill_step(tank, SarLook(50.0, 1.0))
        assert shallow < steep

    def test_fill_is_clipped_to_physical_bounds_and_flagged(self):
        tank = Tank("T", radius_m=40.0, wall_height_m=20.0)
        estimate = estimate_fill(tank, SarLook(35.0, 1.0), 400.0)   # absurd shadow
        assert estimate.fill_fraction == 0.0 and estimate.saturated
        assert not estimate.usable

    def test_aggregation_shrinks_relative_error(self):
        tanks = [Tank(f"T{i}", 40.0, 20.0) for i in range(100)]
        look = SarLook(35.0, 1.0)
        estimates = [estimate_fill(t, look, shadow_pixels_for_fill(t, look, 0.6)) for t in tanks]
        one = aggregate_inventory(estimates[:1], tanks[:1])
        many = aggregate_inventory(estimates, tanks)
        assert many.relative_error < one.relative_error / 5

    def test_saturation_is_reported_as_bias_not_hidden(self):
        """Averaging kills variance, not bias. The aggregate must say when it is biased."""
        tanks = [Tank(f"T{i}", 40.0, 20.0) for i in range(50)]
        look = SarLook(35.0, 10.0)
        estimates = [estimate_fill(t, look, 400.0) for t in tanks]   # all saturated
        inventory = aggregate_inventory(estimates, tanks)
        assert inventory.biased and inventory.warnings()

    def test_bad_geometry_is_rejected(self):
        with pytest.raises(ValueError):
            estimate_fill(Tank("T", 40.0, 0.0), SarLook(35.0, 1.0), 1.0)
        with pytest.raises(ValueError):
            estimate_fill(Tank("T", 40.0, 20.0), SarLook(0.0, 1.0), 1.0)


# ── AIS ───────────────────────────────────────────────────────────────────────

class TestAisDecoding:
    def test_encode_decode_round_trip(self):
        sentence = encode_position(477123456, 1.2650, 103.8200, sog=12.4)
        parsed = parse_aivdm(sentence)
        position = decode_position(sixbit_to_bits(parsed.payload, parsed.fill_bits))
        assert position.mmsi == 477123456
        assert position.latitude == pytest.approx(1.2650, abs=1e-5)
        assert position.longitude == pytest.approx(103.8200, abs=1e-5)
        assert position.sog_knots == pytest.approx(12.4, abs=0.05)

    def test_southern_and_western_hemispheres(self):
        sentence = encode_position(200000000, -33.8688, -151.2093, sog=8.0)
        parsed = parse_aivdm(sentence)
        position = decode_position(sixbit_to_bits(parsed.payload, parsed.fill_bits))
        assert position.latitude == pytest.approx(-33.8688, abs=1e-5)
        assert position.longitude == pytest.approx(-151.2093, abs=1e-5)

    def test_checksum_is_verified(self):
        sentence = encode_position(477123456, 1.0, 103.0)
        assert parse_aivdm(sentence).checksum_ok
        assert not parse_aivdm(sentence[:-2] + "FF").checksum_ok

    def test_sixbit_armouring_round_trips(self):
        bits = "011000" * 7
        payload, fill = bits_to_sixbit(bits)
        assert sixbit_to_bits(payload, fill) == bits

    def test_garbage_is_rejected_not_guessed(self):
        with pytest.raises(AisDecodeError):
            parse_aivdm("hello world")
        with pytest.raises(AisDecodeError):
            parse_aivdm("!AIVDM,1,1,,A")
        with pytest.raises(AisDecodeError):
            sixbit_to_bits("\x01\x02")

    def test_a_non_position_message_is_refused(self):
        bits = format(5, "06b") + "0" * 200        # type 5 is static data, not a position
        with pytest.raises(AisDecodeError, match="not a position report"):
            decode_position(bits)

    def test_stream_decoding_skips_bad_lines(self):
        good = encode_position(1234, 1.0, 103.0)
        positions = decode_stream([good, "garbage", good], timestamps=[0.0, 1.0, 2.0])
        assert len(positions) == 2

    def test_invalid_positions_are_detected(self):
        assert not Position(1, 0, 91.0, 181.0, 0, 0, None, 0, 1).position_valid
        assert Position(1, 0, 1.0, 103.0, 0, 0, None, 0, 1).position_valid


class TestAisAnalytics:
    def test_haversine_against_a_known_distance(self):
        # London Heathrow to JFK is about 2,990 nm
        assert haversine_nm(51.4700, -0.4543, 40.6413, -73.7781) == pytest.approx(2990, rel=0.02)

    def test_gaps_are_found_and_short_silences_are_not(self):
        hour = 3600.0
        positions = [Position(1, 0 * hour, 1.0, 103.0, 10, 90, None, 0, 1),
                     Position(1, 2 * hour, 1.1, 103.1, 10, 90, None, 0, 1),
                     Position(1, 14 * hour, 1.9, 104.0, 10, 90, None, 0, 1)]
        gaps = find_gaps(Track(1, tuple(positions)), threshold_hours=6.0)
        assert len(gaps) == 1 and gaps[0].hours == pytest.approx(12.0)

    def test_an_impossible_jump_is_the_one_signal_coverage_cannot_explain(self):
        hour = 3600.0
        positions = [Position(1, 0.0, 1.0, 103.0, 10, 90, None, 0, 1),
                     Position(1, 7 * hour, 40.0, 140.0, 10, 90, None, 0, 1)]
        gap = find_gaps(Track(1, tuple(positions)), threshold_hours=6.0)[0]
        assert gap.implausible(max_speed_knots=25.0)
        assert "spoofing" in gap.assessment()

    def test_an_ordinary_gap_is_described_as_ordinary(self):
        hour = 3600.0
        positions = [Position(1, 0.0, 1.0, 103.0, 10, 90, None, 0, 1),
                     Position(1, 10 * hour, 1.5, 103.5, 10, 90, None, 0, 1)]
        gap = find_gaps(Track(1, tuple(positions)), threshold_hours=6.0)[0]
        assert not gap.implausible()
        assert "coverage loss" in gap.assessment() and "corroborate" in gap.assessment()

    def test_the_language_never_accuses(self):
        """A coverage hole is not evidence, and the vocabulary must not imply it is."""
        hour = 3600.0
        gap = find_gaps(Track(1, (Position(1, 0.0, 1.0, 103.0, 10, 90, None, 0, 1),
                                  Position(1, 20 * hour, 2.0, 104.0, 10, 90, None, 0, 1))),
                        threshold_hours=6.0)[0]
        text = gap.assessment().lower()
        for word in ("illegal", "smuggl", "sanction", "evading", "guilty"):
            assert word not in text

    def test_rendezvous_needs_close_slow_and_sustained(self):
        hour = 3600.0
        a = Track(1, tuple(Position(1, t * hour, 3.5, 105.2, 0.2, 0, None, 1, 1)
                           for t in range(10, 18)))
        b = Track(2, tuple(Position(2, t * hour, 3.501, 105.2008, 0.3, 0, None, 1, 1)
                           for t in range(10, 18)))
        assert len(find_rendezvous([a, b])) == 1

    def test_ships_passing_at_speed_are_not_a_rendezvous(self):
        hour = 3600.0
        a = Track(1, tuple(Position(1, t * hour, 3.5, 105.2, 14.0, 0, None, 0, 1)
                           for t in range(10, 18)))
        b = Track(2, tuple(Position(2, t * hour, 3.501, 105.2008, 15.0, 0, None, 0, 1)
                           for t in range(10, 18)))
        assert find_rendezvous([a, b]) == []

    def test_distant_ships_are_not_a_rendezvous(self):
        hour = 3600.0
        a = Track(1, tuple(Position(1, t * hour, 3.5, 105.2, 0.1, 0, None, 1, 1)
                           for t in range(10, 18)))
        b = Track(2, tuple(Position(2, t * hour, 8.0, 110.0, 0.1, 0, None, 1, 1)
                           for t in range(10, 18)))
        assert find_rendezvous([a, b]) == []

    def test_tracks_group_by_mmsi_and_drop_invalid_fixes(self):
        positions = [Position(1, 0, 1.0, 103.0, 0, 0, None, 0, 1),
                     Position(2, 0, 2.0, 104.0, 0, 0, None, 0, 1),
                     Position(1, 1, 91.0, 181.0, 0, 0, None, 0, 1)]
        tracks = build_tracks(positions)
        assert len(tracks) == 2
        assert len(tracks[0].positions) == 1, "an invalid fix was kept"
