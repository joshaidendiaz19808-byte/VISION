# Ingestion — standing rules

This tree is OUTSIDE `stark-core/` on purpose. The root `CLAUDE.md` says STARK
holds no keys and makes no network calls; ingestion does both, so it lives here
and hands STARK what STARK accepts — plain arrays with provenance attached. A
bug in a crawler must not be able to reach a risk limit.

## Rules

1. **Provenance is mandatory.** Every record carries its source URL, fetch time
   and content hash. A number whose origin cannot be reconstructed is not data,
   it is a rumour.
2. **Politeness is not optional.** Every source declares a rate limit and honours
   `Retry-After`. arXiv's published limit is one request per three seconds and
   that is the default here. A source whose terms forbid automated access does
   not get a client written for it.
3. **Point-in-time or say so.** Macro series carry release dates. Where they do
   not, `is_point_in_time` is False, `as_of()` refuses rather than guessing, and
   the warning is loud. Backtesting on revised data is look-ahead bias in a suit.
4. **Credentials come from the environment, never the repo.** `FRED_API_KEY` and
   friends are read at runtime. No key is ever written into this tree.
5. **Parsers refuse rather than guess.** A malformed AIS sentence, an
   unparseable date, a FRED `"."` — skipped and counted, never zero-filled.
6. **A flag is a lead, not a finding.** This applies hardest to `telemetry/ais.py`:
   an AIS gap is usually a receiver coverage hole. The vocabulary is deliberately
   flat — `gap`, `unexplained`, `co_located` — and never `illegal` or
   `sanctioned`. Publishing an accusation about a named vessel on the strength of
   a coverage hole would be both analytically illiterate and defamatory.
7. **stdlib + numpy only.** No HTTP library is worth a dependency for a GET.

## What is here

| Module | Does | Does NOT |
|---|---|---|
| `base.py` | rate limiting, caching, retries, provenance | anything source-specific |
| `macro_streams.py` | point-in-time macro series, FRED/ECB/CSV | hold any API key |
| `preprints.py` | arXiv API, version tracking, novelty triage | read or judge papers |
| `telemetry/sar_tanks.py` | shadow geometry, error propagation, aggregation | touch a satellite |
| `telemetry/ais.py` | NMEA decode, gap and rendezvous screening | accuse anyone |

No imagery pipeline, no AIS feed, no news scraper. Shadow widths and AIS
sentences arrive as inputs from a provider you have a relationship with.

## Verification

`pytest tests/test_ingestion.py -q` — 58 cases, none of which touch the network.
Transports are injected, clocks are fake, parsers are fed bytes. A suite that
needs the internet fails when someone else's server has a bad afternoon.
