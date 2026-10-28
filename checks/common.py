"""Shared result format for every VISION check, whatever the industry.

Every check answers the same question - is this result real? - and reports in
the same shape, so one page can render a trading backtest, an A/B test, an ML
model or a code audit side by side.
"""
from dataclasses import dataclass, field

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"


@dataclass
class Check:
    name: str
    status: str            # pass | warn | fail | skip
    reason: str = ""       # plain English: what's wrong (or why it was skipped)
    fix: str = ""          # plain English: what to do about it
    detail: dict = field(default_factory=dict)


def report(kind: str, checks: list[Check], n: int | None = None, unit: str = "rows") -> dict:
    """Roll checks up into one verdict. Any fail -> FAKE; only warnings -> QUESTIONABLE."""
    ran = [c for c in checks if c.status != SKIP]
    fails = [c for c in ran if c.status == FAIL]
    warns = [c for c in ran if c.status == WARN]
    if not ran:
        verdict = "NOT ENOUGH TO JUDGE"
    elif fails:
        verdict = "LOOKS FAKE"
    elif warns:
        verdict = "QUESTIONABLE"
    else:
        verdict = "LOOKS REAL"
    passed = sum(c.status == PASS for c in ran)
    return {
        "kind": kind,
        "verdict": verdict,
        "score": f"{passed}/{len(ran)} checks passed",
        "n": n,
        "unit": unit,
        "checks": checks,
    }


def merge(kind: str, reports: list[dict]) -> dict:
    """Combine several reports (e.g. results + the code that produced them)."""
    checks = [c for r in reports for c in r["checks"]]
    return report(kind, checks, n=None)
