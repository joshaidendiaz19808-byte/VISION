"""Put the repo root, stark-core, and the ingestion package on sys.path for pytest."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
for path in (ROOT, ROOT / "stark-core", ROOT / "ingestion"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_usage_limits():
    """Each test starts with an empty usage counter so the suite never trips the per-hour limit."""
    try:
        from api import server
    except Exception:
        yield
        return
    server.HITS.clear()
    yield
    server.HITS.clear()
