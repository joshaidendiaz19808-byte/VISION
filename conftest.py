"""Put the repo root, stark-core, and the ingestion package on sys.path for pytest."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
for path in (ROOT, ROOT / "stark-core", ROOT / "ingestion"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
