#!/usr/bin/env python3
"""
One-off real-data pull: ES front-month tick history from Rithmic.

This is intentionally OUTSIDE stark-core/ - stark-core makes no network calls
and accepts only plain arrays (CLAUDE.md, stark-core/CLAUDE.md rule 5). This
script does the one network call, then writes a flat file that the pipeline
script reads.

Credentials come from /Users/j/.config/stark/rithmic.env (outside any git
repo, chmod 600) - never hardcoded, never logged.

Usage: python3 real_data/pull_ticks.py
"""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from async_rithmic import RithmicClient

ENV_FILE = Path("/Users/j/.config/stark/rithmic.env")
OUT_FILE = Path(__file__).parent / "es_ticks_raw.jsonl"


def load_env():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k] = v  # explicit override: the env file is the source of truth


async def main():
    load_env()
    client = RithmicClient(
        user=os.environ["RITHMIC_USER"],
        password=os.environ["RITHMIC_PASSWORD"],
        system_name=os.environ.get("RITHMIC_SYSTEM_NAME", "LucidTrading"),
        app_name="STARK:real_data_pull",
        app_version="1.0",
        url="rprotocol.rithmic.com:443",
    )

    await client.connect()
    try:
        symbol = await client.get_front_month_contract("ES", "CME")
        print(f"Front-month ES contract: {symbol}")

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=5)
        print(f"Requesting ticks {start.isoformat()} -> {end.isoformat()} ...")

        ticks = await client.get_historical_tick_data(
            symbol=symbol,
            exchange="CME",
            start_time=start,
            end_time=end,
            wait=True,
            idle_timeout=15.0,
            max_pages=2000,
        )
        print(f"Received {len(ticks)} ticks.")

        if ticks:
            print("Sample tick fields:", sorted(ticks[0].keys()) if isinstance(ticks[0], dict) else type(ticks[0]))
            print("First tick:", ticks[0])
            print("Last tick:", ticks[-1])

        with open(OUT_FILE, "w") as f:
            for t in ticks:
                f.write(json.dumps(t, default=str) + "\n")
        print(f"Wrote {len(ticks)} ticks to {OUT_FILE}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
