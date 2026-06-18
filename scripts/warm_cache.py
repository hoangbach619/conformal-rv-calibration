#!/usr/bin/env python
"""One-off: warm the data cache for every index and VIX before a grid run.

A grid run must never hit the network mid-run (a rate-limit mid-flight crashes
it), so this pre-fetches all eight indices and VIX into the gitignored
``data/raw`` cache up front. It loops the tickers sequentially with a short
sleep between pulls to stay under the Yahoo rate limit; ``load_index_ohlc``
falls back to the curl_cffi chart-JSON endpoint when yfinance is rate-limited.
Each ticker is printed as it caches, and a non-zero exit status signals that at
least one ticker failed (so a launcher can refuse to start the grid).

    PYTHONPATH=src python scripts/warm_cache.py
"""

from __future__ import annotations

import sys
import time

from conformal_rv.data import INDICES, VIX, load_index_ohlc

# Seconds between tickers; a few seconds keeps the burst under the rate limit.
_SLEEP_SECONDS = 5


def main() -> int:
    tickers = [*INDICES, VIX]
    cached: list[str] = []
    failed: list[str] = []
    for position, ticker in enumerate(tickers):
        try:
            frame = load_index_ohlc(ticker)[ticker]
            if frame.empty:
                raise RuntimeError("loaded an empty frame")
            print(f"cached  {ticker:<10} {len(frame):>6} rows", flush=True)
            cached.append(ticker)
        except Exception as exc:
            print(f"FAILED  {ticker:<10} {exc}", flush=True)
            failed.append(ticker)
        # Sleep only between network-eligible pulls, not after the last ticker.
        if position < len(tickers) - 1:
            time.sleep(_SLEEP_SECONDS)

    print(f"\n{len(cached)}/{len(tickers)} cached", flush=True)
    if failed:
        print(f"FAILED: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
