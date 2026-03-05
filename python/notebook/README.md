# Notebook Data Notes

This folder contains marimo notebooks for Livepeer emissions and participation analysis.

## Data source workflow

Notebooks consume prepared datasets. Data retrieval is handled by:

- `python/script/data-fetching.py`

The script can produce:

1. On-chain daily snapshots
- Arbitrum daily block ticks
- Livepeer state (`inflation`, `total-supply`, `bonded`)

2. Daily enriched dataset
- Yahoo Finance prices/volumes for default tickers: `LPT-USD`, `BTC-USD`, `ETH-USD`
- Fear & Greed index from `https://api.alternative.me/fng/`
- Joined dataset written to CSV

## Default files used by analysis

- `python/data/arbitrum-daily-blocks.json`
- `python/data/lpt-daily-data.json`
- `python/data/Data2022-2025.csv`

## Reproduce data retrieval

From `python/`:

```bash
uv run python script/data-fetching.py --start-date 2022-01-01 --end-date 2024-01-01
```

For on-chain only:

```bash
uv run python script/data-fetching.py --start-date 2022-01-01 --end-date 2024-01-01 --ticks --state
```

For market merge only (using existing state JSON):

```bash
uv run python script/data-fetching.py --start-date 2022-01-01 --end-date 2024-01-01 --market --state-file python/data/lpt-daily-data.json
```

## Notes

- `data-fetching.py` is the canonical fetch script.
- Legacy `fetch-data.py` has been removed.
- Ensure your RPC endpoint has archive access for historical state reads.
