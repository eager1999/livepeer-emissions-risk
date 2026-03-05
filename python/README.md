# lpt-stake (Python)

Python tooling for Livepeer emissions-risk data retrieval and analysis.

## Main scripts

- `python/script/data-fetching.py`: unified data pipeline for
  - Arbitrum daily block fetch (`--ticks`)
  - Livepeer on-chain state fetch (`--state`)
  - Market/sentiment enrichment (`--market`)

## Environment and setup

From `python/`:

```bash
uv sync
```

Required env vars for on-chain fetch:

- `ETHERSCAN_API_KEY`
- `ARB_RPC_URL`

Example:

```bash
export ETHERSCAN_API_KEY=...
export ARB_RPC_URL=...
```

## Data fetching examples

Run full pipeline (ticks + state + market):

```bash
uv run python script/data-fetching.py --start-date 2022-01-01 --end-date 2024-01-01
```

Run only on-chain steps:

```bash
uv run python script/data-fetching.py --start-date 2022-01-01 --end-date 2024-01-01 --ticks --state
```

Run only market merge from an existing state file:

```bash
uv run python script/data-fetching.py --start-date 2022-01-01 --end-date 2024-01-01 --market --state-file python/data/lpt-daily-data.json
```

## Outputs (default)

- `python/data/arbitrum-daily-blocks.json`
- `python/data/lpt-daily-data.json`
- `python/data/Data2022-2025.csv`

All output paths are overridable via CLI flags.
