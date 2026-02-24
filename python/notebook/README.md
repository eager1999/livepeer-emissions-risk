# Data Retrieval Notes

This folder contains marimo notebooks used to analyze Livepeer emissions and participation risk. The notebooks consume datasets that are generated outside the notebook runtime. This document describes how the external data is retrieved.

## Sources and retrieval methods

### 1) On-chain protocol state (Arbitrum)
The primary on-chain dataset (daily snapshots of inflation, total supply, and bonded stake) is retrieved using the script:

- `python/script/fetch-data.py`

This script performs two steps:

1. **Get daily Arbitrum block numbers**
   - Uses the Etherscan API for Arbitrum (`chainid=42161`) to resolve the block number closest to each day’s timestamp.
   - Endpoint: `module=block`, `action=getblocknobytime`.
   - Requires environment variable: `ETHERSCAN_API_KEY`.

2. **Fetch historical contract state at those blocks**
   - Uses a Web3 RPC connection to an Arbitrum archive node.
   - Requires environment variable: `ARB_RPC_URL`.
   - Contracts and ABIs are loaded from `protocol/deployments/arbitrumMainnet/`:
     - `Minter.json` (inflation rate, total supply)
     - `BondingManager.json` + `BondingManagerTarget.json` (total bonded stake)

The resulting JSON files are written to the project’s `../data/` directory:

- `../data/arbitrum-daily-blocks-22-24.json` (daily block numbers)
- `../data/lpt-daily-data-22-24.json` (inflation, total supply, bonded, and dates)

The notebooks read these files via helpers in `python/src/data/`.

### 1b) Local on-chain daily JSON used in analysis
The `python/notebook/Data Fetching.ipynb` notebook reads a local JSON export:

- `lpt-daily-data-22-25.json`

This file has the same shape as the JSON produced by `python/script/fetch-data.py` and is used as the base daily dataset in the notebook.

### 2) Per-round historical dataset (CSV)
The forecasting notebooks use a per-round CSV dataset (round-level participation and exogenous variables). The default path is set in:

- `python/src/forecast/config.py` via `DEFAULT_ROUND_DATA_PATH`

This file is expected to live outside the repo (e.g., under your `ShtukaResearch` data directory). The path can be overridden using:

- `LPT_ROUND_DATA_PATH=/path/to/Data2022-2025[perRound].csv`

The notebooks load this CSV using `python/src/forecast/data.py`.

### 3) Exogenous market data (prices, volumes, sentiment)
The per-round CSV also contains external market variables used for regressions and simulations, such as:

- `fear_greed_index`
- `btc_price_usd`, `eth_price_usd`, `lpt_price_usd`
- `btc_volume`, `eth_volume`, `lpt_volume`

In `python/notebook/Data Fetching.ipynb`, these are retrieved as follows:

- Prices and volumes are pulled from Yahoo Finance using `yfinance` for the tickers `LPT-USD`, `BTC-USD`, `ETH-USD`, on a daily interval.
- The Fear & Greed index is fetched from `https://api.alternative.me/fng/` (all available history), filtered to the same date range.

The notebook merges these series with the on-chain daily JSON and writes:

- `Data2022-2025.csv` (daily combined dataset)
- `Data2022-2025[perRound].csv` (round-level dataset)

The round-level dataset is produced by interpolating daily rows to round indices using ~1.142857 rounds/day and a nearest-neighbor pass for the date field. The saved file is the nearest-neighbor version.

## Reproducing data retrieval

1) Fetch on-chain daily state (example):

```bash
python script/fetch-data.py --ticks --state --start-date 2022-01-01 --end-date 2024-01-01
```

2) Point notebooks to the data paths via environment variables (recommended):

```bash
export LPT_DATA_PATH=../data/lpt-daily-data-22-24.json
export LPT_ROUND_DATA_PATH=/path/to/Data2022-2025[perRound].csv
```

3) Run a notebook:

```bash
marimo run python/notebook/emissions-history.py
```

## Notes

- The notebooks do not fetch data directly; they only load precomputed JSON/CSV files.
- Ensure your RPC endpoint supports historical state queries (archive access) for the requested date range.
