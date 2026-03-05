# Livepeer Emissions Risk Framework

This repository contains research artifacts and tooling for analyzing Livepeer emissions, participation, and risk.

## Repository structure

- `objectives.md`: economic objective framing.
- `risk/`: risk framework notes.
- `survey/`: survey outputs (notes and figures).
- `python/`: data retrieval scripts, analysis notebooks, and Python package code.

## Python workflow

The main data retrieval entrypoint is:

- `python/script/data-fetching.py`

It can fetch:

- Daily Arbitrum block ticks from Etherscan.
- Livepeer on-chain state at those ticks (`inflation`, `total-supply`, `bonded`).
- Market and sentiment enrichment (`LPT/BTC/ETH` prices + volumes and Fear & Greed index).

See `python/README.md` for setup and commands.
