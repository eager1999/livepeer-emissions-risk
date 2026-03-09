import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import argparse
    import json
    import os
    import sys
    import time
    from datetime import datetime, timedelta
    from pathlib import Path

    import pandas as pd
    import requests
    from pytz import UTC
    from requests.exceptions import ReadTimeout
    from web3 import Web3

    API_URL = "https://api.etherscan.io/v2/api?chainid=42161"

    SCRIPT_DIR = Path(__file__).resolve().parent
    PYTHON_DIR = SCRIPT_DIR.parent
    REPO_ROOT = PYTHON_DIR.parent

    DATA_DIR = PYTHON_DIR / "data"
    DEPLOYMENTS_DIR = REPO_ROOT / "protocol" / "deployments" / "arbitrumMainnet"

    DEFAULT_TICKS_FILE = DATA_DIR / "arbitrum-daily-blocks.json"
    DEFAULT_STATE_FILE = DATA_DIR / "lpt-daily-data.json"
    DEFAULT_MERGED_FILE = DATA_DIR / "Data.csv"

    return (
        API_URL,
        DATA_DIR,
        DEFAULT_MERGED_FILE,
        DEFAULT_STATE_FILE,
        DEFAULT_TICKS_FILE,
        DEPLOYMENTS_DIR,
        UTC,
        Web3,
        argparse,
        datetime,
        json,
        os,
        pd,
        requests,
        sys,
        time,
        timedelta,
        ReadTimeout,
        Path,
    )


@app.cell
def _(
    API_URL,
    DATA_DIR,
    DEFAULT_MERGED_FILE,
    DEFAULT_STATE_FILE,
    DEFAULT_TICKS_FILE,
    DEPLOYMENTS_DIR,
    UTC,
    Web3,
    argparse,
    datetime,
    json,
    os,
    pd,
    requests,
    time,
    timedelta,
    ReadTimeout,
    Path,
):
    def parse_date(date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)

    def date_range_from_args(args: argparse.Namespace) -> tuple[datetime, datetime, int]:
        start = parse_date(args.start_date)
        if args.end_date:
            end = parse_date(args.end_date)
            num_days = (end - start).days
        else:
            num_days = args.num_days
            end = start + timedelta(days=num_days)

        if num_days <= 0:
            raise ValueError("Date range must include at least one day.")

        return start, end, num_days

    def get_block_number_by_time(apikey: str, timestamp: datetime) -> int:
        params = {
            "module": "block",
            "action": "getblocknobytime",
            "timestamp": int(timestamp.timestamp()),
            "closest": "before",
            "apikey": apikey,
        }
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") == "1":
            return int(payload["result"])
        raise ValueError(f"Etherscan error: {payload.get('message', 'unknown error')}")

    def fetch_arbitrum_daily_blocks(apikey: str, start: datetime, num_days: int) -> dict[str, list]:
        blocks = {"date": [], "block": []}
        current_time = start

        for _ in range(num_days):
            retries_left = 3
            while retries_left > 0:
                try:
                    block_number = get_block_number_by_time(apikey, current_time)
                    blocks["block"].append(block_number)
                    blocks["date"].append(current_time.strftime("%Y-%m-%d"))
                    break
                except (ValueError, ReadTimeout, requests.RequestException) as exc:
                    retries_left -= 1
                    if retries_left == 0:
                        raise RuntimeError(f"Failed to fetch block for {current_time.date()}") from exc
                    time.sleep(2 ** (3 - retries_left))

            current_time += timedelta(days=1)
            time.sleep(0.2)

        return blocks

    def arbitrum_w3() -> Web3:
        arb_rpc_url = os.getenv("ARB_RPC_URL")
        if not arb_rpc_url:
            raise EnvironmentError("ARB_RPC_URL is required for on-chain state fetching.")
        return Web3(Web3.HTTPProvider(arb_rpc_url))

    def load_contract_from_json(w3: Web3, path: Path):
        with path.open("r") as file:
            contract_json = json.load(file)
        return w3.eth.contract(address=contract_json["address"], abi=contract_json["abi"])

    def bonding_manager(w3: Web3):
        implementation_path = DEPLOYMENTS_DIR / "BondingManagerTarget.json"
        deployment_path = DEPLOYMENTS_DIR / "BondingManager.json"

        with implementation_path.open("r") as handle:
            contract_abi = json.load(handle)["abi"]
        with deployment_path.open("r") as handle:
            contract_address = json.load(handle)["address"]

        return w3.eth.contract(address=contract_address, abi=contract_abi)

    def fetch_historic(callable_contract_fn, blocks: list[int]) -> list[int]:
        values = []
        for block in blocks:
            values.append(callable_contract_fn.call(block_identifier=block))
            time.sleep(0.05)
        return values

    def fetch_ticks(start: datetime, num_days: int) -> dict[str, list]:
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not etherscan_api_key:
            raise EnvironmentError("ETHERSCAN_API_KEY is required for tick fetching.")

        return fetch_arbitrum_daily_blocks(
            apikey=etherscan_api_key,
            start=start,
            num_days=num_days,
        )

    def fetch_state(block_nums: dict[str, list]) -> dict[str, list]:
        if not DEPLOYMENTS_DIR.exists():
            raise FileNotFoundError(f"Missing deployments directory: {DEPLOYMENTS_DIR}")

        w3 = arbitrum_w3()

        minter = load_contract_from_json(w3, DEPLOYMENTS_DIR / "Minter.json")
        bonding = bonding_manager(w3)

        callables = {
            "inflation": minter.functions.inflation(),
            "total-supply": minter.functions.getGlobalTotalSupply(),
            "bonded": bonding.functions.getTotalBonded(),
        }

        results = {key: fetch_historic(fn, block_nums["block"]) for key, fn in callables.items()}
        return results | block_nums

    def extend_state(old: dict[str, list], extension: tuple[dict[str, list], dict[str, list]]) -> dict[str, list]:
        before, after = extension
        return {key: before[key] + old[key] + after[key] for key in old}

    def load_state_dataframe(state_json_path: Path) -> pd.DataFrame:
        with state_json_path.open("r") as handle:
            payload = json.load(handle)

        df = pd.DataFrame(payload)
        if "date" not in df.columns:
            raise ValueError(f"State file missing 'date' column: {state_json_path}")

        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def parse_api_dates(series: pd.Series) -> pd.Series:
        """Parse API date fields that may be unix seconds or date strings."""
        numeric_values = pd.to_numeric(series, errors="coerce")
        parsed_from_unix = pd.to_datetime(numeric_values, unit="s", errors="coerce", utc=True)
        parsed_from_text = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce", utc=True)
        return parsed_from_unix.fillna(parsed_from_text)

    def fetch_market_prices(start_date: str, end_date: str, tickers: list[str]) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for market enrichment. Install with: pip install yfinance"
            ) from exc

        # yfinance treats the end bound as exclusive, so request one extra day.
        end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(tickers, start=start_date, end=end_exclusive, interval="1d", progress=False)
        if raw.empty:
            raise ValueError("No market data returned from yfinance for the selected range.")

        close_df = raw["Close"].reset_index()
        volume_df = raw["Volume"].reset_index()

        rename_close = {"Date": "date"}
        rename_volume = {"Date": "date"}

        for symbol in tickers:
            base = symbol.split("-")[0].lower()
            rename_close[symbol] = f"{base}_price_usd"
            rename_volume[symbol] = f"{base}_volume"

        close_df.rename(columns=rename_close, inplace=True)
        volume_df.rename(columns=rename_volume, inplace=True)

        df = pd.merge(close_df, volume_df, on="date", how="inner")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_fear_greed_index(start_date, end_date) -> pd.DataFrame:
        url = "https://api.alternative.me/fng/?limit=0"
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        rows = response.json().get("data", [])
        if not rows:
            return pd.DataFrame(columns=["date", "fear_greed_index"])

        df = pd.DataFrame(rows)

        df["date"] = parse_api_dates(df["timestamp"])
        df = df[["date", "value"]].rename(columns={"value": "fear_greed_index"})
        df["fear_greed_index"] = pd.to_numeric(df["fear_greed_index"], errors="coerce")

        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        df["date"] = df["date"].dt.date

        return df.sort_values("date").reset_index(drop=True)

    def build_merged_dataset(state_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        start = state_df["date"].iloc[0].strftime("%Y-%m-%d")
        end = state_df["date"].iloc[-1].strftime("%Y-%m-%d")

        market_df = fetch_market_prices(start_date=start, end_date=end, tickers=tickers)

        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        fear_df = fetch_fear_greed_index(start_date, end_date)

        merged = fear_df.merge(market_df, on="date", how="inner")
        merged = merged.merge(state_df, on="date", how="inner")
        return merged

    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Livepeer data fetch: on-chain + market enrichment")

        parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD)")

        date_group = parser.add_mutually_exclusive_group(required=True)
        date_group.add_argument("--end-date", help="Exclusive end date (YYYY-MM-DD)")
        date_group.add_argument("--num-days", type=int, help="Number of days from start date")

        parser.add_argument("--ticks", action="store_true", help="Fetch Arbitrum daily block numbers")
        parser.add_argument("--state", action="store_true", help="Fetch on-chain historic state")
        parser.add_argument("--market", action="store_true", help="Build merged dataset with market + fear/greed")
        parser.add_argument("--extend", type=str, help="Path to existing state JSON to extend")

        parser.add_argument("--ticks-file", type=Path, default=DEFAULT_TICKS_FILE)
        parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
        parser.add_argument("--merged-file", type=Path, default=DEFAULT_MERGED_FILE)
        parser.add_argument(
            "--tickers",
            nargs="+",
            default=["LPT-USD", "BTC-USD", "ETH-USD"],
            help="Ticker symbols for yfinance",
        )

        return parser.parse_args()

    def main() -> None:
        args = parse_args()
        start_date, end_date, num_days = date_range_from_args(args)

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        run_all = not any([args.ticks, args.state, args.market])
        run_ticks = args.ticks or run_all
        run_state = args.state or run_all
        run_market = args.market or run_all

        block_nums = None
        state_payload = None

        if args.extend:
            extend_path = Path(args.extend)
            with extend_path.open("r") as handle:
                old_state = json.load(handle)

            old_start_date = parse_date(old_state["date"][0])
            old_end_date = parse_date(old_state["date"][-1]) + timedelta(days=1)

            if old_start_date < start_date or old_end_date > end_date:
                raise ValueError("Requested date range is not an extension of existing file range.")

            extension = []
            ranges = [
                (start_date, (old_start_date - start_date).days),
                (old_end_date, (end_date - old_end_date).days),
            ]

            for extension_start, extension_days in ranges:
                if extension_days == 0:
                    extension.append({key: [] for key in old_state.keys()})
                    continue
                extension_ticks = fetch_ticks(extension_start, extension_days)
                extension.append(fetch_state(extension_ticks))

            state_payload = extend_state(old_state, (extension[0], extension[1]))
            with extend_path.open("w") as handle:
                json.dump(state_payload, handle)
            print(f"Extended state file: {extend_path}")

        else:
            if run_ticks:
                block_nums = fetch_ticks(start_date, num_days)
                args.ticks_file.parent.mkdir(parents=True, exist_ok=True)
                with args.ticks_file.open("w") as handle:
                    json.dump(block_nums, handle)
                print(f"Saved ticks: {args.ticks_file}")

            if run_state:
                if block_nums is None:
                    with args.ticks_file.open("r") as handle:
                        block_nums = json.load(handle)
                state_payload = fetch_state(block_nums)
                args.state_file.parent.mkdir(parents=True, exist_ok=True)
                with args.state_file.open("w") as handle:
                    json.dump(state_payload, handle)
                print(f"Saved on-chain state: {args.state_file}")

        if run_market:
            if state_payload is None:
                state_df = load_state_dataframe(args.state_file)
            else:
                state_df = pd.DataFrame(state_payload)
                state_df["date"] = pd.to_datetime(state_df["date"]).dt.date

            merged_df = build_merged_dataset(state_df=state_df, tickers=args.tickers)
            args.merged_file.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_csv(args.merged_file, index=False)
            print(f"Saved merged daily dataset: {args.merged_file} ({merged_df.shape[0]} rows)")

    return main


@app.cell
def _(main, sys):
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    return


if __name__ == "__main__":
    app.run()
