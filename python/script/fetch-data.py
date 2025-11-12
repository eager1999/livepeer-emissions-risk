# Script for fetching Livepeer staking data from Arbitrum.
# First, fetch daily Arbitrum block numbers at the same time

from datetime import datetime, timedelta
import os, sys
import argparse
import urllib3
import requests
from requests.exceptions import ReadTimeout
import time
import json
from pytz import UTC
from web3 import Web3
import arrow

retries = urllib3.util.Retry(
    total=5,  # Total number of retries
    backoff_factor=1,  # Exponential backoff factor (e.g., 1, 2, 4, 8, 16 seconds)
    status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
)

API_URL = "https://api.etherscan.io/v2/api?chainid=42161"

def get_block_number_by_time(apikey: str, timestamp: datetime) -> int:
    url = API_URL
    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": int(timestamp.timestamp()),
        "closest": "before",
        "apikey": apikey
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    if data["status"] == "1":
        return int(data["result"])
    else:
        raise ValueError(f"Error fetching block number: {data['message']}")



def fetch_arbitrum_daily_blocks(apikey: str, start: datetime, num_days: int) -> list[int]:
    """
    Fetch Arbitrum block numbers at num_days daily intervals starting from start. 
    In case of failed calls, retry a few times with exponential backoff.
    """
    blocks = {
        "date": [],
        "block": []
    }
    current_time = start
    for _ in range(num_days):
        print(current_time)
        retries = 3
        while retries > 0:
            try:
                block_number = get_block_number_by_time(apikey, current_time)
                blocks["block"].append(block_number)
                blocks["date"].append(current_time.strftime("%Y-%m-%d"))
                break
            except (ValueError, ReadTimeout) as e:
                print(f"Error fetching block number: {e}. Retrying...")
                retries -= 1
                time.sleep(2 ** (3 - retries))  # Exponential backoff

        else:
            raise RuntimeError(f"Failed to fetch block number for {current_time} after multiple retries.")
        current_time += timedelta(days=1)
        time.sleep(0.2)
    return blocks


# Fetching blockchain data
DEPLOYMENTS = "../protocol/deployments/arbitrumMainnet"
MINTER_DEPLOYMENT_JSON = os.path.join(DEPLOYMENTS, "Minter.json")
BONDING_MANAGER_DEPLOYMENT_JSON = os.path.join(DEPLOYMENTS, "BondingManager.json")
BONDING_MANAGER_IMPLEMENTATION_JSON = os.path.join(DEPLOYMENTS, "BondingManagerTarget.json")

def arbitrum_w3():
    "Create default Web3 object from Arbitrum RPC URL."
    arb_rpc_url = os.getenv("ARB_RPC_URL")
    return Web3(Web3.HTTPProvider(arb_rpc_url))

def load_contract_from_json(w3, path):
    "Construct Contract object by parsing JSON loaded from path."
    with open(path, 'r') as file:
        contract_json = json.load(file)
    contract_abi = contract_json['abi']
    contract_address = contract_json['address']
    return w3.eth.contract(address=contract_address, abi=contract_abi)

def bonding_manager(w3):
    with open(BONDING_MANAGER_IMPLEMENTATION_JSON) as h:
        contract_abi = json.load(h)['abi']
    with open(BONDING_MANAGER_DEPLOYMENT_JSON) as h:
        contract_address = json.load(h)['address']
    return w3.eth.contract(address=contract_address, abi=contract_abi)
    

def fetch_historic(callable, blocks: list[int]) -> list[int]:
    results = []
    for block in blocks:
        results.append(callable.call(block_identifier=block))
        time.sleep(0.05)
    return results

def help() -> list[str]:
    return [
        "Set environment variables:",
        "ETHERSCAN_API_KEY=<Etherscan API key>\t\t(for fetching Arbitrum block numbers)",
        "ARB_RPC_URL=<Arbitrum archive node RPC URL>\t(for fetching historic state)"
    ]

def parse_args():
    """
    Parse arguments using standard library arg parser module.

    Options:
    --ticks     Fetch Arbitrum block numbers.
    --state     Fetch historic state at ticks in data directory. If --ticks option is present, fetch ticks first and use those.
    """
    parser = argparse.ArgumentParser(description="Fetch Livepeer staking data from Arbitrum.")
    parser.add_argument(
        "--ticks",
        action="store_true",
        help="Fetch Arbitrum block numbers."
    )
    parser.add_argument(
        "--state",
        action="store_true",
        help="Fetch historic state at ticks in data directory. If --ticks option is present, fetch ticks first and use those."
    )
    # Add option for specifying start date and either end date or number of days, not both
    parser.add_argument(
        "--start-date",
        type=(lambda s: arrow.get(s).datetime),
        required=True,
        help="Start date in YYYY-MM-DD format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--end-date",
        type=(lambda s: arrow.get(s).datetime),
        help="End date in YYYY-MM-DD format."
    )
    group.add_argument(
        "--num-days",
        type=int,
        help="Number of days from the start date."
    )
    return parser.parse_args()

TICKS_FILE = "../data/arbitrum-daily-blocks-22-24.json"
DATA_FILE = "../data/lpt-daily-data-22-24.json"

if __name__ == "__main__":
    args = parse_args()

    # Compute num_days from end_date if not provided
    if args.end_date:
        num_days = (args.end_date - args.start_date).days
    else:
        num_days: int = args.num_days
        if num_days <= 0:
            print("Number of days must be greater than 0.")
            sys.exit(1)

    # fetch ticks
    if args.ticks:
        arbiscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not arbiscan_api_key:
            print(*help(), sep='\n')
            sys.exit(1)

        print("Fetching Arbitrum block numbers...")
        block_nums: dict[str,list] = fetch_arbitrum_daily_blocks(
            apikey = arbiscan_api_key, 
            start=args.start_date, 
            num_days=num_days
        )
        
        with open(TICKS_FILE, 'w') as h:
            json.dump(block_nums, h)

    # fetch historic state
    if args.state:
        if not args.ticks: # need to use ticks file
            with open(TICKS_FILE) as h:
                block_nums = json.load(h)

        w3 = arbitrum_w3()

        minter = load_contract_from_json(w3, MINTER_DEPLOYMENT_JSON)
        bonding = bonding_manager(w3)

        callables = {
            "inflation": minter.functions.inflation(),
            "total-supply": minter.functions.getGlobalTotalSupply(),
            "bonded": bonding.functions.getTotalBonded()
        }

        print("Fetching historic data...")
        results = {k: fetch_historic(v, block_nums["block"]) for k, v in callables.items()}
        results = results | block_nums

        with open(DATA_FILE, 'w') as h:
            json.dump(results, h)

        print(json.dumps(results))
