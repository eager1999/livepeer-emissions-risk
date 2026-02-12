"""lpt-stake: libraries for Livepeer emissions risk analysis."""

from lpt_stake.time import (
    # Constants
    SECONDS_PER_ETH_BLOCK,
    ETH_BLOCKS_PER_ROUND,
    SECONDS_PER_ROUND,
    REFERENCE_ETH_BLOCK,
    REFERENCE_DATETIME,
    # Time duration conversions
    timedelta_to_rounds,
    rounds_to_timedelta,
    # Block conversions
    blocks_to_rounds,
    rounds_to_blocks,
    # Ethereum block <-> datetime conversions
    eth_block_to_datetime,
    datetime_to_eth_block,
    # Round <-> datetime conversions
    datetime_to_round,
    round_to_datetime,
    # Utility functions
    round_duration,
)

__all__ = [
    # Constants
    "SECONDS_PER_ETH_BLOCK",
    "ETH_BLOCKS_PER_ROUND",
    "SECONDS_PER_ROUND",
    "REFERENCE_ETH_BLOCK",
    "REFERENCE_DATETIME",
    # Time duration conversions
    "timedelta_to_rounds",
    "rounds_to_timedelta",
    # Block conversions
    "blocks_to_rounds",
    "rounds_to_blocks",
    # Ethereum block <-> datetime conversions
    "eth_block_to_datetime",
    "datetime_to_eth_block",
    # Round <-> datetime conversions
    "datetime_to_round",
    "round_to_datetime",
    # Utility functions
    "round_duration",
]
