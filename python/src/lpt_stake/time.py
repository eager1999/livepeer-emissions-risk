"""Time conversion utilities for Livepeer rounds and natural time units.

This module provides utilities for converting between:
- Natural time units (datetime, timedelta)
- Livepeer rounds
- Ethereum block numbers

The conversion uses a hardcoded reference point (Ethereum block 14185006,
corresponding to the deployment of the RoundsManager contract on Arbitrum)
and assumes 12 seconds per Ethereum block for extrapolation.

Constants:
    SECONDS_PER_ETH_BLOCK: Average time per Ethereum block (12 seconds)
    ETH_BLOCKS_PER_ROUND: Number of Ethereum blocks per Livepeer round (5760)
    SECONDS_PER_ROUND: Average time per Livepeer round (69120 seconds)
    REFERENCE_ETH_BLOCK: Reference Ethereum block for absolute time conversions
    REFERENCE_DATETIME: Datetime corresponding to REFERENCE_ETH_BLOCK

Note:
    This implementation uses approach #2 from the issue specification:
    hardcoded reference point with extrapolation. This is simpler than
    using RPC endpoints but will experience drift (~1%) because actual
    Ethereum block times average slightly more than 12 seconds.
"""

from datetime import datetime, timedelta, timezone

# Constants for time conversion
SECONDS_PER_ETH_BLOCK = 12  # Average time per Ethereum block
ETH_BLOCKS_PER_ROUND = 5760  # Number of Ethereum blocks per Livepeer round
SECONDS_PER_ROUND = SECONDS_PER_ETH_BLOCK * ETH_BLOCKS_PER_ROUND  # 69120 seconds

# Reference point for absolute time conversions
# Ethereum block 14185006 corresponds to the deployment block of RoundsManager on Arbitrum
# Block timestamp: 2022-02-17 04:12:56 UTC
REFERENCE_ETH_BLOCK = 14185006
REFERENCE_DATETIME = datetime(2022, 2, 17, 4, 12, 56, tzinfo=timezone.utc)


def timedelta_to_rounds(duration: timedelta) -> float:
    """Convert a time duration to Livepeer rounds.

    Args:
        duration: Time duration to convert

    Returns:
        Number of Livepeer rounds (can be fractional)

    Examples:
        >>> from datetime import timedelta
        >>> timedelta_to_rounds(timedelta(days=1))
        1.25
        >>> timedelta_to_rounds(timedelta(hours=19.2))
        1.0
    """
    seconds = duration.total_seconds()
    return seconds / SECONDS_PER_ROUND


def rounds_to_timedelta(rounds: float) -> timedelta:
    """Convert Livepeer rounds to a time duration.

    Args:
        rounds: Number of Livepeer rounds (can be fractional)

    Returns:
        Time duration

    Examples:
        >>> rounds_to_timedelta(1.0)
        timedelta(seconds=69120)
        >>> rounds_to_timedelta(7.0)
        timedelta(days=5, seconds=52320)
    """
    seconds = rounds * SECONDS_PER_ROUND
    return timedelta(seconds=seconds)


def blocks_to_rounds(blocks: int) -> float:
    """Convert Ethereum block count to Livepeer rounds.

    Args:
        blocks: Number of Ethereum blocks

    Returns:
        Number of Livepeer rounds (can be fractional)

    Examples:
        >>> blocks_to_rounds(5760)
        1.0
        >>> blocks_to_rounds(11520)
        2.0
    """
    return blocks / ETH_BLOCKS_PER_ROUND


def rounds_to_blocks(rounds: float) -> int:
    """Convert Livepeer rounds to Ethereum block count.

    Args:
        rounds: Number of Livepeer rounds (can be fractional)

    Returns:
        Number of Ethereum blocks (rounded to nearest integer)

    Examples:
        >>> rounds_to_blocks(1.0)
        5760
        >>> rounds_to_blocks(2.5)
        14400
    """
    return round(rounds * ETH_BLOCKS_PER_ROUND)


def eth_block_to_datetime(block_number: int) -> datetime:
    """Convert an Ethereum block number to approximate datetime.

    Uses the reference point (block 14185006) and assumes 12 seconds per block
    for extrapolation. Will have drift for blocks far from the reference point.

    Args:
        block_number: Ethereum block number

    Returns:
        Approximate datetime (UTC timezone)

    Examples:
        >>> eth_block_to_datetime(14185006)
        datetime.datetime(2022, 2, 17, 4, 12, 56, tzinfo=datetime.timezone.utc)
    """
    block_diff = block_number - REFERENCE_ETH_BLOCK
    time_diff = timedelta(seconds=block_diff * SECONDS_PER_ETH_BLOCK)
    return REFERENCE_DATETIME + time_diff


def datetime_to_eth_block(dt: datetime) -> int:
    """Convert a datetime to approximate Ethereum block number.

    Uses the reference point (block 14185006) and assumes 12 seconds per block
    for extrapolation. Will have drift for times far from the reference point.

    Args:
        dt: Datetime to convert (should have timezone info)

    Returns:
        Approximate Ethereum block number

    Raises:
        ValueError: If datetime is naive (has no timezone info)

    Examples:
        >>> from datetime import timezone
        >>> datetime_to_eth_block(datetime(2022, 2, 17, 4, 12, 56, tzinfo=timezone.utc))
        14185006
    """
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")

    time_diff = dt - REFERENCE_DATETIME
    block_diff = round(time_diff.total_seconds() / SECONDS_PER_ETH_BLOCK)
    return REFERENCE_ETH_BLOCK + block_diff


def datetime_to_round(dt: datetime, round_zero_block: int = REFERENCE_ETH_BLOCK) -> int:
    """Convert a datetime to approximate Livepeer round number.

    Args:
        dt: Datetime to convert (should have timezone info)
        round_zero_block: Ethereum block number to use as round zero
                         (defaults to REFERENCE_ETH_BLOCK)

    Returns:
        Approximate Livepeer round number

    Raises:
        ValueError: If datetime is naive (has no timezone info)

    Examples:
        >>> from datetime import timezone
        >>> datetime_to_round(datetime(2022, 2, 17, 4, 12, 56, tzinfo=timezone.utc))
        0
    """
    block_number = datetime_to_eth_block(dt)
    blocks_since_zero = block_number - round_zero_block
    return blocks_since_zero // ETH_BLOCKS_PER_ROUND


def round_to_datetime(
    round_number: int,
    round_zero_block: int = REFERENCE_ETH_BLOCK
) -> datetime:
    """Convert a Livepeer round number to approximate datetime.

    Returns the datetime at the start of the specified round.

    Args:
        round_number: Livepeer round number
        round_zero_block: Ethereum block number to use as round zero
                         (defaults to REFERENCE_ETH_BLOCK)

    Returns:
        Approximate datetime at the start of the round (UTC timezone)

    Examples:
        >>> round_to_datetime(0)
        datetime.datetime(2022, 2, 17, 4, 12, 56, tzinfo=datetime.timezone.utc)
    """
    block_number = round_zero_block + (round_number * ETH_BLOCKS_PER_ROUND)
    return eth_block_to_datetime(block_number)


def round_duration() -> timedelta:
    """Get the duration of one Livepeer round.

    Returns:
        Duration of one round (19.2 hours or 69120 seconds)

    Examples:
        >>> round_duration()
        timedelta(seconds=69120)
    """
    return timedelta(seconds=SECONDS_PER_ROUND)
