# Time Conversion Library

This library provides utilities for converting between natural time units (datetime, timedelta) and Livepeer protocol rounds.

## Background

The Livepeer protocol uses "rounds" as its fundamental time unit for state transitions and reward distribution. Understanding the relationship between rounds and calendar time is essential for:

- Analyzing historical protocol data
- Simulating future protocol states
- Setting time-based parameters in models
- Interpreting time series data

## Key Concepts

### Livepeer Rounds

A **round** in Livepeer is a fixed period defined by Ethereum block numbers:
- **1 round = 5,760 Ethereum blocks**
- At ~12 seconds per block, this is approximately **19.2 hours** or **0.8 days**

### Time Conversion Approach

This library uses **approach #2** from [issue #13](https://github.com/shtukaresearch/livepeer-emissions-risk/issues/13):

1. **Hardcoded reference point**: Ethereum block 14185006 (RoundsManager deployment on Arbitrum)
   - Timestamp: 2022-02-17 04:12:56 UTC
2. **Fixed rate**: 12 seconds per Ethereum block
3. **Simple extrapolation**: No RPC calls required

**Trade-offs**:
- ✅ Simple and fast
- ✅ No external dependencies
- ✅ Suitable for analysis and simulation
- ⚠️ Will drift ~1% from actual times (Ethereum blocks average slightly >12s)
- ⚠️ Drift increases with distance from reference point

For most analysis purposes, this drift is acceptable. A more accurate RPC-based approach can be added later if needed.

## Installation

The time conversion library is part of the `lpt-stake` package:

```bash
cd python
uv sync
```

## Usage

### Import Functions

```python
from lpt_stake import (
    # Duration conversions
    timedelta_to_rounds,
    rounds_to_timedelta,

    # Block conversions
    blocks_to_rounds,
    rounds_to_blocks,

    # Absolute time conversions
    datetime_to_round,
    round_to_datetime,
    eth_block_to_datetime,
    datetime_to_eth_block,

    # Constants
    SECONDS_PER_ETH_BLOCK,
    ETH_BLOCKS_PER_ROUND,
    SECONDS_PER_ROUND,
)
```

### Basic Examples

#### Convert Time Durations to Rounds

```python
from datetime import timedelta
from lpt_stake import timedelta_to_rounds

# How many rounds in a week?
week = timedelta(weeks=1)
rounds = timedelta_to_rounds(week)
print(f"1 week = {rounds:.2f} rounds")  # 1 week = 8.75 rounds

# How many rounds in 30 days?
month = timedelta(days=30)
rounds = timedelta_to_rounds(month)
print(f"30 days = {rounds:.2f} rounds")  # 30 days = 37.50 rounds
```

#### Convert Rounds to Time Durations

```python
from lpt_stake import rounds_to_timedelta

# How long is 10 rounds?
duration = rounds_to_timedelta(10)
print(f"10 rounds = {duration.days} days, {duration.seconds // 3600} hours")

# How long is 1 round?
one_round = rounds_to_timedelta(1)
print(f"1 round = {one_round.total_seconds() / 3600:.1f} hours")  # 19.2 hours
```

#### Convert Calendar Dates to Rounds

```python
from datetime import datetime, timezone
from lpt_stake import datetime_to_round

# What round was January 1, 2024?
date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
round_num = datetime_to_round(date)
print(f"January 1, 2024 was round {round_num}")

# Note: Always use timezone-aware datetimes!
```

#### Convert Rounds to Calendar Dates

```python
from lpt_stake import round_to_datetime

# When did round 1000 start?
start_time = round_to_datetime(1000)
print(f"Round 1000 started at {start_time.isoformat()}")

# When will round 5000 start?
future_time = round_to_datetime(5000)
print(f"Round 5000 will start around {future_time.date()}")
```

#### Working with Ethereum Blocks

```python
from lpt_stake import (
    eth_block_to_datetime,
    datetime_to_eth_block,
    blocks_to_rounds,
)

# When did block 20000000 occur?
block_time = eth_block_to_datetime(20000000)
print(f"Block 20000000: {block_time.isoformat()}")

# What block number corresponds to a date?
from datetime import timezone
date = datetime(2025, 1, 1, tzinfo=timezone.utc)
block = datetime_to_eth_block(date)
print(f"Block at 2025-01-01: ~{block}")

# How many rounds in a block range?
block_range = 50000
rounds = blocks_to_rounds(block_range)
print(f"{block_range} blocks = {rounds:.2f} rounds")
```

## API Reference

### Constants

- `SECONDS_PER_ETH_BLOCK = 12` - Average time per Ethereum block
- `ETH_BLOCKS_PER_ROUND = 5760` - Ethereum blocks per Livepeer round
- `SECONDS_PER_ROUND = 69120` - Seconds per Livepeer round (19.2 hours)
- `REFERENCE_ETH_BLOCK = 14185006` - Reference Ethereum block for conversions
- `REFERENCE_DATETIME` - Datetime of reference block (2022-02-17 04:12:56 UTC)

### Duration Conversion Functions

#### `timedelta_to_rounds(duration: timedelta) -> float`

Convert a time duration to Livepeer rounds.

**Parameters:**
- `duration`: Time duration to convert

**Returns:**
- Number of Livepeer rounds (can be fractional)

**Example:**
```python
timedelta_to_rounds(timedelta(days=1))  # 1.25 rounds
```

#### `rounds_to_timedelta(rounds: float) -> timedelta`

Convert Livepeer rounds to a time duration.

**Parameters:**
- `rounds`: Number of Livepeer rounds (can be fractional)

**Returns:**
- Time duration

**Example:**
```python
rounds_to_timedelta(1.0)  # timedelta(seconds=69120)
```

### Block Conversion Functions

#### `blocks_to_rounds(blocks: int) -> float`

Convert Ethereum block count to Livepeer rounds.

**Parameters:**
- `blocks`: Number of Ethereum blocks

**Returns:**
- Number of Livepeer rounds (can be fractional)

**Example:**
```python
blocks_to_rounds(5760)  # 1.0 round
```

#### `rounds_to_blocks(rounds: float) -> int`

Convert Livepeer rounds to Ethereum block count.

**Parameters:**
- `rounds`: Number of Livepeer rounds (can be fractional)

**Returns:**
- Number of Ethereum blocks (rounded to nearest integer)

**Example:**
```python
rounds_to_blocks(2.5)  # 14400 blocks
```

### Ethereum Block ↔ Datetime Functions

#### `eth_block_to_datetime(block_number: int) -> datetime`

Convert an Ethereum block number to approximate datetime.

**Parameters:**
- `block_number`: Ethereum block number

**Returns:**
- Approximate datetime (UTC timezone)

**Note:** Uses reference point and 12s/block. Drift increases with distance from reference.

#### `datetime_to_eth_block(dt: datetime) -> int`

Convert a datetime to approximate Ethereum block number.

**Parameters:**
- `dt`: Datetime to convert (must be timezone-aware)

**Returns:**
- Approximate Ethereum block number

**Raises:**
- `ValueError`: If datetime is naive (has no timezone info)

**Note:** Always use timezone-aware datetimes! Use `datetime(..., tzinfo=timezone.utc)`.

### Round ↔ Datetime Functions

#### `datetime_to_round(dt: datetime, round_zero_block: int = REFERENCE_ETH_BLOCK) -> int`

Convert a datetime to approximate Livepeer round number.

**Parameters:**
- `dt`: Datetime to convert (must be timezone-aware)
- `round_zero_block`: Ethereum block number to use as round zero (optional)

**Returns:**
- Approximate Livepeer round number

**Raises:**
- `ValueError`: If datetime is naive

#### `round_to_datetime(round_number: int, round_zero_block: int = REFERENCE_ETH_BLOCK) -> datetime`

Convert a Livepeer round number to approximate datetime.

Returns the datetime at the **start** of the specified round.

**Parameters:**
- `round_number`: Livepeer round number
- `round_zero_block`: Ethereum block number to use as round zero (optional)

**Returns:**
- Approximate datetime at the start of the round (UTC timezone)

### Utility Functions

#### `round_duration() -> timedelta`

Get the duration of one Livepeer round.

**Returns:**
- Duration of one round (19.2 hours or 69120 seconds)

## Common Patterns

### Finding Current Round

```python
from datetime import datetime, timezone
from lpt_stake import datetime_to_round

now = datetime.now(timezone.utc)
current_round = datetime_to_round(now)
print(f"Current round: {current_round}")
```

### Calculating Round Boundaries

```python
from lpt_stake import round_to_datetime

round_num = 3000
start = round_to_datetime(round_num)
end = round_to_datetime(round_num + 1)
print(f"Round {round_num}: {start.isoformat()} to {end.isoformat()}")
```

### Working with Custom Reference Points

If you want to use a different "round zero" (e.g., first round of a specific protocol upgrade):

```python
from lpt_stake import datetime_to_round, round_to_datetime

# Use a custom reference block as round zero
UPGRADE_BLOCK = 15000000

# Find rounds relative to upgrade
current_time = datetime.now(timezone.utc)
rounds_since_upgrade = datetime_to_round(current_time, round_zero_block=UPGRADE_BLOCK)

# Convert back
round_time = round_to_datetime(100, round_zero_block=UPGRADE_BLOCK)
```

## Testing

Run the test suite:

```bash
cd python
uv sync --dev  # Install dev dependencies including pytest
uv run pytest test_time.py -v
```

The test suite includes:
- Unit tests for all conversion functions
- Roundtrip tests (ensuring A→B→A conversions are consistent)
- Integration tests for realistic scenarios
- Edge case handling

## Accuracy and Limitations

### Expected Drift

The library assumes 12 seconds per Ethereum block. Actual Ethereum block times average slightly more than 12 seconds, so this approximation will drift over time:

- **Short term** (days to weeks): Negligible drift (<0.1%)
- **Medium term** (months): ~1% drift
- **Long term** (years): Increasing drift, may reach several percent

For most analysis and simulation purposes, this is acceptable.

### When to Use

✅ **Good for:**
- Historical data analysis
- Protocol simulations
- Setting relative time parameters
- Rough estimates of future dates
- Understanding order of magnitude

❌ **Not suitable for:**
- Precise timing of critical events
- Smart contract integration
- Applications requiring exact block numbers

### Future Improvements

If greater accuracy is needed, the library could be extended to:
1. Use RPC endpoints to fetch actual block timestamps
2. Build a mapping cache between blocks and timestamps
3. Interpolate for better accuracy

This would be **approach #1** from the original issue specification.

## References

- [Issue #13: Enhancement: add library for converting between natural time units and Livepeer rounds](https://github.com/shtukaresearch/livepeer-emissions-risk/issues/13)
- [RoundsManager Contract on Arbiscan](https://arbiscan.io/address/0xdd6f56DcC28D3F5f27084381fE8Df634985cc39f#readProxyContract)
- [Arbitrum block.number documentation](https://docs.arbitrum.io/build-decentralized-apps/arbitrum-vs-ethereum/block-numbers-and-time)
- [Ethereum block times after The Merge](https://forum.livepeer.org/t/increase-blocks-per-round-after-the-merge/1864)
