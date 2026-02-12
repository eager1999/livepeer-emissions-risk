"""Practical examples of using the time conversion library.

This file demonstrates common use cases for the lpt_stake.time module.
Run this file to see the examples in action:

    cd python
    uv run python examples/time_conversion_examples.py
"""

from datetime import datetime, timedelta, timezone

from lpt_stake.time import (
    SECONDS_PER_ROUND,
    ETH_BLOCKS_PER_ROUND,
    timedelta_to_rounds,
    rounds_to_timedelta,
    blocks_to_rounds,
    rounds_to_blocks,
    datetime_to_round,
    round_to_datetime,
    eth_block_to_datetime,
    datetime_to_eth_block,
    round_duration,
)


def example_1_basic_duration_conversions():
    """Example 1: Converting between time durations and rounds."""
    print("=" * 70)
    print("Example 1: Basic Duration Conversions")
    print("=" * 70)

    # How many rounds in common time periods?
    day = timedelta(days=1)
    week = timedelta(weeks=1)
    month = timedelta(days=30)

    print(f"\n1 day = {timedelta_to_rounds(day):.2f} rounds")
    print(f"1 week = {timedelta_to_rounds(week):.2f} rounds")
    print(f"30 days = {timedelta_to_rounds(month):.2f} rounds")

    # How long are rounds in different units?
    one_round = rounds_to_timedelta(1.0)
    print(f"\n1 round = {one_round.total_seconds() / 3600:.1f} hours")
    print(f"1 round = {one_round.total_seconds() / 86400:.3f} days")

    ten_rounds = rounds_to_timedelta(10.0)
    print(f"\n10 rounds = {ten_rounds.days} days, {ten_rounds.seconds // 3600} hours")

    # Round duration utility
    duration = round_duration()
    print(f"\nRound duration: {duration.total_seconds()} seconds ({duration.total_seconds() / 3600:.1f} hours)")


def example_2_block_conversions():
    """Example 2: Working with Ethereum blocks and rounds."""
    print("\n" + "=" * 70)
    print("Example 2: Block and Round Conversions")
    print("=" * 70)

    # Basic block conversions
    print(f"\n{ETH_BLOCKS_PER_ROUND} blocks = {blocks_to_rounds(ETH_BLOCKS_PER_ROUND):.1f} round")
    print(f"11,520 blocks = {blocks_to_rounds(11520):.1f} rounds")
    print(f"100,000 blocks = {blocks_to_rounds(100000):.2f} rounds")

    # Converting rounds to blocks
    print(f"\n1.5 rounds = {rounds_to_blocks(1.5)} blocks")
    print(f"10 rounds = {rounds_to_blocks(10)} blocks")

    # Analyzing a block range
    start_block = 20000000
    end_block = 20050000
    block_range = end_block - start_block
    print(f"\nBlocks {start_block} to {end_block}:")
    print(f"  Range: {block_range} blocks")
    print(f"  Rounds: {blocks_to_rounds(block_range):.2f}")


def example_3_absolute_time_conversions():
    """Example 3: Converting between calendar dates and rounds."""
    print("\n" + "=" * 70)
    print("Example 3: Calendar Dates and Rounds")
    print("=" * 70)

    # Find round number for specific dates
    dates_to_check = [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 6, 15, tzinfo=timezone.utc),
        datetime(2025, 1, 1, tzinfo=timezone.utc),
    ]

    print("\nRound numbers for specific dates:")
    for dt in dates_to_check:
        round_num = datetime_to_round(dt)
        print(f"  {dt.date()}: Round {round_num}")

    # Convert rounds to dates
    rounds_to_check = [1000, 2500, 5000]

    print("\nDates for specific rounds:")
    for round_num in rounds_to_check:
        dt = round_to_datetime(round_num)
        print(f"  Round {round_num}: {dt.date()} at {dt.strftime('%H:%M:%S')} UTC")


def example_4_round_boundaries():
    """Example 4: Finding round start and end times."""
    print("\n" + "=" * 70)
    print("Example 4: Round Boundaries")
    print("=" * 70)

    # Check which round contains a specific date
    target_date = datetime(2024, 7, 4, 15, 30, 0, tzinfo=timezone.utc)
    round_num = datetime_to_round(target_date)

    round_start = round_to_datetime(round_num)
    round_end = round_to_datetime(round_num + 1)

    print(f"\nTarget date: {target_date.isoformat()}")
    print(f"Falls in Round: {round_num}")
    print(f"Round start: {round_start.isoformat()}")
    print(f"Round end:   {round_end.isoformat()}")
    print(f"Round duration: {(round_end - round_start).total_seconds() / 3600:.1f} hours")


def example_5_time_series_analysis():
    """Example 5: Analyzing time series data by rounds."""
    print("\n" + "=" * 70)
    print("Example 5: Time Series Analysis")
    print("=" * 70)

    # Simulate analyzing data over a date range
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

    start_round = datetime_to_round(start_date)
    end_round = datetime_to_round(end_date)
    total_rounds = end_round - start_round + 1

    print(f"\nAnalyzing data from {start_date.date()} to {end_date.date()}:")
    print(f"  Start round: {start_round}")
    print(f"  End round: {end_round}")
    print(f"  Total rounds: {total_rounds}")

    # Calculate expected duration
    duration = end_date - start_date
    expected_rounds = timedelta_to_rounds(duration)
    print(f"  Duration: {duration.days} days")
    print(f"  Expected rounds (from duration): {expected_rounds:.2f}")


def example_6_working_with_blocks():
    """Example 6: Ethereum block number conversions."""
    print("\n" + "=" * 70)
    print("Example 6: Ethereum Block Conversions")
    print("=" * 70)

    # Convert block numbers to approximate timestamps
    blocks_to_check = [20000000, 21000000, 22000000]

    print("\nApproximate timestamps for Ethereum blocks:")
    for block in blocks_to_check:
        dt = eth_block_to_datetime(block)
        print(f"  Block {block}: {dt.date()} at {dt.strftime('%H:%M:%S')} UTC")

    # Convert dates to approximate block numbers
    dates = [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 1, tzinfo=timezone.utc),
    ]

    print("\nApproximate Ethereum blocks for dates:")
    for dt in dates:
        block = datetime_to_eth_block(dt)
        print(f"  {dt.date()}: Block ~{block}")


def example_7_simulation_parameters():
    """Example 7: Setting simulation parameters."""
    print("\n" + "=" * 70)
    print("Example 7: Simulation Parameters")
    print("=" * 70)

    # Set up a simulation timeframe
    simulation_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    simulation_duration_days = 90  # 3 months

    simulation_duration = timedelta(days=simulation_duration_days)
    simulation_rounds = timedelta_to_rounds(simulation_duration)

    print(f"\nSimulation setup:")
    print(f"  Start date: {simulation_start.date()}")
    print(f"  Duration: {simulation_duration_days} days")
    print(f"  Number of rounds: {simulation_rounds:.1f}")
    print(f"  Rounded to: {round(simulation_rounds)} rounds")

    # Calculate end date
    actual_rounds = round(simulation_rounds)
    actual_duration = rounds_to_timedelta(actual_rounds)
    simulation_end = simulation_start + actual_duration

    print(f"\nActual simulation parameters:")
    print(f"  Rounds: {actual_rounds}")
    print(f"  Duration: {actual_duration.days} days, {actual_duration.seconds // 3600} hours")
    print(f"  End date: {simulation_end.date()} at {simulation_end.strftime('%H:%M:%S')} UTC")


def example_8_current_round():
    """Example 8: Finding the current round."""
    print("\n" + "=" * 70)
    print("Example 8: Current Round")
    print("=" * 70)

    # Get current time and round
    now = datetime.now(timezone.utc)
    current_round = datetime_to_round(now)

    round_start = round_to_datetime(current_round)
    round_end = round_to_datetime(current_round + 1)
    time_elapsed = now - round_start
    time_remaining = round_end - now

    print(f"\nCurrent time: {now.isoformat()}")
    print(f"Current round: {current_round}")
    print(f"Round started: {round_start.isoformat()}")
    print(f"Round ends: {round_end.isoformat()}")
    print(f"Time elapsed in round: {time_elapsed.total_seconds() / 3600:.1f} hours")
    print(f"Time remaining in round: {time_remaining.total_seconds() / 3600:.1f} hours")

    # Progress percentage
    round_progress = time_elapsed.total_seconds() / SECONDS_PER_ROUND * 100
    print(f"Round progress: {round_progress:.1f}%")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("TIME CONVERSION LIBRARY - PRACTICAL EXAMPLES")
    print("=" * 70)

    example_1_basic_duration_conversions()
    example_2_block_conversions()
    example_3_absolute_time_conversions()
    example_4_round_boundaries()
    example_5_time_series_analysis()
    example_6_working_with_blocks()
    example_7_simulation_parameters()
    example_8_current_round()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
