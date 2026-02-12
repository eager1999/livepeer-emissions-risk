"""Tests for time conversion utilities."""

import pytest
from datetime import datetime, timedelta, timezone

from src.lpt_stake.time import (
    SECONDS_PER_ETH_BLOCK,
    ETH_BLOCKS_PER_ROUND,
    SECONDS_PER_ROUND,
    REFERENCE_ETH_BLOCK,
    REFERENCE_DATETIME,
    timedelta_to_rounds,
    rounds_to_timedelta,
    blocks_to_rounds,
    rounds_to_blocks,
    eth_block_to_datetime,
    datetime_to_eth_block,
    datetime_to_round,
    round_to_datetime,
    round_duration,
)


class TestConstants:
    """Test that constants have expected values."""

    def test_seconds_per_eth_block(self):
        assert SECONDS_PER_ETH_BLOCK == 12

    def test_eth_blocks_per_round(self):
        assert ETH_BLOCKS_PER_ROUND == 5760

    def test_seconds_per_round(self):
        assert SECONDS_PER_ROUND == 69120
        assert SECONDS_PER_ROUND == SECONDS_PER_ETH_BLOCK * ETH_BLOCKS_PER_ROUND

    def test_reference_constants(self):
        assert REFERENCE_ETH_BLOCK == 14185006
        assert REFERENCE_DATETIME == datetime(2022, 2, 17, 4, 12, 56, tzinfo=timezone.utc)


class TestTimedeltaConversions:
    """Test conversions between timedelta and rounds."""

    def test_timedelta_to_rounds_one_round(self):
        """One round is 69120 seconds (19.2 hours)."""
        duration = timedelta(seconds=SECONDS_PER_ROUND)
        assert timedelta_to_rounds(duration) == 1.0

    def test_timedelta_to_rounds_one_day(self):
        """One day is 1.25 rounds."""
        duration = timedelta(days=1)
        rounds = timedelta_to_rounds(duration)
        assert abs(rounds - 1.25) < 0.001

    def test_timedelta_to_rounds_one_week(self):
        """One week is 8.75 rounds."""
        duration = timedelta(weeks=1)
        rounds = timedelta_to_rounds(duration)
        assert abs(rounds - 8.75) < 0.001

    def test_rounds_to_timedelta_one_round(self):
        """One round converts to 69120 seconds."""
        duration = rounds_to_timedelta(1.0)
        assert duration == timedelta(seconds=SECONDS_PER_ROUND)

    def test_rounds_to_timedelta_fractional(self):
        """Fractional rounds work correctly."""
        duration = rounds_to_timedelta(0.5)
        assert duration == timedelta(seconds=SECONDS_PER_ROUND / 2)

    def test_rounds_to_timedelta_seven_rounds(self):
        """Seven rounds is 5 days plus some hours."""
        duration = rounds_to_timedelta(7.0)
        expected_seconds = 7 * SECONDS_PER_ROUND
        assert duration == timedelta(seconds=expected_seconds)

    def test_timedelta_roundtrip(self):
        """Converting timedelta -> rounds -> timedelta should be identity."""
        original = timedelta(days=3, hours=7, minutes=23)
        rounds = timedelta_to_rounds(original)
        recovered = rounds_to_timedelta(rounds)
        # Allow small floating point error
        assert abs((recovered - original).total_seconds()) < 0.001


class TestBlockConversions:
    """Test conversions between blocks and rounds."""

    def test_blocks_to_rounds_exact(self):
        """5760 blocks is exactly 1 round."""
        assert blocks_to_rounds(5760) == 1.0

    def test_blocks_to_rounds_multiple(self):
        """11520 blocks is exactly 2 rounds."""
        assert blocks_to_rounds(11520) == 2.0

    def test_blocks_to_rounds_fractional(self):
        """2880 blocks is 0.5 rounds."""
        assert blocks_to_rounds(2880) == 0.5

    def test_rounds_to_blocks_exact(self):
        """1 round is 5760 blocks."""
        assert rounds_to_blocks(1.0) == 5760

    def test_rounds_to_blocks_multiple(self):
        """2.5 rounds is 14400 blocks."""
        assert rounds_to_blocks(2.5) == 14400

    def test_rounds_to_blocks_rounding(self):
        """Fractional blocks are rounded."""
        # 0.5001 rounds should round to nearest integer
        blocks = rounds_to_blocks(0.5001)
        assert isinstance(blocks, int)
        assert blocks == 2881  # rounds to nearest

    def test_blocks_roundtrip(self):
        """Converting blocks -> rounds -> blocks should be identity for multiples."""
        original_blocks = 23040  # 4 rounds exactly
        rounds = blocks_to_rounds(original_blocks)
        recovered_blocks = rounds_to_blocks(rounds)
        assert recovered_blocks == original_blocks


class TestEthBlockDatetimeConversions:
    """Test conversions between Ethereum blocks and datetime."""

    def test_eth_block_to_datetime_reference(self):
        """Reference block should return reference datetime."""
        dt = eth_block_to_datetime(REFERENCE_ETH_BLOCK)
        assert dt == REFERENCE_DATETIME

    def test_eth_block_to_datetime_after_reference(self):
        """Block after reference should be later in time."""
        # 5760 blocks later (1 round)
        dt = eth_block_to_datetime(REFERENCE_ETH_BLOCK + 5760)
        expected = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        assert dt == expected

    def test_eth_block_to_datetime_before_reference(self):
        """Block before reference should be earlier in time."""
        # 5760 blocks earlier (1 round)
        dt = eth_block_to_datetime(REFERENCE_ETH_BLOCK - 5760)
        expected = REFERENCE_DATETIME - timedelta(seconds=SECONDS_PER_ROUND)
        assert dt == expected

    def test_datetime_to_eth_block_reference(self):
        """Reference datetime should return reference block."""
        block = datetime_to_eth_block(REFERENCE_DATETIME)
        assert block == REFERENCE_ETH_BLOCK

    def test_datetime_to_eth_block_after_reference(self):
        """Datetime after reference should return higher block."""
        dt = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        block = datetime_to_eth_block(dt)
        assert block == REFERENCE_ETH_BLOCK + 5760

    def test_datetime_to_eth_block_before_reference(self):
        """Datetime before reference should return lower block."""
        dt = REFERENCE_DATETIME - timedelta(seconds=SECONDS_PER_ROUND)
        block = datetime_to_eth_block(dt)
        assert block == REFERENCE_ETH_BLOCK - 5760

    def test_datetime_to_eth_block_naive_raises(self):
        """Naive datetime (no timezone) should raise ValueError."""
        naive_dt = datetime(2022, 1, 1, 12, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            datetime_to_eth_block(naive_dt)

    def test_eth_block_datetime_roundtrip(self):
        """Converting block -> datetime -> block should be identity."""
        original_block = 15000000
        dt = eth_block_to_datetime(original_block)
        recovered_block = datetime_to_eth_block(dt)
        assert recovered_block == original_block

    def test_datetime_eth_block_roundtrip(self):
        """Converting datetime -> block -> datetime should be close."""
        original_dt = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        block = datetime_to_eth_block(original_dt)
        recovered_dt = eth_block_to_datetime(block)
        # Should be within 12 seconds (1 block) due to rounding
        diff = abs((recovered_dt - original_dt).total_seconds())
        assert diff < SECONDS_PER_ETH_BLOCK


class TestRoundDatetimeConversions:
    """Test conversions between rounds and datetime."""

    def test_datetime_to_round_reference(self):
        """Reference datetime should be round 0."""
        round_num = datetime_to_round(REFERENCE_DATETIME)
        assert round_num == 0

    def test_datetime_to_round_one_round_later(self):
        """One round after reference should be round 1."""
        dt = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        round_num = datetime_to_round(dt)
        assert round_num == 1

    def test_datetime_to_round_partial_round(self):
        """Partial round should round down."""
        # Half a round later should still be round 0
        dt = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND / 2)
        round_num = datetime_to_round(dt)
        assert round_num == 0

    def test_datetime_to_round_custom_zero(self):
        """Can specify custom round zero block."""
        custom_zero = REFERENCE_ETH_BLOCK + 5760  # 1 round later
        dt = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        round_num = datetime_to_round(dt, round_zero_block=custom_zero)
        assert round_num == 0

    def test_round_to_datetime_zero(self):
        """Round 0 should return reference datetime."""
        dt = round_to_datetime(0)
        assert dt == REFERENCE_DATETIME

    def test_round_to_datetime_one(self):
        """Round 1 should be one round after reference."""
        dt = round_to_datetime(1)
        expected = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        assert dt == expected

    def test_round_to_datetime_negative(self):
        """Negative rounds should work (go back in time)."""
        dt = round_to_datetime(-1)
        expected = REFERENCE_DATETIME - timedelta(seconds=SECONDS_PER_ROUND)
        assert dt == expected

    def test_round_to_datetime_custom_zero(self):
        """Can specify custom round zero block."""
        custom_zero = REFERENCE_ETH_BLOCK + 5760  # 1 round later
        dt = round_to_datetime(0, round_zero_block=custom_zero)
        expected = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        assert dt == expected

    def test_round_datetime_roundtrip(self):
        """Converting round -> datetime -> round should be identity."""
        original_round = 100
        dt = round_to_datetime(original_round)
        recovered_round = datetime_to_round(dt)
        assert recovered_round == original_round


class TestRoundDuration:
    """Test round_duration utility function."""

    def test_round_duration_value(self):
        """round_duration() should return expected timedelta."""
        duration = round_duration()
        assert duration == timedelta(seconds=SECONDS_PER_ROUND)

    def test_round_duration_hours(self):
        """One round is 19.2 hours."""
        duration = round_duration()
        hours = duration.total_seconds() / 3600
        assert abs(hours - 19.2) < 0.001


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    def test_calculate_rounds_in_week(self):
        """Calculate how many rounds occur in a week."""
        week = timedelta(weeks=1)
        rounds = timedelta_to_rounds(week)
        assert abs(rounds - 8.75) < 0.001

    def test_find_round_for_specific_date(self):
        """Find which round a specific date falls in."""
        # Some date in 2024
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        round_num = datetime_to_round(dt)
        # Should be a positive round number (after reference)
        assert round_num > 0
        # Verify roundtrip
        round_start = round_to_datetime(round_num)
        round_end = round_to_datetime(round_num + 1)
        assert round_start <= dt < round_end

    def test_convert_block_range_to_rounds(self):
        """Convert a block range to equivalent rounds."""
        start_block = 20000000
        end_block = 20050000
        block_count = end_block - start_block
        rounds = blocks_to_rounds(block_count)
        # Should be about 8.68 rounds
        assert 8.6 < rounds < 8.7

    def test_time_range_to_rounds(self):
        """Convert a time range to rounds."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 8, tzinfo=timezone.utc)  # 7 days later
        duration = end - start
        rounds = timedelta_to_rounds(duration)
        # 7 days = 8.75 rounds
        assert abs(rounds - 8.75) < 0.001
