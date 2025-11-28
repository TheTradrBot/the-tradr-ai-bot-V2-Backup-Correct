"""
Forex Holiday Calendar for Blueprint Trader AI.

This module provides a list of major forex market holidays
when trading should be avoided or skipped.
"""

from datetime import date
from typing import Set


def get_forex_holidays(year: int) -> Set[date]:
    """
    Get set of major forex market holidays for a given year.
    
    These are days when the FX market is closed or has very low liquidity.
    We skip new trade entries on these days.
    
    Args:
        year: The year to get holidays for
        
    Returns:
        Set of date objects representing holidays
    """
    holidays = set()
    
    holidays.add(date(year, 1, 1))
    
    holidays.add(date(year, 12, 25))
    holidays.add(date(year, 12, 26))
    
    holidays.add(date(year, 12, 31))
    
    easter_dates = {
        2020: date(2020, 4, 13),
        2021: date(2021, 4, 5),
        2022: date(2022, 4, 18),
        2023: date(2023, 4, 10),
        2024: date(2024, 4, 1),
        2025: date(2025, 4, 21),
        2026: date(2026, 4, 6),
        2027: date(2027, 3, 29),
        2028: date(2028, 4, 17),
        2029: date(2029, 4, 2),
        2030: date(2030, 4, 22),
    }
    
    if year in easter_dates:
        holidays.add(easter_dates[year])
        
        good_friday = easter_dates[year]
        good_friday = date(good_friday.year, good_friday.month, good_friday.day - 3)
        holidays.add(good_friday)
    
    return holidays


def is_forex_holiday(check_date: date) -> bool:
    """
    Check if a given date is a forex market holiday.
    
    Args:
        check_date: The date to check
        
    Returns:
        True if the date is a holiday, False otherwise
    """
    holidays = get_forex_holidays(check_date.year)
    return check_date in holidays


def is_friday(check_date: date) -> bool:
    """
    Check if a given date is a Friday.
    
    Fridays are optionally filtered for new entries to avoid
    weekend gap risk.
    
    Args:
        check_date: The date to check
        
    Returns:
        True if the date is a Friday (weekday 4)
    """
    return check_date.weekday() == 4


def is_weekend(check_date: date) -> bool:
    """
    Check if a given date is a weekend (Saturday or Sunday).
    
    Args:
        check_date: The date to check
        
    Returns:
        True if the date is a weekend
    """
    return check_date.weekday() >= 5


def should_skip_trading(check_date: date, skip_fridays: bool = False) -> bool:
    """
    Determine if trading should be skipped on a given date.
    
    Args:
        check_date: The date to check
        skip_fridays: Whether to also skip Fridays (optional filter)
        
    Returns:
        True if trading should be skipped
    """
    if is_weekend(check_date):
        return True
    
    if is_forex_holiday(check_date):
        return True
    
    if skip_fridays and is_friday(check_date):
        return True
    
    return False


FX_HOLIDAYS_2024 = {
    date(2024, 1, 1),
    date(2024, 3, 29),
    date(2024, 4, 1),
    date(2024, 12, 25),
    date(2024, 12, 26),
    date(2024, 12, 31),
}

FX_HOLIDAYS_2025 = {
    date(2025, 1, 1),
    date(2025, 4, 18),
    date(2025, 4, 21),
    date(2025, 12, 25),
    date(2025, 12, 26),
    date(2025, 12, 31),
}
