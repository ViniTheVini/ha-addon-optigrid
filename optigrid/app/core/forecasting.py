from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict
import statistics

from .time_alignment import generate_time_index
from .decomposition import TimeSeries

ForecastSeries = Dict[datetime, float]


def compute_daily_profile(
    history: TimeSeries,
    window_days: int = 14
) -> Dict[int, List[float]]:
    """
    Compute daily profile:
    - key: minutes since midnight
    - value: list of historical load values at that time
    """

    profile: Dict[int, List[float]] = defaultdict(list)

    sorted_ts = sorted(history.keys())
    start_day = sorted_ts[-1] - timedelta(days=window_days)

    for ts in sorted_ts:
        if ts < start_day:
            continue
        minutes = ts.hour * 60 + ts.minute
        profile[minutes].append(history[ts])

    return profile


def forecast_rolling_profile(
    history: TimeSeries,
    start: datetime,
    horizon_steps: int = 96,
    step_minutes: int = 15,
    window_days: int = 14,
) -> ForecastSeries:
    """
    Deterministic rolling profile forecast

    Inputs:
    - history: uncontrolled load series (aligned)
    - start: first forecast timestep
    - horizon_steps: number of steps (default 96)
    - step_minutes: timestep resolution (default 15 min)
    - window_days: number of days to build profile

    Returns:
    - ForecastSeries: datetime -> load (kW)
    """

    # Step 1: compute daily profile
    profile = compute_daily_profile(history, window_days)

    # Step 2: build canonical time index
    time_index = list(generate_time_index(start, start + timedelta(minutes=step_minutes*horizon_steps), step_minutes))

    forecast: ForecastSeries = {}

    for ts in time_index:
        minutes = ts.hour * 60 + ts.minute

        # Base profile: median to be robust
        base_vals = profile.get(minutes, [0.0])
        base = statistics.median(base_vals)

        forecast[ts] = base

    return forecast
