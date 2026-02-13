from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional
from schemas import TimeSeries


class TimeAlignmentError(Exception):
    pass


def generate_time_index(
    start: datetime,
    end: datetime,
    step_minutes: int,
) -> Iterable[datetime]:
    """
    Generate a fixed time index [start, end) with given resolution.
    """
    if start >= end:
        raise TimeAlignmentError("start must be before end")

    step = timedelta(minutes=step_minutes)
    t = start
    while t < end:
        yield t
        t += step


def resample_forward_fill(
    series: TimeSeries,
    time_index: Iterable[datetime],
    default: Optional[float] = None
) -> TimeSeries:
    """
    Resample a time series onto a fixed time index using forward-fill.

    - Assumes series timestamps are sorted or sortable
    - If no previous value exists:
        - use default if provided
        - otherwise raise
    """

    if not series and default is None:
        raise TimeAlignmentError("Cannot resample empty series without default")

    sorted_items = sorted(series.items(), key=lambda x: x[0])
    result: TimeSeries = {}

    idx = 0
    last_value = None

    timestamps = [t for t, _ in sorted_items]
    values = [v for _, v in sorted_items]

    for t in time_index:
        while idx < len(timestamps) and timestamps[idx] <= t:
            last_value = values[idx]
            idx += 1

        if last_value is None:
            if default is None:
                raise TimeAlignmentError(
                    f"No value available to fill timestamp {t}"
                )
            result[t] = default
        else:
            result[t] = last_value

    return result


def align_series(
    series: TimeSeries,
    start: datetime,
    end: datetime,
    step_minutes: int,
    default: Optional[float] = None
) -> TimeSeries:
    """
    Convenience wrapper:
    - build canonical time index
    - resample using forward-fill
    """
    time_index = generate_time_index(start, end, step_minutes)
    return resample_forward_fill(series, time_index, default)
