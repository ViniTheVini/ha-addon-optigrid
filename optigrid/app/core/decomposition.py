from typing import Dict, Optional
from datetime import datetime
from schemas import TimeSeries

class DecompositionError(Exception):
    pass


def decompose_uncontrolled_load(
    total_load: TimeSeries,
    battery_power: Optional[TimeSeries] = None,
    pv_power: Optional[TimeSeries] = None,
    deferrable_loads: Optional[Dict[str, TimeSeries]] = None,
) -> TimeSeries:
    """
    Compute uncontrolled load time series.

    uncontrolled_load(t) =
        total_load(t)
      + pv_power(t)
      - battery_power(t)
      - sum(deferrable_loads_i(t))

    Args:
        total_load: Net grid consumption (positive = import, negative = export)
                   This is already reduced by PV production
        battery_power: Battery power (positive = discharge, negative = charge)
        pv_power: Solar PV production (positive = producing)
                 Added back because total_load is net consumption
        deferrable_loads: Historical deferrable loads to subtract

    Rules:
    - Missing battery, PV, or deferrable loads are treated as zero
    - Missing timestamps in total_load are not allowed
    - Missing timestamps in optional series default to zero
    """

    if not total_load:
        raise DecompositionError("total_load is required and cannot be empty")

    battery_power = battery_power or {}
    pv_power = pv_power or {}
    deferrable_loads = deferrable_loads or {}

    uncontrolled_load: TimeSeries = {}

    for ts, total in total_load.items():
        value = total

        # Add PV power back (total_load is net consumption, already reduced by PV)
        pv = pv_power.get(ts, 0.0)
        value += pv

        # Subtract battery power if available
        batt = battery_power.get(ts, 0.0)
        value -= batt

        # Subtract each deferrable load
        for load_name, series in deferrable_loads.items():
            value -= series.get(ts, 0.0)

        uncontrolled_load[ts] = value

    return uncontrolled_load
