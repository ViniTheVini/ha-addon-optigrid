"""
Data preprocessing for optimization pipeline.
Converts sensor history to aligned time series and forecasts uncontrolled load.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import csv
import logging
import os
from schemas import SensorHistory, TimeSeries
from .decomposition import decompose_uncontrolled_load
from .forecasting import forecast_rolling_profile
from .time_alignment import align_series

_LOGGER = logging.getLogger(__name__)

def preprocess_sensor_history(
    sensor_history: SensorHistory,
    start: datetime,
    timestep_minutes: int,
    lookback_days: int = 2
) -> TimeSeries:
    """
    Process sensor history to extract uncontrolled load time series.

    Steps:
    1. Convert sensor data to time series dicts
    2. Align all series to common time grid
    3. Decompose: uncontrolled_load = house_load + pv_power - battery_power - deferrable_loads
       (house_load is net grid consumption, already reduced by PV, so we add PV back)
    4. Return aligned uncontrolled load for forecasting

    Args:
        sensor_history: Historical sensor data from HA (includes optional deferrable load data)
        start: Start time for optimization
        timestep_minutes: Time resolution
        lookback_days: How many days of history to use

    Returns:
        Aligned uncontrolled load time series (gross house consumption excluding deferrable loads)
    """

    house_load = sensor_history.house_load_w

    # Handle optional PV and battery data
    pv_power = sensor_history.pv_power_w if sensor_history.pv_power_w else {}
    battery_power = sensor_history.battery_power_w if sensor_history.battery_power_w else {}

    # Handle optional deferrable loads data
    deferrable_loads_raw = sensor_history.deferrable_loads_power_w if sensor_history.deferrable_loads_power_w else {}

    # Determine time range for alignment
    end = start
    start_history = start - timedelta(days=lookback_days)

    # Align all series to common grid
    house_load_aligned = align_series(
        house_load,
        start_history,
        end,
        timestep_minutes,
        default=0.0
    )

    pv_power_aligned = align_series(
        pv_power,
        start_history,
        end,
        timestep_minutes,
        default=0.0
    )

    battery_power_aligned = align_series(
        battery_power,
        start_history,
        end,
        timestep_minutes,
        default=0.0
    )

    # Align each deferrable load's historical data to the common time grid
    deferrable_loads_aligned: Dict[str, TimeSeries] = {}
    if deferrable_loads_raw:
        for load_name, load_timeseries in deferrable_loads_raw.items():
            deferrable_loads_aligned[load_name] = align_series(
                load_timeseries,
                start_history,
                end,
                timestep_minutes,
                default=0.0
            )

    # Export aligned data to CSV for charting (only in debug mode)
    if _LOGGER.isEnabledFor(logging.DEBUG):
        csv_filename = "/data/aligned_data.csv"
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Build header with deferrable load columns
                header = ['timestamp', 'house_load_w', 'pv_power_w', 'battery_power_w']
                for load_name in sorted(deferrable_loads_aligned.keys()):
                    header.append(f'deferrable_{load_name}_w')
                writer.writerow(header)

                # Get all timestamps (they should all be aligned to the same grid)
                all_timestamps = sorted(house_load_aligned.keys())

                for ts in all_timestamps:
                    row = [
                        ts.isoformat(),
                        house_load_aligned.get(ts, 0.0),
                        pv_power_aligned.get(ts, 0.0),
                        battery_power_aligned.get(ts, 0.0)
                    ]
                    # Add deferrable load values
                    for load_name in sorted(deferrable_loads_aligned.keys()):
                        row.append(deferrable_loads_aligned[load_name].get(ts, 0.0))
                    writer.writerow(row)

            _LOGGER.debug(f"Aligned data CSV saved to: {csv_filename}")
        except Exception as e:
            _LOGGER.warning(f"Could not save aligned data CSV: {e}")

    # Decompose to get uncontrolled load
    # uncontrolled_load = house_load + pv_power - battery_power - deferrable_loads
    # Note: house_load is net grid consumption (already reduced by PV production)
    # We ADD pv_power back to get the gross house consumption
    # Battery power: positive = discharge (adds to house), negative = charge (subtracts from house)
    # Deferrable loads: subtract historical consumption to prevent double-counting in optimization
    uncontrolled_load = decompose_uncontrolled_load(
        total_load=house_load_aligned,
        battery_power=battery_power_aligned,
        pv_power=pv_power_aligned,
        deferrable_loads=deferrable_loads_aligned if deferrable_loads_aligned else None
    )

    # Convert from W to kW for consistency
    uncontrolled_load_kw = {ts: val / 1000.0 for ts, val in uncontrolled_load.items()}

    return uncontrolled_load_kw


def prepare_forecast_inputs(
    uncontrolled_load_history: TimeSeries,
    pv_forecast_points: Optional[TimeSeries],
    start: datetime,
    horizon_steps: int,
    timestep_minutes: int
) -> tuple[TimeSeries, TimeSeries]:
    """
    Prepare forecast inputs for optimization.

    Args:
        uncontrolled_load_history: Historical uncontrolled load (kW)
        pv_forecast_points: User-provided PV forecast (W), None if PV disabled
        start: Optimization start time
        horizon_steps: Number of steps
        timestep_minutes: Time resolution

    Returns:
        (uncontrolled_load_forecast_kw, pv_forecast_kw)
    """

    # Forecast uncontrolled load using rolling profile
    uncontrolled_forecast_kw = forecast_rolling_profile(
        history=uncontrolled_load_history,
        start=start,
        horizon_steps=horizon_steps,
        step_minutes=timestep_minutes,
        window_days=14
    )

    # Save CSV for charting: historical + forecasted data (only in debug mode)
    if _LOGGER.isEnabledFor(logging.DEBUG):
        csv_path = "data/house_load_forecast.csv"
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            with open(csv_path, 'w') as f:
                f.write("Timestamp,Load_kW,Type\n")

                # Write historical data (sorted by timestamp)
                for ts in sorted(uncontrolled_load_history.keys()):
                    f.write(f"{ts.isoformat()},{uncontrolled_load_history[ts]:.3f},Historical\n")

                # Write forecasted data (sorted by timestamp)
                for ts in sorted(uncontrolled_forecast_kw.keys()):
                    f.write(f"{ts.isoformat()},{uncontrolled_forecast_kw[ts]:.3f},Forecasted\n")

            _LOGGER.debug(f"House load forecast CSV saved to: {csv_path}")
        except Exception as e:
            _LOGGER.warning(f"Could not save forecast CSV: {e}")

    # Align PV forecast to optimization grid (or use zeros if PV disabled)
    if pv_forecast_points:
        pv_forecast_w = pv_forecast_points
    else:
        pv_forecast_w = {}

    end = start + timedelta(minutes=timestep_minutes * horizon_steps)

    pv_forecast_aligned_w = align_series(
        pv_forecast_w,
        start,
        end,
        timestep_minutes,
        default=0.0
    )

    # Convert PV from W to kW
    pv_forecast_kw = {ts: val / 1000.0 for ts, val in pv_forecast_aligned_w.items()}

    return uncontrolled_forecast_kw, pv_forecast_kw

