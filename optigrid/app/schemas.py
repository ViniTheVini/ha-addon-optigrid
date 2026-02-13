from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Optional, TypeAlias
from datetime import datetime, timezone

# Type alias for time series data: mapping of datetime to float values
TimeSeries: TypeAlias = Dict[datetime, float]

class SensorHistory(BaseModel):
    """Historical sensor data from Home Assistant"""
    house_load_w: TimeSeries = Field(description="Total house load power in Watts (positive = importing, negative = exporting)")
    pv_power_w: Optional[TimeSeries] = Field(None, description="Solar PV production power in Watts (positive = producing)")
    battery_power_w: Optional[TimeSeries] = Field(None, description="Battery power in Watts (positive = discharge, negative = charge)")
    deferrable_loads_power_w: Optional[Dict[str, TimeSeries]] = Field(None, description="Power consumption of deferrable loads in Watts")

class PVConfig(BaseModel):
    """Solar PV system configuration"""
    enabled: bool = Field(default=False)
    power_sensor: Optional[str] = Field(None, description="PV power sensor entity ID")

    @model_validator(mode='after')
    def validate_required_when_enabled(self):
        """Ensure the PV power sensor is provided when enabled"""
        if self.enabled and self.power_sensor is None:
            raise ValueError("PV power_sensor is required when enabled=True")
        return self

class BatteryConfig(BaseModel):
    """Battery system configuration"""
    enabled: bool = Field(default=False)
    current_soc: Optional[float] = Field(None, ge=0, le=1, description="Current state of charge (0-1)")
    power_sensor: Optional[str] = Field(None, description="Battery power sensor entity ID")
    capacity_kwh: Optional[float] = Field(None, gt=0, description="Battery capacity in kWh")
    min_soc: Optional[float] = Field(None, ge=0, le=1, description="Minimum state of charge (0-1)")
    max_soc: Optional[float] = Field(None, ge=0, le=1, description="Maximum state of charge (0-1)")
    max_charge_kw: Optional[float] = Field(None, gt=0, description="Maximum charging power in kW")
    max_discharge_kw: Optional[float] = Field(None, gt=0, description="Maximum discharging power in kW")
    efficiency: Optional[float] = Field(default=1, gt=0, le=1, description="Round-trip efficiency (0-1)")
    degradation_cost_per_kwh: Optional[float] = Field(
        default=0.0, 
        ge=0, 
        description="Battery degradation cost per kWh cycled (€/kWh). Set to 0 to disable."
    )

    @model_validator(mode='after')
    def validate_required_when_enabled(self):
        """Ensure all required fields are provided when battery is enabled"""
        if self.enabled:
            if self.power_sensor is None:
                raise ValueError("Battery power_sensor is required when enabled=True")
            if self.current_soc is None:
                raise ValueError("Battery current_soc is required when enabled=True")
            if self.capacity_kwh is None:
                raise ValueError("Battery capacity_kwh is required when enabled=True")
            if self.min_soc is None:
                raise ValueError("Battery min_soc is required when enabled=True")
            if self.max_soc is None:
                raise ValueError("Battery max_soc is required when enabled=True")
            if self.max_charge_kw is None:
                raise ValueError("Battery max_charge_kw is required when enabled=True")
            if self.max_discharge_kw is None:
                raise ValueError("Battery max_discharge_kw is required when enabled=True")
        return self

    @model_validator(mode='after')
    def validate_min_max_soc(self):
        """Ensure the Battery Minimum State of Charge is less than the Maximum State of Charge"""
        if self.enabled:
            # At this point, validate_required_when_enabled has already ensured these are not None
            if self.min_soc >= self.max_soc:
                raise ValueError(f"Battery min_soc ({self.min_soc}) must be less than max_soc ({self.max_soc})")
        return self

class GridConfig(BaseModel):
    """Grid connection configuration"""
    power_sensor: str = Field(description="Grid power sensor entity ID")
    max_import_kw: float = Field(gt=0, description="Maximum grid import power in kW")
    max_export_kw: float = Field(gt=0, description="Maximum grid export power in kW")

class DeferrableLoadConfig(BaseModel):
    """
    Configuration for a single deferrable load.

    The optimizer schedules this load to deliver the required energy
    at minimum cost, subject to time constraints.

    Important: You must calculate and provide remaining_energy_kwh based
    on your own energy tracking. The optimizer does not fetch sensor data
    or track energy consumption across optimization runs.

    Example:
        If your EV needs 30 kWh total and has already received 14.8 kWh,
        set remaining_energy_kwh to 15.2 kWh.
    """
    enabled: bool = Field(
        default=False,
        description="Whether this load should be included in optimization"
    )

    power_w: Optional[float] = Field(
        default=None,
        gt=0,
        description="Nominal power consumption in Watts (required when enabled=True)"
    )

    power_sensor: Optional[str] = Field(None, description="Power sensor entity ID")
    
    remaining_energy_kwh: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Energy still needed to complete the task (kWh). "
            "You are responsible for calculating this value based on "
            "your energy tracking system (e.g., target - current). "
            "(required when enabled=True)"
        )
    )

    continuity_penalty: float = Field(
        default=0.10,
        ge=0,
        description=(
            "Penalty cost per start/stop cycle (€). "
            "Higher values discourage fragmentation and encourage continuous operation. "
            "Lower values allow more flexibility to split charging across multiple windows. "
            "Default: €0.10 per start. "
            "Typical values: "
            "  - €0.05-0.10: Flexible (allows economically justified splits) "
            "  - €0.20-0.50: Balanced (prevents most fragmentation) "
            "  - €0.50-1.00: Strict (forces nearly continuous operation) "
            "Set to 0 to disable continuity preference (not recommended - may cause excessive fragmentation)."
        )
    )

    earliest_start: Optional[datetime] = Field(
        None,
        description="Earliest allowed start time (optional, must be timezone-aware)"
    )

    latest_end: Optional[datetime] = Field(
        None,
        description="Deadline for completion (optional, must be timezone-aware)"
    )

    @model_validator(mode='after')
    def validate_required_when_enabled(self):
        """Ensure required fields are provided when enabled=True"""
        if self.enabled:
            if self.power_sensor is None:
                raise ValueError("DeferrableLoad power_sensor is required when enabled=True")
            if self.power_w is None:
                raise ValueError("DeferrableLoad power_w is required when enabled=True")
            if self.remaining_energy_kwh is None:
                raise ValueError("DeferrableLoad remaining_energy_kwh is required when enabled=True")
        return self

    @model_validator(mode='after')
    def validate_timezone_awareness(self):
        """Ensure datetime fields are timezone-aware"""
        if self.earliest_start is not None and self.earliest_start.tzinfo is None:
            raise ValueError(
                f"DeferrableLoad earliest_start must be timezone-aware, got naive datetime: {self.earliest_start}"
            )
        if self.latest_end is not None and self.latest_end.tzinfo is None:
            raise ValueError(
                f"DeferrableLoad latest_end must be timezone-aware, got naive datetime: {self.latest_end}"
            )
        return self

    @model_validator(mode='after')
    def validate_time_window(self):
        """Ensure earliest_start is before latest_end"""
        if self.earliest_start is not None and self.latest_end is not None:
            if self.earliest_start >= self.latest_end:
                raise ValueError(
                    f"DeferrableLoad earliest_start ({self.earliest_start}) must be before "
                    f"latest_end ({self.latest_end})"
                )
        return self

    @model_validator(mode='after')
    def validate_energy_feasibility(self):
        """Ensure the required energy can be delivered within the time window"""
        if self.enabled and self.earliest_start is not None and self.latest_end is not None:
            # Calculate available time window in hours
            time_window_seconds = (self.latest_end - self.earliest_start).total_seconds()
            time_window_hours = time_window_seconds / 3600

            # Calculate maximum energy that can be delivered
            power_kw = self.power_w / 1000  # Convert W to kW
            max_deliverable_kwh = power_kw * time_window_hours

            # Check if required energy can be delivered
            if self.remaining_energy_kwh > max_deliverable_kwh:
                raise ValueError(
                    f"DeferrableLoad cannot deliver {self.remaining_energy_kwh} kWh within the time window. "
                    f"Maximum deliverable: {max_deliverable_kwh:.2f} kWh "
                    f"(power: {power_kw:.2f} kW, window: {time_window_hours:.2f} hours)"
                )
        return self



class HAConnection(BaseModel):
    """Home Assistant connection details for fetching sensor data"""
    url: str = Field(description="Home Assistant URL (e.g., http://homeassistant.local:8123)")
    token: str = Field(description="Long-lived access token for Home Assistant API")

class Config(BaseModel):
    """System configuration"""
    pv: PVConfig = Field(default_factory=PVConfig)
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    lookback_days: int = Field(default=2, gt=0, description="Number of days of history to fetch")
    deferrable_loads: Dict[str, DeferrableLoadConfig] = Field(default_factory=dict)
    ha_connection: Optional[HAConnection] = Field(None, description="Home Assistant connection details for fetching sensor history")

class TimeConfig(BaseModel):
    """Time configuration for optimization"""
    start: Optional[datetime] = Field(description="Start time for optimization horizon", default_factory=lambda: datetime.now(timezone.utc))
    timestep_minutes: int = Field(gt=0, description="Time step in minutes (e.g., 15)")
    horizon_steps: int = Field(gt=0, description="Number of time steps in horizon (e.g., 96 for 24h)")


class PriceForecast(BaseModel):
    """Grid price forecast"""
    import_price: TimeSeries = Field(description="Import price in currency/kWh")
    export_price: TimeSeries = Field(description="Export price in currency/kWh")


class Forecasts(BaseModel):
    """External forecasts provided by user"""
    pv_power_w: Optional[TimeSeries] = Field(None, description="Solar PV production forecast in Watts")
    grid_price: PriceForecast = Field(description="Grid import/export price forecast")

class OptimizeRequest(BaseModel):
    """Complete optimization request"""
    meta: Dict[str, str] = Field(default_factory=dict, description="Metadata (request_id, version, etc.)")
    time: TimeConfig
    config: Config
    sensor_history: Optional[SensorHistory] = Field(None, description="Historical sensor data from Home Assistant (deprecated - use ha_connection instead)")
    forecasts: Forecasts

    @model_validator(mode='after')
    def validate_deferrable_load_time_windows(self):
        """Ensure deferrable load time windows are within the optimization horizon"""
        from datetime import timedelta

        # Calculate optimization horizon end time
        horizon_end = self.time.start + timedelta(minutes=self.time.timestep_minutes * self.time.horizon_steps)

        # Check each deferrable load
        for load_name, load_config in self.config.deferrable_loads.items():
            if not load_config.enabled:
                continue

            # Check earliest_start is within horizon
            if load_config.earliest_start is not None:
                if load_config.earliest_start < self.time.start:
                    raise ValueError(
                        f"DeferrableLoad '{load_name}' earliest_start ({load_config.earliest_start}) "
                        f"is before optimization start time ({self.time.start})"
                    )
                if load_config.earliest_start >= horizon_end:
                    raise ValueError(
                        f"DeferrableLoad '{load_name}' earliest_start ({load_config.earliest_start}) "
                        f"is at or after optimization horizon end ({horizon_end})"
                    )

            # Check latest_end is within horizon
            if load_config.latest_end is not None:
                if load_config.latest_end <= self.time.start:
                    raise ValueError(
                        f"DeferrableLoad '{load_name}' latest_end ({load_config.latest_end}) "
                        f"is at or before optimization start time ({self.time.start})"
                    )
                if load_config.latest_end > horizon_end:
                    raise ValueError(
                        f"DeferrableLoad '{load_name}' latest_end ({load_config.latest_end}) "
                        f"is after optimization horizon end ({horizon_end})"
                    )

        return self


# ---------------- RESPONSE ----------------

class BatterySchedulePoint(BaseModel):
    """Single point in battery schedule"""
    ts: datetime
    power_kw: float = Field(description="Battery power in kW (positive = discharge, negative = charge)")
    soc: float = Field(ge=0, le=1, description="Predicted state of charge at this time")


class DeferrableLoadWindow(BaseModel):
    """Single time window for a deferrable load"""
    start_time: datetime
    end_time: datetime

class DeferrableLoadSchedule(BaseModel):
    """Schedule for a deferrable load (may have multiple windows)"""
    windows: List[DeferrableLoadWindow]
    power_w: float = Field(description="Power consumption in Watts")
    total_duration_hours: float
    is_active_now: bool = Field(description="Whether the load should be active at the optimization start time")


class CostBreakdown(BaseModel):
    """Cost breakdown"""
    total_cost: float = Field(description="Total cost in currency units")
    import_cost: float = Field(description="Cost of imported energy")
    export_revenue: float = Field(description="Revenue from exported energy (negative cost)")
    import_energy_kwh: float = Field(description="Total imported energy in kWh")
    export_energy_kwh: float = Field(description="Total exported energy in kWh")


class BatteryResult(BaseModel):
    """Battery optimization result"""
    next_power_kw: float = Field(description="Recommended battery power for next timestep (kW)")
    schedule: List[BatterySchedulePoint] = Field(description="Full battery schedule for horizon")


class OptimizeResult(BaseModel):
    """Optimization results"""
    battery: Optional[BatteryResult] = Field(None, description="Battery optimization result (None if battery disabled)")
    deferrable_loads: Dict[str, DeferrableLoadSchedule] = Field(default_factory=dict)
    cost: CostBreakdown


class Diagnostics(BaseModel):
    """Diagnostic information"""
    solver_status: str = Field(description="Optimization solver status (Optimal, Infeasible, Unbounded, etc.)")
    solve_time_ms: float = Field(description="Time taken to solve in milliseconds")
    warnings: List[str] = Field(default_factory=list)
    uncontrolled_load_forecast: Optional[TimeSeries] = Field(None, description="Forecasted uncontrolled load")
    
    is_optimal: bool = Field(description="True if solver found optimal solution")
    fallback_used: bool = Field(default=False, description="True if fallback values were used")
    problem_stats: Optional[Dict[str, int]] = Field(None, description="Problem size statistics")
    debug_file: Optional[str] = Field(None, description="Path to debug LP file (if generated)")


class OptimizeResponse(BaseModel):
    """Complete optimization response"""
    status: str = Field(description="Response status: 'ok' or 'error'")
    result: Optional[OptimizeResult] = None
    diagnostics: Diagnostics
    error: Optional[str] = Field(None, description="Error message if status is 'error'")
