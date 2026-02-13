from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from schemas import (
    OptimizeRequest,
    OptimizeResponse,
    BatterySchedulePoint,
    BatteryResult,
    DeferrableLoadSchedule,
    DeferrableLoadWindow,
    CostBreakdown,
    OptimizeResult,
    Diagnostics,
    SensorHistory
)
from datetime import date, datetime, timezone, timedelta
import traceback
import logging
import os
import json

# Configure logging - level will be controlled by uvicorn's --log-level parameter
# Don't set level here to allow uvicorn to control it
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

_LOGGER = logging.getLogger(__name__)

# Supervisor API support - automatically available when running as HA addon
SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")
SUPERVISOR_API_URL = "http://supervisor/core"  # Base URL - /api/... will be added by ha_client

if SUPERVISOR_TOKEN:
    _LOGGER.info("Running as Home Assistant addon - Supervisor API available")
else:
    _LOGGER.info("Running in standalone mode - Supervisor API not available")

from core.preprocessing import (
    preprocess_sensor_history,
    prepare_forecast_inputs,
)
from core.optimizer import EnergyOptimizer, OptimizationError
from ha_client import fetch_sensor_history_from_ha



def stringify_datetime_keys(d: dict) -> dict:
    return {
        (k.isoformat() if isinstance(k, (datetime, date)) else str(k)): v for k, v in d.items()    
    }

app = FastAPI(
    title="HA Forecasting Extension",
    description="Energy optimization service for Home Assistant",
    version="0.1.0"
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for validation errors (422).
    Provides detailed error messages with field locations and error types.
    """
    errors = exc.errors()

    # Log the full validation error for debugging
    _LOGGER.error(f"Validation error on {request.url.path}")
    _LOGGER.error(f"Request body: {await request.body()}")

    # Safely serialize errors by converting to string representation
    # This avoids JSON serialization errors when errors contain exception objects
    try:
        _LOGGER.error(f"Validation errors: {json.dumps(errors, indent=2, default=str)}")
    except (TypeError, ValueError) as e:
        _LOGGER.error(f"Validation errors (raw): {errors}")
        _LOGGER.error(f"Failed to serialize errors to JSON: {e}")

    # Format errors in a more readable way
    detailed_errors = []
    for error in errors:
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        detailed_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation Error",
            "errors": detailed_errors,
            "error_count": len(detailed_errors),
            "help": "Check the 'errors' field for detailed information about each validation failure."
        }
    )


@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "HA Forecasting Extension"}


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    """
    Main optimization endpoint.

    Processes sensor history, forecasts uncontrolled load,
    and optimizes battery + deferrable loads to minimize cost.

    Supports two modes:
    1. Legacy mode: sensor_history is provided directly in the request
    2. New mode: ha_connection is provided, and sensor history is fetched from HA
    """

    start_time = req.time.start
    warnings = []

    _LOGGER.info(f"Received optimization request: {req}")

    try:
        # Step 1: Get sensor history (either from request or fetch from HA)
        lookback_days = req.config.lookback_days

        # Determine HA connection details
        # Priority: Supervisor API (production) > ha_connection (development/testing) > sensor_history (legacy)
        ha_url = None
        ha_token = None

        if SUPERVISOR_TOKEN:
            # Running as HA addon - use Supervisor API automatically
            ha_url = SUPERVISOR_API_URL
            ha_token = SUPERVISOR_TOKEN
            _LOGGER.info("✓ Using Supervisor API for Home Assistant access")
            warnings.append("Using Supervisor API for automatic authentication")

            # Warn if ha_connection was also provided (should not happen in production)
            if req.config.ha_connection:
                _LOGGER.warning(
                    "Both Supervisor API and ha_connection provided. "
                    "Using Supervisor API (ha_connection ignored)."
                )
        elif req.config.ha_connection:
            # Development/testing mode - user provided explicit connection details
            ha_url = req.config.ha_connection.url
            ha_token = req.config.ha_connection.token
            _LOGGER.info(f"Using user-provided HA connection: {ha_url}")
            warnings.append("Using manual HA connection (development mode)")

        if ha_url and ha_token:
            # Fetch sensor history from Home Assistant

            # Validate required sensor configuration
            if not req.config.grid.power_sensor:
                raise ValueError(
                    "grid.power_sensor is required when using ha_connection mode or Supervisor API"
                )

            history_start = start_time - timedelta(days=lookback_days)

            try:
                house_load_w = await fetch_sensor_history_from_ha(
                    ha_url,
                    ha_token,
                    req.config.grid.power_sensor,
                    history_start,
                    start_time
                )

                pv_power_w = await fetch_sensor_history_from_ha(
                    ha_url,
                    ha_token,
                    req.config.pv.power_sensor,
                    history_start,
                    start_time
                ) if req.config.pv.enabled and req.config.pv.power_sensor else None

                battery_power_w = await fetch_sensor_history_from_ha(
                    ha_url,
                    ha_token,
                    req.config.battery.power_sensor,
                    history_start,
                    start_time
                ) if req.config.battery.enabled and req.config.battery.power_sensor else None

                deferrable_loads_power_w = {}
                deferrable_loads_with_sensors = []
                deferrable_loads_without_sensors = []

                for name, load_cfg in req.config.deferrable_loads.items():
                    if load_cfg.power_sensor is None:
                        deferrable_loads_without_sensors.append(name)
                        continue

                    load_power_w = await fetch_sensor_history_from_ha(
                        ha_url,
                        ha_token,
                        load_cfg.power_sensor,
                        history_start,
                        start_time
                    )
                    deferrable_loads_power_w[name] = load_power_w
                    deferrable_loads_with_sensors.append(name)

                sensor_history = SensorHistory(
                    house_load_w=house_load_w,
                    pv_power_w=pv_power_w,
                    battery_power_w=battery_power_w,
                    deferrable_loads_power_w=deferrable_loads_power_w
                )

                warnings.append(f"Fetched sensor history from HA (lookback: {lookback_days} days)")

                # Add informative warnings about deferrable load handling
                if deferrable_loads_with_sensors:
                    warnings.append(
                        f"Deferrable loads with power sensors (will be subtracted from historical load): "
                        f"{', '.join(deferrable_loads_with_sensors)}"
                    )

                if deferrable_loads_without_sensors:
                    warnings.append(
                        f"⚠️ Deferrable loads WITHOUT power sensors: {', '.join(deferrable_loads_without_sensors)}. "
                        f"Ensure these loads are NOT included in house_load_w sensor to avoid double-counting."
                    )
            except Exception as e:
                raise ValueError(f"Failed to fetch sensor history from Home Assistant: {str(e)}")

        elif req.sensor_history:
            # Legacy mode: sensor_history provided directly in request
            sensor_history = req.sensor_history
            _LOGGER.info("Using sensor history provided in request (legacy mode)")
            warnings.append("Using sensor history from request payload (legacy mode)")

        else:
            # No sensor history source available
            if not SUPERVISOR_TOKEN:
                raise ValueError(
                    "❌ No sensor history source available.\n\n"
                    "This addon requires Home Assistant Supervisor API to function.\n"
                    "Please ensure:\n"
                    "  1. The addon is installed via Home Assistant Supervisor\n"
                    "  2. The addon manifest has 'homeassistant_api: true'\n"
                    "  3. The addon is running (not in standalone mode)\n\n"
                    "For development/testing, you can provide 'ha_connection' in the config or 'sensor_history' in the request."
                )
            else:
                raise ValueError(
                    "❌ Supervisor API is available but no sensor history could be fetched.\n"
                    "This should not happen. Please check addon logs for errors."
                )

        # Step 2: Preprocess sensor history to get uncontrolled load
        uncontrolled_load_kw = preprocess_sensor_history(
            sensor_history=sensor_history,
            start=start_time,
            timestep_minutes=req.time.timestep_minutes,
            lookback_days=lookback_days
        )

        # Step 3: Prepare forecasts
        uncontrolled_forecast_kw, pv_forecast_kw = prepare_forecast_inputs(
            uncontrolled_load_history=uncontrolled_load_kw,
            pv_forecast_points=req.forecasts.pv_power_w,
            start=start_time,
            horizon_steps=req.time.horizon_steps,
            timestep_minutes=req.time.timestep_minutes
        )

        # Step 4: Prepare price forecasts
        import_price = req.forecasts.grid_price.import_price
        export_price = req.forecasts.grid_price.export_price

        # Step 5: Prepare deferrable loads config
        deferrable_loads_config = None
        if req.config.deferrable_loads:
            deferrable_loads_config = {}
            for name, load_cfg in req.config.deferrable_loads.items():
                if not load_cfg.enabled or load_cfg.remaining_energy_kwh <= 0:
                    continue

                _LOGGER.debug(f"Deferrable load config: {name} - {load_cfg}")
                duration_hours = load_cfg.remaining_energy_kwh / (load_cfg.power_w / 1000.0)
                deferrable_loads_config[name] = {
                    'power_kw': load_cfg.power_w / 1000.0,
                    'duration_hours': duration_hours,
                    'earliest_start': load_cfg.earliest_start,
                    'latest_end': load_cfg.latest_end,
                    'continuity_penalty': load_cfg.continuity_penalty
                }
            _LOGGER.info(f"Deferrable loads config: {deferrable_loads_config}")

        _LOGGER.info(f"All prepared. Starting optimization...")
        # Step 6: Run optimization
        optimizer = EnergyOptimizer(
            timestep_minutes=req.time.timestep_minutes,
            horizon_steps=req.time.horizon_steps,
            start_time=start_time
        )

        # Check which features are enabled
        pv_enabled = req.config.pv.enabled
        battery_enabled = req.config.battery.enabled

        # Add warnings if features are disabled
        if not pv_enabled:
            warnings.append("PV is disabled - optimization will not consider solar generation")
        if not battery_enabled:
            warnings.append("Battery is disabled - optimization will not include battery scheduling")

        # Prepare battery parameters (use defaults if battery disabled)
        if battery_enabled:
            # Validate required battery parameters
            required_params = {
                'capacity_kwh': req.config.battery.capacity_kwh,
                'current_soc': req.config.battery.current_soc,
                'min_soc': req.config.battery.min_soc,
                'max_soc': req.config.battery.max_soc,
                'max_charge_kw': req.config.battery.max_charge_kw,
                'max_discharge_kw': req.config.battery.max_discharge_kw,
                'efficiency': req.config.battery.efficiency,
            }

            missing = [name for name, value in required_params.items() if value is None]
            if missing:
                raise ValueError(
                    f"Battery is enabled but missing required parameters: {', '.join(missing)}"
                )

            battery_capacity_kwh = req.config.battery.capacity_kwh
            battery_soc_init = req.config.battery.current_soc
            battery_min_soc = req.config.battery.min_soc
            battery_max_soc = req.config.battery.max_soc
            battery_max_charge_kw = req.config.battery.max_charge_kw
            battery_max_discharge_kw = req.config.battery.max_discharge_kw
            battery_efficiency = req.config.battery.efficiency
            battery_degradation_cost_per_kwh = req.config.battery.degradation_cost_per_kwh

            # Validate battery SOC constraints
            if battery_soc_init < battery_min_soc:
                _LOGGER.warning(
                    f"Battery current_soc ({battery_soc_init:.3f}) is below min_soc ({battery_min_soc:.3f}). "
                    f"This may indicate a sensor error or configuration issue."
                )
                warnings.append(f"Battery SOC ({battery_soc_init:.3f}) below minimum ({battery_min_soc:.3f})")

            if battery_soc_init > battery_max_soc:
                _LOGGER.warning(
                    f"Battery current_soc ({battery_soc_init:.3f}) exceeds max_soc ({battery_max_soc:.3f}). "
                    f"This should not be possible and indicates a sensor error or configuration issue. "
                )
                warnings.append(f"Battery SOC ({battery_soc_init:.3f}) exceeds maximum ({battery_max_soc:.3f}) - clamped to max")

        opt_result = optimizer.optimize(
            uncontrolled_load_kw=uncontrolled_forecast_kw,
            pv_forecast_kw=pv_forecast_kw,
            import_price=import_price,
            export_price=export_price,
            battery_enabled=battery_enabled,
            battery_capacity_kwh=battery_capacity_kwh,
            battery_soc_init=battery_soc_init,
            battery_min_soc=battery_min_soc,
            battery_max_soc=battery_max_soc,
            battery_max_charge_kw=battery_max_charge_kw,
            battery_max_discharge_kw=battery_max_discharge_kw,
            battery_efficiency=battery_efficiency,
            grid_max_import_kw=req.config.grid.max_import_kw,
            grid_max_export_kw=req.config.grid.max_export_kw,
            deferrable_loads=deferrable_loads_config,
            battery_degradation_cost_per_kwh=battery_degradation_cost_per_kwh
        )


        _LOGGER.info(f"Optimization completed: {opt_result['solver_status']}")

        # Step 7: Format response
        # Only include battery result if battery is enabled
        if battery_enabled:
            battery_schedule = [
                BatterySchedulePoint(
                    ts=ts,
                    power_kw=opt_result['battery_power_kw'][ts],
                    soc=opt_result['battery_soc'][ts]
                )
                for ts in optimizer.time_index
            ]
            battery_result = BatteryResult(
                next_power_kw=battery_schedule[0].power_kw,
                schedule=battery_schedule
            )
        else:
            battery_result = None

        _LOGGER.debug(f"Deferrable schedules: {opt_result['deferrable_schedules']}")
        deferrable_schedules = {}
        for name, schedule_data in opt_result['deferrable_schedules'].items():
            # Convert optimizer output format to API schema format
            windows = [
                DeferrableLoadWindow(
                    start_time=window['start'],
                    end_time=window['end']
                )
                for window in schedule_data['windows']
            ]
            deferrable_schedules[name] = DeferrableLoadSchedule(
                windows=windows,
                power_w=req.config.deferrable_loads[name].power_w,
                total_duration_hours=schedule_data['total_duration_hours'],
                is_active_now=schedule_data['is_active_now']
            )

        # Calculate cost breakdown
        dt_hours = req.time.timestep_minutes / 60.0
        import_energy = sum(max(0, p) for p in opt_result['grid_power_kw'].values()) * dt_hours
        export_energy = sum(max(0, -p) for p in opt_result['grid_power_kw'].values()) * dt_hours
        import_cost = sum(
            max(0, opt_result['grid_power_kw'][ts]) * import_price.get(ts, 0) * dt_hours
            for ts in optimizer.time_index
        )
        export_revenue = sum(
            max(0, -opt_result['grid_power_kw'][ts]) * export_price.get(ts, 0) * dt_hours
            for ts in optimizer.time_index
        )

        cost_breakdown = CostBreakdown(
            total_cost=opt_result['total_cost'],
            import_cost=import_cost,
            export_revenue=export_revenue,
            import_energy_kwh=import_energy,
            export_energy_kwh=export_energy
        )

        result = OptimizeResult(
            battery=battery_result,
            deferrable_loads=deferrable_schedules,
            cost=cost_breakdown
        )

        diagnostics = Diagnostics(
            solver_status=opt_result['solver_status'],  # "Infeasible", "Unbounded", etc.
            solve_time_ms=opt_result['solve_time_ms'],
            is_optimal=opt_result.get('fallback_reason') is None,  # True if no fallback
            fallback_used=opt_result.get('fallback_reason') is not None,  # True if fallback
            warnings=[
                f"Optimization failed: {opt_result.get('fallback_reason', 'Unknown')}",
                "Using safe fallback schedule (no battery action, no deferrable loads)",
                "Check constraints and input data for conflicts"
            ] if opt_result.get('fallback_reason') else warnings
        )

        return OptimizeResponse(
            status="ok",
            result=result,
            diagnostics=diagnostics
        )

    except Exception as e:
        # Log full traceback for debugging
        _LOGGER.error(f"Unexpected error: {traceback.format_exc()}")
        return OptimizeResponse(
            status="error",
            result=None,
            diagnostics=Diagnostics(
                solver_status="error",
                solve_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                is_optimal=False,  # Error means not optimal
                fallback_used=False,  # Error, not fallback
                warnings=warnings
            ),
            error=f"Internal error: {str(e)}"
        )
