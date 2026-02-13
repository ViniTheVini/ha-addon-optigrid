"""
Energy optimization using linear programming.
Optimizes battery schedule and deferrable load scheduling to minimize cost.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pulp
import csv
import logging
import os
from .decomposition import TimeSeries
from .time_alignment import resample_forward_fill

# Logging is configured by main.py
_LOGGER = logging.getLogger(__name__)

class OptimizationError(Exception):
    """Raised when optimization fails"""
    pass


class EnergyOptimizer:
    """
    Linear programming optimizer for home energy management.
    
    Minimizes total energy cost by optimizing:
    - Battery charge/discharge schedule
    - Deferrable load start times
    
    Subject to:
    - Power balance constraints
    - Battery SOC and power limits
    - Grid import/export limits
    - Deferrable load time windows
    """
    
    def __init__(
        self,
        timestep_minutes: int,
        horizon_steps: int,
        start_time: datetime
    ):
        self.dt = timestep_minutes / 60.0  # Convert to hours
        self.horizon_steps = horizon_steps
        self.start_time = start_time
        self.time_index = [
            start_time + timedelta(minutes=timestep_minutes * i)
            for i in range(horizon_steps)
        ]
        
    def optimize(
        self,
        # Forecasts (kW)
        uncontrolled_load_kw: TimeSeries,
        pv_forecast_kw: TimeSeries,
        import_price: TimeSeries,  # €/kWh
        export_price: TimeSeries,  # €/kWh
        
        # Battery config
        battery_enabled: bool,
        battery_capacity_kwh: Optional[float],
        battery_soc_init: Optional[float],
        battery_min_soc: Optional[float],
        battery_max_soc: Optional[float],
        battery_max_charge_kw: Optional[float],
        battery_max_discharge_kw: Optional[float],
        battery_efficiency: Optional[float],
        battery_degradation_cost_per_kwh: Optional[float],

        # Grid limits
        grid_max_import_kw: Optional[float] = None,
        grid_max_export_kw: Optional[float] = None,
        
        # Deferrable loads
        deferrable_loads: Optional[Dict[str, dict]] = None
    ) -> dict:
        """
        Run optimization.
        
        Returns:
            {
                'battery_power_kw': {ts: power},  # Positive = discharge
                'battery_soc': {ts: soc},
                'grid_power_kw': {ts: power},  # Positive = import
                'deferrable_schedules': {name: {'start': ts, 'end': ts}},
                'total_cost': float,
                'solver_status': str,
                'solve_time_ms': float
            }
        """
        # Validate price data (required for cost optimization)
        if not import_price or not export_price:
            message = "Price data is required for cost optimization. Both import_price and export_price must be provided."
            _LOGGER.error(message)
            raise OptimizationError(message)

        # Align price data to optimization time grid using forward-fill interpolation
        # This handles timestamp mismatches between price data and optimization grid
        try:
            import_price_aligned = resample_forward_fill(
                import_price,
                self.time_index,
                default=None  # Will raise error if no data available
            )
            export_price_aligned = resample_forward_fill(
                export_price,
                self.time_index,
                default=None  # Will raise error if no data available
            )
        except Exception as e:
            message = f"Failed to align price data to optimization time grid: {str(e)}"
            _LOGGER.error(message)
            raise OptimizationError(message)

        # Use aligned price data for the rest of the optimization
        import_price = import_price_aligned
        export_price = export_price_aligned

        # Create LP problem
        prob = pulp.LpProblem("EnergyOptimization", pulp.LpMinimize)

        # Check if PV is enabled (has non-zero forecast)
        pv_enabled = bool(pv_forecast_kw) and any(v > 0 for v in pv_forecast_kw.values())

        # Decision variables
        # Battery power: positive = discharge, negative = charge
        if battery_enabled:
            P_bat = pulp.LpVariable.dicts("P_bat", self.time_index, lowBound=-battery_max_charge_kw, upBound=battery_max_discharge_kw)
            # Battery SOC
            SOC = pulp.LpVariable.dicts("SOC", self.time_index, lowBound=battery_min_soc, upBound=battery_max_soc)

            # Always create auxiliary variables for charge and discharge
            # This is required for correct efficiency modeling in SOC dynamics
            # P_bat = P_bat_discharge - P_bat_charge
            # where P_bat_charge >= 0 (charging power magnitude)
            #       P_bat_discharge >= 0 (discharging power magnitude)
            P_bat_charge = pulp.LpVariable.dicts("P_bat_charge", self.time_index, lowBound=0, upBound=battery_max_charge_kw)
            P_bat_discharge = pulp.LpVariable.dicts("P_bat_discharge", self.time_index, lowBound=0, upBound=battery_max_discharge_kw)

            # Defensive constraint: If battery is at or very close to max SOC, prevent charging
            # Use a small tolerance (0.5%) to account for numerical precision
            soc_tolerance = 0.005
            if battery_soc_init >= battery_max_soc - soc_tolerance:
                _LOGGER.info(
                    f"Battery SOC ({battery_soc_init:.3f}) is at or near max_soc ({battery_max_soc:.3f}). "
                    f"Preventing charging for all timesteps."
                )
                # Force charging power to zero for all timesteps
                for t in self.time_index:
                    prob += P_bat_charge[t] == 0
        else:
            # Battery disabled: create dummy variables fixed at 0
            P_bat = {t: 0 for t in self.time_index}
            SOC = {t: battery_soc_init for t in self.time_index}

        # Grid power: positive = import, negative = export
        grid_import_max = grid_max_import_kw if grid_max_import_kw else 1e6
        grid_export_max = grid_max_export_kw if grid_max_export_kw else 1e6
        P_grid = pulp.LpVariable.dicts("P_grid", self.time_index, lowBound=-grid_export_max, upBound=grid_import_max)
        
        # Split grid power into import and export for cost calculation
        P_import = pulp.LpVariable.dicts("P_import", self.time_index, lowBound=0)
        P_export = pulp.LpVariable.dicts("P_export", self.time_index, lowBound=0)
        
        # Deferrable load variables
        deferrable_vars = {}
        if deferrable_loads:
            for name, config in deferrable_loads.items():
                # Binary variable: 1 if load is running at time t
                deferrable_vars[name] = pulp.LpVariable.dicts(
                    f"def_{name}",
                    self.time_index,
                    cat='Binary'
                )

        # Objective: minimize total cost including battery degradation cost
        # Prices are validated above, so we can safely access them
        if battery_enabled and battery_degradation_cost_per_kwh > 0:
            # Modified objective with degradation cost
            total_cost_expr = pulp.lpSum([
                P_import[t] * import_price[t] * self.dt  # Import cost
                - P_export[t] * export_price[t] * self.dt  # Export revenue
                + (P_bat_charge[t] + P_bat_discharge[t]) * battery_degradation_cost_per_kwh * self.dt  # Degradation cost
                for t in self.time_index
            ])
        else:
            # No battery
            total_cost_expr = pulp.lpSum([
                P_import[t] * import_price[t] * self.dt  # Import cost
                - P_export[t] * export_price[t] * self.dt  # Export revenue
                for t in self.time_index
            ])
        
        # Constraints
        for i, t in enumerate(self.time_index):
            # Get forecasts
            load_unc = uncontrolled_load_kw.get(t, 0)
            pv = pv_forecast_kw.get(t, 0) if pv_enabled else 0

            # Deferrable load power at this timestep
            def_load_power = 0
            if deferrable_loads:
                for name, config in deferrable_loads.items():
                    def_load_power += deferrable_vars[name][t] * config['power_kw']

            # Power balance: load = pv + battery_discharge + grid_import
            # Rearranged: grid = load - pv - battery_discharge
            # load_total = uncontrolled + deferrable
            # battery: positive P_bat = discharge (provides power)
            # pv: reduces net grid demand (can enable export if > load)
            if battery_enabled:
                prob += P_grid[t] == load_unc + def_load_power - pv - P_bat[t]

                # Link P_bat with charge/discharge components (always needed for correct SOC dynamics)
                prob += P_bat[t] == P_bat_discharge[t] - P_bat_charge[t]
            else:
                # No battery: grid must supply all net load (after PV)
                prob += P_grid[t] == load_unc + def_load_power - pv

            # Grid power split
            prob += P_grid[t] == P_import[t] - P_export[t]

            # Battery SOC dynamics (only if battery enabled)
            if battery_enabled:
                if i == 0:
                    # Initial SOC
                    prob += SOC[t] == battery_soc_init
                else:
                    t_prev = self.time_index[i-1]
                    # SOC dynamics with correct efficiency modeling:
                    # - When charging: SOC increases by charge_power * efficiency * dt / capacity
                    #   (energy is lost during charging, so we multiply by efficiency < 1)
                    # - When discharging: SOC decreases by discharge_power / efficiency * dt / capacity
                    #   (more energy is drawn from battery than delivered, so we divide by efficiency)
                    #
                    # Using auxiliary variables P_bat_charge and P_bat_discharge:
                    # SOC[t] = SOC[t-1] + P_bat_charge * efficiency * dt / capacity - P_bat_discharge / efficiency * dt / capacity
                    # Rearranged: SOC[t] = SOC[t-1] - (P_bat_discharge / efficiency - P_bat_charge * efficiency) * dt / capacity
                    #
                    # Note: PuLP doesn't support dividing LpVariable by float, so we multiply by reciprocal
                    efficiency_reciprocal = 1.0 / battery_efficiency

                    prob += SOC[t] == SOC[t_prev] - (P_bat_discharge[t_prev] * efficiency_reciprocal - P_bat_charge[t_prev] * battery_efficiency) * self.dt / battery_capacity_kwh
        
        # Deferrable load constraints
        if deferrable_loads:
            for name, config in deferrable_loads.items():
                duration_steps = int(config['duration_hours'] / self.dt)

                # Must run for exactly duration_steps timesteps
                prob += pulp.lpSum([deferrable_vars[name][t] for t in self.time_index]) == duration_steps

                # Time window constraints
                earliest = config.get('earliest_start')
                latest = config.get('latest_end')

                if earliest or latest:
                    for t in self.time_index:
                        if earliest and t < earliest:
                            prob += deferrable_vars[name][t] == 0
                        if latest and t > latest:
                            prob += deferrable_vars[name][t] == 0
                
                # Continuity penalty: discourage fragmentation by adding cost for each start
                # This prevents pathological fragmentation (many tiny windows) while allowing
                # economically justified splits (e.g., avoiding price spikes).
                # The penalty is added to the objective function as a soft constraint.
                penalty_per_start = config.get('continuity_penalty', 0.10)

                if penalty_per_start > 0:
                    # Track starts using auxiliary binary variables
                    # start_var[t] = 1 if load transitions from off to on at timestep t
                    start_var = pulp.LpVariable.dicts(f"start_{name}", self.time_index, cat='Binary')

                    for i, t in enumerate(self.time_index):
                        if i == 0:
                            # At first timestep, start_var = 1 if load is running
                            prob += start_var[t] >= deferrable_vars[name][t]
                        else:
                            t_prev = self.time_index[i-1]
                            # start_var[t] = 1 if load starts at timestep t
                            # This happens when: running now AND not running before
                            prob += start_var[t] >= deferrable_vars[name][t] - deferrable_vars[name][t_prev]

                    # Add penalty to objective function
                    # Higher penalty → stronger preference for continuous operation
                    # Lower penalty → more flexibility to split across windows
                    total_cost_expr += penalty_per_start * pulp.lpSum([start_var[t] for t in self.time_index])

        # Set objective function (after adding any continuity penalties)
        prob += total_cost_expr

        # Solve

        def _create_fallback_result(self, reason: str) -> dict:
            """Create safe fallback when optimization fails"""
            return {
                'battery_power_kw': {t: 0.0 for t in self.time_index},
                'battery_soc': {t: battery_soc_init for t in self.time_index},
                'grid_power_kw': {t: uncontrolled_load_kw.get(t, 0) - pv_forecast_kw.get(t, 0) 
                                for t in self.time_index},
                'deferrable_schedules': {},  # Don't schedule any loads
                'total_cost': self._calculate_baseline_cost(...),
                'solver_status': pulp.LpStatus[status],
                'solve_time_ms': solve_time_ms,
                'fallback_reason': reason
            }
        
        import time
        start_solve = time.time()
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        solve_time_ms = (time.time() - start_solve) * 1000
        
        if status != pulp.LpStatusOptimal:
            message = f"Optimization failed with status: {pulp.LpStatus[status]}"
            _LOGGER.error(message)

            return self._create_fallback_result(
                reason=f"Solver returned {pulp.LpStatus[status]}"
            )

        # Extract results
        if battery_enabled:
            battery_power = {t: P_bat[t].varValue for t in self.time_index}
            battery_soc = {t: SOC[t].varValue for t in self.time_index}
        else:
            # Battery disabled: return zeros
            battery_power = {t: 0.0 for t in self.time_index}
            battery_soc = {t: battery_soc_init for t in self.time_index}

        grid_power = {t: P_grid[t].varValue for t in self.time_index}
        
        deferrable_schedules = {}
        if deferrable_loads:
            for name in deferrable_loads:
                active_timesteps = sorted([t for t in self.time_index
                                   if deferrable_vars[name][t].varValue > 0.5])
                if active_timesteps:
                    # Group consecutive timesteps into windows
                    windows = []
                    window_start = active_timesteps[0]
                    prev_t = active_timesteps[0]

                    for t in active_timesteps[1:]:
                        # Check if this timestep is consecutive
                        if (t - prev_t).total_seconds() / 60 > self.dt * 60 * 1.1:  # Allow small tolerance
                            # Gap detected - close current window and start new one
                            windows.append({
                                'start': window_start,
                                'end': prev_t + timedelta(minutes=self.dt * 60)
                            })
                            window_start = t
                        prev_t = t

                    # Close final window
                    windows.append({
                        'start': window_start,
                        'end': prev_t + timedelta(minutes=self.dt * 60)
                    })

                    # Check if load is active at the start time (first timestep)
                    is_active_now = self.time_index[0] in active_timesteps

                    deferrable_schedules[name] = {
                        'windows': windows,
                        'total_duration_hours': len(active_timesteps) * self.dt,
                        'is_active_now': is_active_now
                    }
                
        # Get total cost from objective function
        total_cost = pulp.value(prob.objective)
        if total_cost is None:
            message = "Failed to calculate total cost from optimization result"
            _LOGGER.error(message)
            raise OptimizationError(message)

        # Export CSV for troubleshooting (only in debug mode)
        if _LOGGER.isEnabledFor(logging.DEBUG):
            self._export_debug_csv(
                uncontrolled_load_kw=uncontrolled_load_kw,
                pv_forecast_kw=pv_forecast_kw,
                battery_soc=battery_soc,
                battery_power=battery_power,
                import_price=import_price,
                export_price=export_price,
                grid_power=grid_power,
                pv_enabled=pv_enabled,
                deferrable_loads=deferrable_loads,
                deferrable_vars=deferrable_vars
            )

        return {
            'battery_power_kw': battery_power,
            'battery_soc': battery_soc,
            'grid_power_kw': grid_power,
            'deferrable_schedules': deferrable_schedules,
            'total_cost': total_cost,
            'solver_status': pulp.LpStatus[status],
            'solve_time_ms': solve_time_ms
        }

    def _export_debug_csv(
        self,
        uncontrolled_load_kw: TimeSeries,
        pv_forecast_kw: TimeSeries,
        battery_soc: Dict[datetime, float],
        battery_power: Dict[datetime, float],
        import_price: TimeSeries,
        export_price: TimeSeries,
        grid_power: Dict[datetime, float],
        pv_enabled: bool,
        deferrable_loads: Optional[Dict[str, dict]] = None,
        deferrable_vars: Optional[Dict[str, dict]] = None
    ):
        """
        Export optimization results to CSV for troubleshooting.

        Outputs per timestep:
        - timestamp
        - forecast_house_load_kw
        - forecast_pv_kw
        - forecast_battery_soc
        - forecast_battery_power_kw
        - energy_import_cost_eur_per_kwh
        - energy_export_cost_eur_per_kwh
        - grid_power_kw (positive = import, negative = export)
        - deferrable_load_total_kw (total power from all deferrable loads)
        - deferrable_load_{name}_kw (power from each individual deferrable load)
        """

        csv_path = os.path.join(f"data/optimizer_debug.csv")

        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Build fieldnames dynamically based on deferrable loads
        fieldnames = [
            'timestamp',
            'forecast_house_load_kw',
            'forecast_pv_kw',
            'forecast_battery_soc',
            'forecast_battery_power_kw',
            'energy_import_cost_eur_per_kwh',
            'energy_export_cost_eur_per_kwh',
            'grid_power_kw',
            'deferrable_load_total_kw'
        ]

        # Add individual deferrable load columns
        deferrable_load_names = []
        if deferrable_loads:
            deferrable_load_names = sorted(deferrable_loads.keys())
            for name in deferrable_load_names:
                fieldnames.append(f'deferrable_load_{name}_kw')

        # Write CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for t in self.time_index:
                # Calculate deferrable load power at this timestep
                deferrable_total_kw = 0.0
                deferrable_individual = {}

                if deferrable_loads and deferrable_vars:
                    for name, config in deferrable_loads.items():
                        # Check if load is running at this timestep
                        is_running = deferrable_vars[name][t].varValue > 0.5 if deferrable_vars[name][t].varValue is not None else 0
                        power_kw = config['power_kw'] * is_running
                        deferrable_individual[name] = power_kw
                        deferrable_total_kw += power_kw

                row = {
                    'timestamp': t.isoformat(),
                    'forecast_house_load_kw': uncontrolled_load_kw.get(t, 0.0),
                    'forecast_pv_kw': pv_forecast_kw.get(t, 0.0) if pv_enabled else 0.0,
                    'forecast_battery_soc': battery_soc.get(t, 0.0),
                    'forecast_battery_power_kw': battery_power.get(t, 0.0),
                    'energy_import_cost_eur_per_kwh': import_price.get(t, 0.0),
                    'energy_export_cost_eur_per_kwh': export_price.get(t, 0.0),
                    'grid_power_kw': grid_power.get(t, 0.0),
                    'deferrable_load_total_kw': deferrable_total_kw
                }

                # Add individual deferrable load columns
                for name in deferrable_load_names:
                    row[f'deferrable_load_{name}_kw'] = deferrable_individual.get(name, 0.0)

                writer.writerow(row)

        _LOGGER.debug(f"Debug CSV exported to: {csv_path}")

