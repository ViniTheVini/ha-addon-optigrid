# OptiGrid Add-on Documentation

## About

OptiGrid is an energy optimization add-on that uses linear programming to optimize battery charging/discharging and deferrable load scheduling based on electricity prices and solar production forecasts.

## How to Use

### 1. Start the Add-on

After installation, start the add-on. It will run a REST API on port 8000.

### 2. Call the Optimization API

Use Home Assistant automations or REST commands to call the optimization endpoint:

**Endpoint:** `POST http://localhost:8000/optimize`

**Example payload:**

```json
{
  "time": {
    "start": "2024-02-11T10:00:00",
    "timestep_minutes": 15,
    "horizon_steps": 96
  },
  "config": {
    "battery": {
      "enabled": true,
      "capacity_kwh": 10.0,
      "current_soc": 0.5,
      "min_soc": 0.1,
      "max_soc": 0.9,
      "max_charge_kw": 5.0,
      "max_discharge_kw": 5.0,
      "efficiency": 0.95
    },
    "grid": {
      "power_sensor": "sensor.grid_power",
      "max_import_kw": 50.0,
      "max_export_kw": 10.0
    },
    "pv": {
      "enabled": true,
      "power_sensor": "sensor.solar_power"
    },
    "lookback_days": 7
  },
  "forecasts": {
    "grid_price": {
      "import_price": [0.20, 0.18, 0.15, ...],
      "export_price": [0.10, 0.08, 0.05, ...]
    }
  }
}
```

### 3. Process the Response

The API returns an optimization schedule with battery charge/discharge commands and load scheduling recommendations.

## Configuration Options

### Add-on Configuration

- **log_level**: Set logging verbosity (critical, error, warning, info, debug, trace)

### API Parameters

- **battery**: Battery configuration (capacity, SOC limits, power limits, efficiency)
- **grid**: Grid connection settings and sensor references
- **pv**: Solar PV configuration
- **deferrable_loads**: Loads that can be scheduled (EV charger, etc.)
- **forecasts**: Price and production forecasts

## Integration with Home Assistant

The add-on automatically connects to Home Assistant via the Supervisor API. You can:

1. Fetch historical sensor data automatically
2. Use REST commands to trigger optimization
3. Create automations based on time or price updates

## Troubleshooting

### Add-on won't start

- Check the logs for error messages
- Verify port 8000 is not in use by another service

### Optimization fails

- Ensure sensor names are correct
- Verify historical data is available
- Check that price forecasts are provided

### No results returned

- Increase log_level to "debug" to see detailed solver output
- Check that constraints are feasible (e.g., battery can meet requirements)

## Support

For issues and questions:

- GitHub Issues: https://github.com/ViniTheVini/ha-addon-optigrid/issues
- Documentation: https://github.com/ViniTheVini/ha-addon-optigrid
