"""
Home Assistant API client for fetching sensor data.
"""

import aiohttp
import logging
from datetime import datetime
from schemas import TimeSeries

_LOGGER = logging.getLogger(__name__)


async def fetch_sensor_history_from_ha(
    ha_url: str,
    ha_token: str,
    entity_id: str,
    start_time: datetime,
    end_time: datetime
) -> TimeSeries:
    """
    Fetch sensor history from Home Assistant API.
    
    Args:
        ha_url: Home Assistant URL (e.g., http://homeassistant.local:8123)
        ha_token: Long-lived access token
        entity_id: Sensor entity ID (e.g., sensor.house_load)
        start_time: Start of history period
        end_time: End of history period
    
    Returns:
        TimeSeries dict {datetime: float}
    
    Raises:
        aiohttp.ClientError: If request fails
        ValueError: If data is invalid
    """
    
    # Build API URL
    # Format: /api/history/period/<start_time>?filter_entity_id=<entity_id>&end_time=<end_time>&minimal_response=true
    start_iso = start_time.isoformat()
    end_iso = end_time.isoformat()
    
    url = f"{ha_url}/api/history/period/{start_iso}"
    params = {
        "filter_entity_id": entity_id,
        "end_time": end_iso,
        "minimal_response": "true"
    }
    
    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json"
    }
    
    _LOGGER.debug(f"Fetching history for {entity_id} from {start_time} to {end_time}.")
    _LOGGER.debug(f"URL: {url}\nPARAMS: {params}\nHEADERS: {headers}")
    
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
    
    # Parse response
    # Response format: [[{state, last_changed, ...}, ...]]
    if not data or not isinstance(data, list) or len(data) == 0:
        _LOGGER.warning(f"No history data returned for {entity_id}")
        return {}
    
    history_points = data[0]  # First element is the entity's history
    
    # Convert to TimeSeries
    timeseries: TimeSeries = {}
    
    for point in history_points:
        state_str = point.get("state")
        last_changed_str = point.get("last_changed")
        
        if state_str in ["unknown", "unavailable", None]:
            continue
        
        try:
            value = float(state_str)
            timestamp = datetime.fromisoformat(last_changed_str.replace("Z", "+00:00"))
            timeseries[timestamp] = value
        except (ValueError, TypeError, AttributeError) as e:
            _LOGGER.warning(f"Could not parse data point for {entity_id}: {e}")
            continue
    
    _LOGGER.info(f"Fetched {len(timeseries)} data points for {entity_id}")

    return timeseries
