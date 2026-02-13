#!/usr/bin/env bash
set -e

# Default log level
LOG_LEVEL="info"

# Read log level from Home Assistant addon options if available
if [ -f /data/options.json ]; then
    LOG_LEVEL=$(python3 -c 'import json; print(json.load(open("/data/options.json")).get("log_level", "info"))' 2>/dev/null || echo "info")
fi

# Print the log level being used
echo "Starting OptiGrid with log level: ${LOG_LEVEL}"

# Start uvicorn with the configured log level
exec uvicorn main:app --host 0.0.0.0 --port 8000 --log-level "${LOG_LEVEL}"

