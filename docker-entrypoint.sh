#!/bin/bash
set -e

# Docker entrypoint for sofia-phone
# Handles initialization and graceful shutdown

echo "========================================="
echo "Sofia-Phone Production Container"
echo "========================================="
echo "Environment: ${SOFIA_PHONE_ENV:-development}"
echo "Log Level: ${SOFIA_PHONE_LOG_LEVEL:-INFO}"
echo "ESL Port: ${SOFIA_PHONE_ESL_PORT:-8084}"
echo "Health Check Port: ${SOFIA_PHONE_HEALTH_PORT:-8080}"
echo "========================================="

# Create log directory
mkdir -p /app/logs

# Wait for FreeSWITCH if host is specified
if [ -n "$FREESWITCH_HOST" ]; then
    echo "Waiting for FreeSWITCH at $FREESWITCH_HOST:${FREESWITCH_PORT:-5060}..."

    for i in {1..30}; do
        if nc -z "$FREESWITCH_HOST" "${FREESWITCH_PORT:-5060}" 2>/dev/null; then
            echo "FreeSWITCH is ready!"
            break
        fi

        if [ $i -eq 30 ]; then
            echo "Warning: FreeSWITCH not reachable after 30 attempts"
        fi

        echo "Waiting... (attempt $i/30)"
        sleep 2
    done
fi

# Handle signals for graceful shutdown
_term() {
    echo "Received termination signal"
    kill -TERM "$child" 2>/dev/null
}

trap _term SIGTERM SIGINT

# Start application
echo "Starting sofia-phone..."
exec "$@" &

child=$!
wait "$child"
