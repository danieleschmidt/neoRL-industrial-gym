#!/bin/bash
REPO_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
cd "$REPO_ROOT"

case "$1" in
    "discovery")
        python3 .terragon/value_discovery_simple.py > /tmp/terragon-discovery.log 2>&1
        ;;
    "execution")
        python3 .terragon/autonomous_executor.py > /tmp/terragon-execution.log 2>&1
        ;;
    *)
        echo "Usage: $0 {discovery|execution}"
        exit 1
        ;;
esac
