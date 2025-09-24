#!/usr/bin/env bash
# profile.sh
# Usage: ./profile.sh path_to_data

set -e  # exit on any error

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path_to_data"
    exit 1
fi

DATA_PATH="$1"
CALLGRIND_FILE="call_grind.txt"

# Run the benchmark under callgrind
valgrind --tool=callgrind --callgrind-out-file="$CALLGRIND_FILE" ./build/benchmark_runner "$DATA_PATH"

# Launch kcachegrind to visualize the profiling results
kcachegrind "$CALLGRIND_FILE"
