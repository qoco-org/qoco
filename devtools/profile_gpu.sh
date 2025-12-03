#!/usr/bin/env bash
# profile_gpu.sh
# Usage: ./profile_gpu.sh path_to_data

set -e  # exit on any error

DATA_PATH="$1"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path_to_data"
    exit 1
fi

rm -f profile_report.nsys-rep profile_report.sqlite

nsys profile -o profile_report ./build/benchmark_runner "$DATA_PATH" verbose=1

nsys stats profile_report.nsys-rep > profile_report.txt
