#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build_tidy"

cmake -B "$BUILD_DIR" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DQOCO_BUILD_TYPE=Debug \
  -S "$REPO_ROOT" > /dev/null

find "$REPO_ROOT/src" -name "*.c" \
  ! -name "timer_macos.c" \
  ! -name "timer_windows.c" \
| xargs clang-tidy -p "$BUILD_DIR" --config-file="$REPO_ROOT/.clang-tidy"

rm -rf "$BUILD_DIR"
