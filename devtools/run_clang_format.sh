#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$1" = "--check" ]; then
  find "$REPO_ROOT/src" "$REPO_ROOT/include" "$REPO_ROOT/algebra/builtin" \
    -name "*.c" -o -name "*.h" \
  | xargs clang-format --dry-run --Werror
else
  find "$REPO_ROOT/src" "$REPO_ROOT/include" "$REPO_ROOT/algebra/builtin" \
    -name "*.c" -o -name "*.h" \
  | xargs clang-format -i
fi
