#include "qoco_status.h"

QOCOInt qoco_error(enum qoco_error_code error_code)
{
  printf("ERROR: %s\n", QOCO_ERROR_MESSAGE[error_code]);
  return (QOCOInt)error_code;
}

// clang-format off
const char *QOCO_ERROR_MESSAGE[] = {
  "", // Error codes start from 1.
  "data validation error",
  "settings validation error",
  "amd error",
  "memory allocation error"
};

const char *QOCO_SOLVE_STATUS_MESSAGE[] = {
  "unsolved", // Solver not run.
  "solved",
  "solved inaccurately",
  "numerical error",
  "maximum iterations reached",
};
// clang-format on
