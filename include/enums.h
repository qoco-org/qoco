#ifndef ENUMS_H
#define ENUMS_H

/**
 * @brief Enum for error codes.
 *
 */
enum qcos_error_code {
  QCOS_NO_ERROR = 0,

  // Error in problem data validation.
  QCOS_DATA_VALIDATION_ERROR,

  // Error in settings validation.
  QCOS_SETTINGS_VALIDATION_ERROR,

  // Memory allocation error.
  QCOS_MALLOC_ERROR
};

/**
 * @brief Enum for solver status.
 *
 */
enum qcos_solve_status {
  // Solved successfully.
  QCOS_SOLVED = 1,

  // Maximum number of iterations reached.
  QCOS_MAX_ITER,

  // Time limit reached.
  QCOS_TIME_LIMIT
};

// clang-format off
static const char *QCOS_ERROR_MESSAGE[] = {
    "", // Error codes start from 1.
    "data validation error",
    "settings validation error",
    "memory allocation error"
};


static const char *QCOS_SOLVE_STATUS_MESSAGE[] = {
    "", // Solve status start from 1.
    "solved",
    "maximum iterations reached",
    "run time limit exceeded"
};
// clang-format on

#endif /* #ifndef ENUMS_H */