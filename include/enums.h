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

  // Error in setup.
  QCOS_SETUP_ERROR,

  // Error in performing amd ordering.
  QCOS_AMD_ERROR,

  // Memory allocation error.
  QCOS_MALLOC_ERROR
};

/**
 * @brief Enum for solver status.
 *
 */
enum qcos_solve_status {
  // Unsolved (Solver needs to be called.)
  QCOS_UNSOLVED = 0,

  // Solved successfully.
  QCOS_SOLVED = 1,

  // Solved Inaccurately.
  QCOS_SOLVED_INACCURATE,

  // Numerical error (occurs when a = 0 and inaccurate stopping criteria not
  // met).
  QCOS_NUMERICAL_ERROR,

  // Maximum number of iterations reached.
  QCOS_MAX_ITER,
};

// clang-format off
static const char *QCOS_ERROR_MESSAGE[] = {
    "", // Error codes start from 1.
    "data validation error",
    "settings validation error",
    "amd error",
    "memory allocation error"
};


static const char *QCOS_SOLVE_STATUS_MESSAGE[] = {
    "unsolved", // Solver not run.
    "solved",
    "solved inaccurately",
    "numerical error",
    "maximum iterations reached",
};
// clang-format on

#endif /* #ifndef ENUMS_H */