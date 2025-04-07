#ifndef QOCO_ENUMS_H
#define QOCO_ENUMS_H

/**
 * @brief Enum for error codes.
 *
 */
enum qoco_error_code {
  QOCO_NO_ERROR = 0,

  // Error in problem data validation.
  QOCO_DATA_VALIDATION_ERROR,

  // Error in settings validation.
  QOCO_SETTINGS_VALIDATION_ERROR,

  // Error in setup.
  QOCO_SETUP_ERROR,

  // Error in performing amd ordering.
  QOCO_AMD_ERROR,

  // Memory allocation error.
  QOCO_MALLOC_ERROR
};

/**
 * @brief Enum for solver status.
 *
 */
enum qoco_solve_status {
  // Unsolved (Solver needs to be called.)
  QOCO_UNSOLVED = 0,

  // Solved successfully.
  QOCO_SOLVED = 1,

  // Solved Inaccurately.
  QOCO_SOLVED_INACCURATE,

  // Numerical error (occurs when a = 0 and inaccurate stopping criteria not
  // met).
  QOCO_NUMERICAL_ERROR,

  // Maximum number of iterations reached.
  QOCO_MAX_ITER,
};

#endif /* #ifndef QOCO_ENUMS_H */