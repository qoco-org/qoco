#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "qcos.h"
#include "gtest/gtest.h"

/**
 * @brief Utility function to extend EXPECT_NEAR to vectors.
 * Passes if ||x-y||_\infty < tol.
 *
 * @param x Input vector.
 * @param y Input vector.
 * @param n Length of vectors.
 * @param tol Tolerace.
 *
 */
void expect_eq_vectorf(QCOSFloat* x, QCOSFloat* y, QCOSInt n, QCOSFloat tol);

#endif /* #ifndef TEST_UTILS_H */