#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "qoco.h"
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
void expect_eq_vectorf(QOCOFloat* x, QOCOFloat* y, QOCOInt n, QOCOFloat tol);

/**
 * @brief Utility function to extend EXPECT_EQ to QOCOCscMatrix.
 *
 * @param A Input matrix.
 * @param B Input matrix.
 * @param tol Tolerace.
 */
void expect_eq_csc(QOCOCscMatrix* A, QOCOCscMatrix* B, QOCOFloat tol);

/**
 * @brief Utility function to test if
 * |x-y|/|y| <= tol
 *
 * @param x Tested value.
 * @param y True value.
 * @param tol Tolerance.
 */
void expect_rel_error(QOCOFloat x, QOCOFloat y, QOCOFloat tol);

#endif /* #ifndef TEST_UTILS_H */