#ifndef QCOS_API_H
#define QCOS_API_H

#include "definitions.h"
#include "kkt.h"
#include "linalg.h"
#include "structs.h"
#include "validation.h"

/*
 QCOS solves second-order cone problems of the following form
 minimize   (1/2)x'Px + c'x
 subject to Gx \leq_C h <==> h - Gx \in C
            Ax = b

 where C is the cartesian product of the non-negative orthant and second-order
 cones
*/

/**
 * @brief Allocates all memory needed for QCOS to solve the SOCP.
 *
 * @param P Upper triangular part of quadratic cost Hessian in CSC form
 * @param c Linear cost vector
 * @param A Affine equality constraint matrix in CSC form
 * @param b Affine equality constraint offset vector
 * @param G Conic constraint matrix in CSC form
 * @param h Conic constraint offset vector
 * @param l Dimension of non-negative orthant
 * @param ncones Number of second-order cones
 * @param q Dimension of each second-order cone
 * @param settings Settings struct
 * @return Pointer to solver
 */
QCOSSolver* qcos_setup(QCOSCscMatrix* P, QCOSFloat* c, QCOSCscMatrix* A,
                       QCOSFloat* b, QCOSCscMatrix* G, QCOSFloat* h, QCOSInt l,
                       QCOSInt ncones, QCOSInt* q, QCOSSettings* settings);

/**
 * @brief Sets the data for a compressed sparse column matrix.
 *
 * @param A Pointer to the CSC matrix
 * @param m Number of rows in the matrix
 * @param n Number of columns in the matrix
 * @param Annz Number of nonzero elements in the matrix
 * @param Ax Array of data for the matrix
 * @param Ap Array of column pointers for the data
 * @param Ai Array of row indices for data
 */
void qcos_set_csc(QCOSCscMatrix* A, QCOSInt m, QCOSInt n, QCOSInt Annz,
                  QCOSFloat* Ax, QCOSInt* Ap, QCOSInt* Ai);

/**
 * @brief Set the default settings struct
 *
 * @param settings Pointer to settings struct
 */
void set_default_settings(QCOSSettings* settings);

QCOSInt qcos_solve();

QCOSInt qcos_cleanup(QCOSSolver* solver);

#endif /* #ifndef QCOS_API_H */