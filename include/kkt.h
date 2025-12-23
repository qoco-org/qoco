/**
 * @file kkt.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides various functions for solving, constructing and updating KKT
 * systems.
 */

#ifndef QOCO_KKT_H
#define QOCO_KKT_H

#include "cone.h"
#include "qdldl.h"
#include "qoco_linalg.h"
#include "structs.h"

/**
 * @brief Constructs and returns the upper triangular portion of the KKT matrix
 in CSC form.
 *
 *     [ P + e*I      A^T       G^T    ]
 * K = | A         - e * I       0     |
 *     [ G             0        -I     ]

 * @param P P.
 * @param A A.
 * @param G G.
 * @param At A^T.
 * @param Gt G^T.
 * @param static_reg Static regularization parameter.
 * @param n Rows of P.
 * @param m Rows of G.
 * @param p Rows of A.
 * @param l Dimension of nonneg orthant.
 * @param nsoc Number of socs
 * @param q Dimension of socs
 * @param PregtoKKT Mapping from Preg scaling to KKT matrix.
 * @param AttoKKT Mapping from At scaling to KKT matrix.
 * @param GttoKKT Mapping from Gt scaling to KKT matrix.
 * @param nt2kkt Mapping from NT scaling to KKT matrix.
 * @param ntdiag2kkt Mapping from NT scaling diagonal to KKT matrix.
 * @param Wnnz Number of nonzeros in upper triangular portion of NT scaling.
 * @return QOCOCscMatrix*
 */
QOCOCscMatrix* construct_kkt(QOCOCscMatrix* P, QOCOCscMatrix* A,
                             QOCOCscMatrix* G, QOCOCscMatrix* At,
                             QOCOCscMatrix* Gt, QOCOFloat static_reg, QOCOInt n,
                             QOCOInt m, QOCOInt p, QOCOInt l, QOCOInt nsoc,
                             QOCOInt* q, QOCOInt* PregtoKKT, QOCOInt* AttoKKT,
                             QOCOInt* GttoKKT, QOCOInt* nt2kkt,
                             QOCOInt* ntdiag2kkt, QOCOInt Wnnz);

/**
 * @brief Gets initial values for primal and dual variables such that (s,z) \in
 * C.
 *
 * @param solver Pointer to solver.
 */
void initialize_ipm(QOCOSolver* solver);

/**
 * @brief Computes residual of KKT conditions and stores in work->kkt->rhs.
 *
 * clang-format off
 *
 *       [ P   A^T   G^T ] [ x ]   [    c   ]
 * res = | A    0     0  | | y ] + |   -b   |
 *       [ G    0     0  ] [ z ]   [ -h + s ]
 *
 * clang-format on
 * @param data Pointer to problem data
 * @param x_vec Primal iterate.
 * @param y_vec Dual iterate.
 * @param s_vec Slack iterate.
 * @param z_vec Dual iterate.
 * @param kktres_vec Computed residual is stored here.
 * @param static_reg Static regularization parameter.
 * @param xyzbuff_vec Buffer of length n+p+m.
 * @param nbuff_vec Buffer of length n.
 * @param mbuff_vec Buffer of length m.
 */
void compute_kkt_residual(QOCOProblemData* data, QOCOVectorf* x_vec,
                          QOCOVectorf* y_vec, QOCOVectorf* s_vec,
                          QOCOVectorf* z_vec, QOCOVectorf* kktres_vec,
                          QOCOFloat static_reg, QOCOVectorf* xyzbuff_vec,
                          QOCOVectorf* nbuff_vec, QOCOVectorf* mbuff_vec);

/**
 * @brief Computes mu = s'*z / m.
 *
 * @param s_vec slack iterate.
 * @param z_vec dual iterate.
 * @param m Length of s and z.
 * @return mu
 */
QOCOFloat compute_mu(QOCOVectorf* s_vec, QOCOVectorf* z_vec, QOCOInt m);

/**
 * @brief Computes the objective
 * obj = (1/2)*x'*P*x + c'*x
 *
 * @param data Pointer to problem data.
 * @param x_vec Primal solution at current iterate.
 * @param nbuff_vec Buffer of length n.
 * @param static_reg Static regularization value;
 * @param k Objective scaling from ruiz equilibration.
 * @return Computed objective
 */
QOCOFloat compute_objective(QOCOProblemData* data, QOCOVectorf* x_vec,
                            QOCOVectorf* nbuff_vec, QOCOFloat static_reg,
                            QOCOFloat k);
/**
 * @brief Constructs rhs for the affine scaling KKT system.
 * Before calling this function, work->kkt->kktres must contain the
 * residual of the KKT conditions as computed by compute_kkt_residual().
 *
 * @param work Pointer to workspace.
 */
void construct_kkt_aff_rhs(QOCOWorkspace* work);

/**
 * @brief Constructs rhs for the combined direction KKT system.
 * Before calling this function, work->kkt->kktres must contain the
 * negative residual of the KKT conditions as computed by
 * compute_kkt_residual().
 *
 * @param work Pointer to workspace.
 */
void construct_kkt_comb_rhs(QOCOWorkspace* work);

/**
 * @brief Performs Mehrotra predictor-corrector step.
 *
 * @param solver Pointer to solver.
 */
void predictor_corrector(QOCOSolver* solver);

/**
 * @brief Computes y = Kx where
 *     [ P   A^T       G^T      ]
 * K = | A    0         0       |
 *     [ G    0   -W'W - e * I  ]
 *
 * @param work Pointer to workspace.
 * @param x Pointer to input vector.
 * @param y Pointer to output vector.
 * @param data Pointer to problem data.
 * @param Wfull Pointer to full NT scaling matrix W.
 * @param nbuff Temporary buffer of length n.
 * @param mbuff1 Temporary buffer of length m.
 * @param mbuff2 Temporary buffer of length m.
 */
void kkt_multiply(QOCOFloat* x, QOCOFloat* y, QOCOProblemData* data,
                  QOCOFloat* Wfull, QOCOFloat* nbuff, QOCOFloat* mbuff1,
                  QOCOFloat* mbuff2);
#endif /* #ifndef QOCO_KKT_H */