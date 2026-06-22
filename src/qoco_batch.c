/**
 * @file qoco_batch.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2026, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Implements the batched QOCO API.
 */

#include "qoco_batch.h"

#ifdef QOCO_ALGEBRA_BACKEND_CUDA
QOCOInt qoco_cuda_batch_setup(QOCOBatchSolver* batch);
QOCOInt qoco_cuda_batch_solve(QOCOBatchSolver* batch);
void qoco_cuda_batch_cleanup(QOCOBatchSolver* batch);
#endif

static unsigned char qoco_batch_valid_item(const QOCOBatchSolver* batch,
                                           QOCOInt item)
{
  return batch && item >= 0 && item < batch->batch_count && batch->solvers &&
         batch->solvers[item];
}

static void qoco_batch_zero(QOCOBatchSolver* batch)
{
  batch->batch_count = 0;
  batch->solvers = NULL;
  batch->statuses = NULL;
  batch->batch_linsys_data = NULL;
  batch->batch_linsys_stale = 0;
}

QOCOInt qoco_batch_setup(QOCOBatchSolver* batch, QOCOInt batch_count, QOCOInt n,
                         QOCOInt m, QOCOInt p, QOCOCscMatrix* P, QOCOFloat* c,
                         QOCOCscMatrix* A, QOCOFloat* b, QOCOCscMatrix* G,
                         QOCOFloat* h, QOCOInt l, QOCOInt nsoc, QOCOInt* q,
                         QOCOSettings* settings)
{
  if (!batch || batch_count <= 0) {
    return qoco_error(QOCO_DATA_VALIDATION_ERROR);
  }

  qoco_batch_zero(batch);
  batch->batch_count = batch_count;
  batch->solvers = (QOCOSolver**)qoco_calloc(batch_count, sizeof(QOCOSolver*));
  batch->statuses = (QOCOInt*)qoco_calloc(batch_count, sizeof(QOCOInt));
  if (!batch->solvers || !batch->statuses) {
    qoco_batch_cleanup(batch);
    return qoco_error(QOCO_MALLOC_ERROR);
  }

  for (QOCOInt item = 0; item < batch_count; ++item) {
    batch->solvers[item] = (QOCOSolver*)qoco_malloc(sizeof(QOCOSolver));
    if (!batch->solvers[item]) {
      qoco_batch_cleanup(batch);
      return qoco_error(QOCO_MALLOC_ERROR);
    }

    QOCOInt exit = qoco_setup(batch->solvers[item], n, m, p, P, c, A, b, G, h,
                              l, nsoc, q, settings);
    if (exit != QOCO_NO_ERROR) {
      qoco_free(batch->solvers[item]);
      batch->solvers[item] = NULL;
      qoco_batch_cleanup(batch);
      return exit;
    }
    batch->statuses[item] = QOCO_UNSOLVED;
  }

#ifdef QOCO_ALGEBRA_BACKEND_CUDA
  QOCOInt exit = qoco_cuda_batch_setup(batch);
  if (exit != QOCO_NO_ERROR) {
    qoco_batch_cleanup(batch);
    return exit;
  }
#endif

  return QOCO_NO_ERROR;
}

QOCOInt qoco_batch_update_vector_data(QOCOBatchSolver* batch, QOCOInt item,
                                      QOCOFloat* cnew, QOCOFloat* bnew,
                                      QOCOFloat* hnew)
{
  if (!qoco_batch_valid_item(batch, item)) {
    return qoco_error(QOCO_DATA_VALIDATION_ERROR);
  }

  qoco_update_vector_data(batch->solvers[item], cnew, bnew, hnew);
  batch->statuses[item] = QOCO_UNSOLVED;
  return QOCO_NO_ERROR;
}

QOCOInt qoco_batch_update_matrix_data(QOCOBatchSolver* batch, QOCOInt item,
                                      QOCOFloat* Pxnew, QOCOFloat* Axnew,
                                      QOCOFloat* Gxnew)
{
  if (!qoco_batch_valid_item(batch, item)) {
    return qoco_error(QOCO_DATA_VALIDATION_ERROR);
  }

  qoco_update_matrix_data(batch->solvers[item], Pxnew, Axnew, Gxnew);
  batch->statuses[item] = QOCO_UNSOLVED;
  return QOCO_NO_ERROR;
}

QOCOInt qoco_batch_solve(QOCOBatchSolver* batch)
{
  if (!batch || !batch->solvers || !batch->statuses) {
    return qoco_error(QOCO_DATA_VALIDATION_ERROR);
  }

#ifdef QOCO_ALGEBRA_BACKEND_CUDA
  return qoco_cuda_batch_solve(batch);
#else
  for (QOCOInt item = 0; item < batch->batch_count; ++item) {
    if (!batch->solvers[item]) {
      return qoco_error(QOCO_DATA_VALIDATION_ERROR);
    }
    batch->statuses[item] = qoco_solve(batch->solvers[item]);
  }
  return QOCO_NO_ERROR;
#endif
}

QOCOSolution* qoco_batch_get_solution(QOCOBatchSolver* batch, QOCOInt item)
{
  if (!qoco_batch_valid_item(batch, item)) {
    return NULL;
  }
  return batch->solvers[item]->sol;
}

QOCOInt qoco_batch_cleanup(QOCOBatchSolver* batch)
{
  if (!batch) {
    return QOCO_NO_ERROR;
  }

#ifdef QOCO_ALGEBRA_BACKEND_CUDA
  qoco_cuda_batch_cleanup(batch);
#endif

  if (batch->solvers) {
    for (QOCOInt item = 0; item < batch->batch_count; ++item) {
      if (batch->solvers[item]) {
        qoco_cleanup(batch->solvers[item]);
      }
    }
  }

  qoco_free(batch->solvers);
  qoco_free(batch->statuses);
  qoco_batch_zero(batch);
  return QOCO_NO_ERROR;
}
