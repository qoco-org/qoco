/**
 * @file qoco_batch.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2026, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "qoco_api.h"

#if !defined(IS_WINDOWS)
#include <pthread.h>
#endif

static void qoco_batch_init_empty(QOCOBatchSolver* batch)
{
  if (!batch) {
    return;
  }

  batch->batch_size = 0;
  batch->solvers = NULL;
  batch->statuses = NULL;
  batch->setup_time_sec = 0.0;
  batch->solve_time_sec = 0.0;
}

static QOCOInt qoco_batch_valid_item(QOCOBatchSolver* batch, QOCOInt item)
{
  return batch && batch->solvers && item >= 0 && item < batch->batch_size &&
         batch->solvers[item];
}

static void qoco_batch_set_solver_stream(QOCOSolver* solver)
{
  if (solver && solver->linsys && solver->linsys->linsys_set_active_stream) {
    solver->linsys->linsys_set_active_stream(solver->linsys_data);
  }
}

static void qoco_batch_clear_solver_stream(QOCOSolver* solver)
{
  if (solver && solver->linsys && solver->linsys->linsys_clear_active_stream) {
    solver->linsys->linsys_clear_active_stream(solver->linsys_data);
  }
}

static QOCOInt qoco_batch_solve_one(QOCOBatchSolver* batch, QOCOInt item)
{
  QOCOSolver* solver = batch->solvers[item];
  qoco_batch_set_solver_stream(solver);
  QOCOInt status = qoco_solve(solver);
  qoco_synchronize_current_stream();
  qoco_batch_clear_solver_stream(solver);
  batch->statuses[item] = status;
  return status;
}

QOCOInt qoco_batch_setup(QOCOBatchSolver* batch, QOCOInt batch_size,
                         QOCOInt n, QOCOInt m, QOCOInt p, QOCOCscMatrix* P,
                         QOCOFloat* c, QOCOCscMatrix* A, QOCOFloat* b,
                         QOCOCscMatrix* G, QOCOFloat* h, QOCOInt l,
                         QOCOInt nsoc, QOCOInt* q, QOCOSettings* settings)
{
  qoco_batch_init_empty(batch);

  if (!batch || batch_size <= 0) {
    return QOCO_DATA_VALIDATION_ERROR;
  }

#ifndef QOCO_ALGEBRA_BACKEND_CUDA
  (void)n;
  (void)m;
  (void)p;
  (void)P;
  (void)c;
  (void)A;
  (void)b;
  (void)G;
  (void)h;
  (void)l;
  (void)nsoc;
  (void)q;
  (void)settings;
  return QOCO_UNSUPPORTED_ERROR;
#else
  QOCOTimer setup_timer;
  start_timer(&setup_timer);

  batch->batch_size = batch_size;
  batch->solvers = (QOCOSolver**)qoco_calloc(batch_size, sizeof(QOCOSolver*));
  batch->statuses = (QOCOInt*)qoco_malloc(batch_size * sizeof(QOCOInt));
  if (!batch->solvers || !batch->statuses) {
    qoco_batch_cleanup(batch);
    return QOCO_MALLOC_ERROR;
  }

  for (QOCOInt i = 0; i < batch_size; ++i) {
    batch->statuses[i] = QOCO_UNSOLVED;
  }

  for (QOCOInt i = 0; i < batch_size; ++i) {
    batch->solvers[i] = (QOCOSolver*)qoco_malloc(sizeof(QOCOSolver));
    if (!batch->solvers[i]) {
      qoco_batch_cleanup(batch);
      return QOCO_MALLOC_ERROR;
    }

    QOCOInt exit = qoco_setup(batch->solvers[i], n, m, p, P, c, A, b, G, h, l,
                              nsoc, q, settings);
    if (exit != QOCO_NO_ERROR) {
      QOCOSolver* failed_solver = batch->solvers[i];
      batch->solvers[i] = NULL;
      qoco_free(failed_solver);
      qoco_batch_cleanup(batch);
      return exit;
    }
  }

  stop_timer(&setup_timer);
  batch->setup_time_sec = get_elapsed_time_sec(&setup_timer);
  return QOCO_NO_ERROR;
#endif
}

QOCOInt qoco_batch_update_vector_data(QOCOBatchSolver* batch, QOCOInt item,
                                      QOCOFloat* cnew, QOCOFloat* bnew,
                                      QOCOFloat* hnew)
{
  if (!qoco_batch_valid_item(batch, item)) {
    return QOCO_DATA_VALIDATION_ERROR;
  }

  QOCOSolver* solver = batch->solvers[item];
  qoco_batch_set_solver_stream(solver);
  qoco_update_vector_data(solver, cnew, bnew, hnew);
  qoco_synchronize_current_stream();
  qoco_batch_clear_solver_stream(solver);
  batch->statuses[item] = QOCO_UNSOLVED;
  return QOCO_NO_ERROR;
}

QOCOInt qoco_batch_set_x0(QOCOBatchSolver* batch, QOCOInt item,
                          const QOCOFloat* x0)
{
  if (!qoco_batch_valid_item(batch, item)) {
    return QOCO_DATA_VALIDATION_ERROR;
  }

  QOCOSolver* solver = batch->solvers[item];
  qoco_batch_set_solver_stream(solver);
  qoco_set_x0(solver, x0);
  qoco_synchronize_current_stream();
  qoco_batch_clear_solver_stream(solver);
  batch->statuses[item] = QOCO_UNSOLVED;
  return QOCO_NO_ERROR;
}

QOCOInt qoco_batch_update_matrix_data(QOCOBatchSolver* batch, QOCOInt item,
                                      QOCOFloat* Pxnew, QOCOFloat* Axnew,
                                      QOCOFloat* Gxnew)
{
  if (!qoco_batch_valid_item(batch, item)) {
    return QOCO_DATA_VALIDATION_ERROR;
  }

  QOCOSolver* solver = batch->solvers[item];
  qoco_batch_set_solver_stream(solver);
  qoco_update_matrix_data(solver, Pxnew, Axnew, Gxnew);
  qoco_synchronize_current_stream();
  qoco_batch_clear_solver_stream(solver);
  batch->statuses[item] = QOCO_UNSOLVED;
  return QOCO_NO_ERROR;
}

#if !defined(IS_WINDOWS)
typedef struct {
  QOCOBatchSolver* batch;
  QOCOInt item;
} QOCOBatchThreadArg;

static void* qoco_batch_thread_main(void* arg)
{
  QOCOBatchThreadArg* thread_arg = (QOCOBatchThreadArg*)arg;
  qoco_batch_solve_one(thread_arg->batch, thread_arg->item);
  return NULL;
}
#endif

QOCOInt qoco_batch_solve(QOCOBatchSolver* batch)
{
  if (!batch || !batch->solvers || !batch->statuses || batch->batch_size <= 0) {
    return QOCO_DATA_VALIDATION_ERROR;
  }

  QOCOTimer solve_timer;
  start_timer(&solve_timer);

#if defined(IS_WINDOWS)
  for (QOCOInt i = 0; i < batch->batch_size; ++i) {
    qoco_batch_solve_one(batch, i);
  }
#else
  pthread_t* threads =
      (pthread_t*)qoco_malloc(batch->batch_size * sizeof(pthread_t));
  QOCOBatchThreadArg* args = (QOCOBatchThreadArg*)qoco_malloc(
      batch->batch_size * sizeof(QOCOBatchThreadArg));

  if (!threads || !args) {
    qoco_free(threads);
    qoco_free(args);
    for (QOCOInt i = 0; i < batch->batch_size; ++i) {
      qoco_batch_solve_one(batch, i);
    }
  }
  else {
    QOCOInt launched = 0;
    for (; launched < batch->batch_size; ++launched) {
      args[launched].batch = batch;
      args[launched].item = launched;
      if (pthread_create(&threads[launched], NULL, qoco_batch_thread_main,
                         &args[launched]) != 0) {
        break;
      }
    }

    for (QOCOInt i = launched; i < batch->batch_size; ++i) {
      qoco_batch_solve_one(batch, i);
    }

    for (QOCOInt i = 0; i < launched; ++i) {
      pthread_join(threads[i], NULL);
    }

    qoco_free(threads);
    qoco_free(args);
  }
#endif

  stop_timer(&solve_timer);
  batch->solve_time_sec = get_elapsed_time_sec(&solve_timer);
  return QOCO_NO_ERROR;
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
    return QOCO_DATA_VALIDATION_ERROR;
  }

  if (batch->solvers) {
    for (QOCOInt i = 0; i < batch->batch_size; ++i) {
      if (batch->solvers[i]) {
        qoco_cleanup(batch->solvers[i]);
      }
    }
  }

  qoco_free(batch->solvers);
  qoco_free(batch->statuses);
  qoco_batch_init_empty(batch);
  return QOCO_NO_ERROR;
}
