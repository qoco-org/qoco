/**
 * @file qcos_api.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 */

#include "qcos_api.h"
#include "amd.h"

QCOSInt qcos_setup(QCOSSolver* solver, QCOSInt n, QCOSInt m, QCOSInt p,
                   QCOSCscMatrix* P, QCOSFloat* c, QCOSCscMatrix* A,
                   QCOSFloat* b, QCOSCscMatrix* G, QCOSFloat* h, QCOSInt l,
                   QCOSInt nsoc, QCOSInt* q, QCOSSettings* settings)
{
  // Start setup timer.
  QCOSTimer setup_timer;
  start_timer(&setup_timer);

  // Validate problem data.
  if (qcos_validate_data(P, c, A, b, G, h, l, nsoc, q)) {
    return qcos_error(QCOS_DATA_VALIDATION_ERROR);
  }

  // Validate settings.
  if (qcos_validate_settings(settings)) {
    return qcos_error(QCOS_SETTINGS_VALIDATION_ERROR);
  }

  solver->settings = copy_settings(settings);

  // Allocate workspace.
  solver->work = qcos_malloc(sizeof(QCOSWorkspace));

  // Malloc error.
  if (!(solver->work)) {
    return QCOS_MALLOC_ERROR;
  }

  solver->work->data = qcos_malloc(sizeof(QCOSProblemData));
  // Malloc error
  if (!(solver->work->data)) {
    return QCOS_MALLOC_ERROR;
  }

  // Copy problem data.
  solver->work->data->m = m;
  solver->work->data->n = n;
  solver->work->data->p = p;
  solver->work->data->A = new_qcos_csc_matrix(A);
  solver->work->data->G = new_qcos_csc_matrix(G);
  solver->work->data->c = qcos_malloc(n * sizeof(QCOSFloat));
  solver->work->data->b = qcos_malloc(p * sizeof(QCOSFloat));
  solver->work->data->h = qcos_malloc(m * sizeof(QCOSFloat));
  copy_arrayf(c, solver->work->data->c, n);
  copy_arrayf(b, solver->work->data->b, p);
  copy_arrayf(h, solver->work->data->h, m);
  solver->work->data->q = q;
  solver->work->data->l = l;
  solver->work->data->nsoc = nsoc;

  // Copy P.
  if (P) {
    solver->work->data->P = new_qcos_csc_matrix(P);
  }
  else {
    solver->work->data->P = NULL;
  }

  // Equilibrate data.
  solver->work->kkt = qcos_malloc(sizeof(QCOSKKT));
  solver->work->kkt->delta = qcos_malloc((n + p + m) * sizeof(QCOSFloat));
  solver->work->kkt->Druiz = qcos_malloc(n * sizeof(QCOSFloat));
  solver->work->kkt->Eruiz = qcos_malloc(p * sizeof(QCOSFloat));
  solver->work->kkt->Fruiz = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->kkt->Dinvruiz = qcos_malloc(n * sizeof(QCOSFloat));
  solver->work->kkt->Einvruiz = qcos_malloc(p * sizeof(QCOSFloat));
  solver->work->kkt->Finvruiz = qcos_malloc(m * sizeof(QCOSFloat));
  ruiz_equilibration(solver);

  // Regularize P.
  solver->work->kkt->Pnzadded_idx = qcos_calloc(n, sizeof(QCOSInt));
  if (P) {
    solver->work->kkt->Pnum_nzadded =
        regularize(solver->work->data->P, solver->settings->reg,
                   solver->work->kkt->Pnzadded_idx);
  }
  else {
    solver->work->data->P = construct_identity(n, solver->settings->reg);
    solver->work->kkt->Pnum_nzadded = n;
  }

  // Allocate KKT struct.
  allocate_kkt(solver->work);
  solver->work->kkt->nt2kkt = qcos_calloc(solver->work->Wnnz, sizeof(QCOSInt));
  solver->work->kkt->ntdiag2kkt = qcos_calloc(m, sizeof(QCOSInt));
  solver->work->kkt->PregtoKKT =
      qcos_calloc(solver->work->data->P->nnz, sizeof(QCOSInt));
  solver->work->kkt->AtoKKT =
      qcos_calloc(solver->work->data->A->nnz, sizeof(QCOSInt));
  solver->work->kkt->GtoKKT =
      qcos_calloc(solver->work->data->G->nnz, sizeof(QCOSInt));
  solver->work->kkt->rhs = qcos_malloc((n + m + p) * sizeof(QCOSFloat));
  solver->work->kkt->kktres = qcos_malloc((n + m + p) * sizeof(QCOSFloat));
  solver->work->kkt->xyz = qcos_malloc((n + m + p) * sizeof(QCOSFloat));
  solver->work->kkt->xyzbuff = qcos_malloc((n + m + p) * sizeof(QCOSFloat));
  construct_kkt(solver);

  // Allocate primal and dual variables.
  solver->work->x = qcos_malloc(n * sizeof(QCOSFloat));
  solver->work->s = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->y = qcos_malloc(p * sizeof(QCOSFloat));
  solver->work->z = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->mu = 0.0;

  // Allocate Nesterov-Todd scalings and scaled variables.
  QCOSInt Wnnzfull = solver->work->data->l;
  for (QCOSInt i = 0; i < solver->work->data->nsoc; ++i) {
    Wnnzfull += solver->work->data->q[i] * solver->work->data->q[i];
  }

  solver->work->W = qcos_malloc(solver->work->Wnnz * sizeof(QCOSFloat));
  solver->work->Wfull = qcos_malloc(Wnnzfull * sizeof(QCOSFloat));
  solver->work->Winv = qcos_malloc(solver->work->Wnnz * sizeof(QCOSFloat));
  solver->work->Winvfull = qcos_malloc(Wnnzfull * sizeof(QCOSFloat));
  solver->work->WtW = qcos_malloc(solver->work->Wnnz * sizeof(QCOSFloat));
  solver->work->lambda = qcos_malloc(m * sizeof(QCOSFloat));
  QCOSInt qmax = 0;
  if (solver->work->data->q) {
    qmax = max_arrayi(solver->work->data->q, solver->work->data->nsoc);
  }
  solver->work->sbar = qcos_malloc(qmax * sizeof(QCOSFloat));
  solver->work->zbar = qcos_malloc(qmax * sizeof(QCOSFloat));
  solver->work->xbuff = qcos_malloc(n * sizeof(QCOSFloat));
  solver->work->ybuff = qcos_malloc(p * sizeof(QCOSFloat));
  solver->work->ubuff1 = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->ubuff2 = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->ubuff3 = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->Ds = qcos_malloc(m * sizeof(QCOSFloat));

  // Number of columns of KKT matrix.
  QCOSInt Kn = solver->work->kkt->K->n;

  // Allocate memory for QDLDL.
  solver->work->kkt->etree = qcos_malloc(sizeof(QCOSInt) * Kn);
  solver->work->kkt->Lnz = qcos_malloc(sizeof(QCOSInt) * Kn);
  solver->work->kkt->Lp = qcos_malloc(sizeof(QCOSInt) * (Kn + 1));
  solver->work->kkt->D = qcos_malloc(sizeof(QCOSFloat) * Kn);
  solver->work->kkt->Dinv = qcos_malloc(sizeof(QCOSFloat) * Kn);
  solver->work->kkt->iwork = qcos_malloc(sizeof(QCOSInt) * 3 * Kn);
  solver->work->kkt->bwork = qcos_malloc(sizeof(unsigned char) * Kn);
  solver->work->kkt->fwork = qcos_malloc(sizeof(QCOSFloat) * Kn);

  // Compute AMD ordering.
  QCOSCscMatrix* K = solver->work->kkt->K;
  solver->work->kkt->p = qcos_malloc(K->n * sizeof(QCOSInt));
  solver->work->kkt->pinv = qcos_malloc(K->n * sizeof(QCOSInt));
  QCOSInt amd_status = amd_order(K->n, K->p, K->i, solver->work->kkt->p,
                                 (double*)NULL, (double*)NULL);
  if (amd_status < 0) {
    return qcos_error(QCOS_AMD_ERROR);
  }
  invert_permutation(solver->work->kkt->p, solver->work->kkt->pinv, K->n);

  // Permute KKT matrix.
  QCOSInt* KtoPKPt = qcos_malloc(K->nnz * sizeof(QCOSInt));
  QCOSCscMatrix* PKPt = csc_symperm(K, solver->work->kkt->pinv, KtoPKPt);

  // Update mappings from NT matrix to permuted matrix.
  for (QCOSInt i = 0; i < solver->work->Wnnz; ++i) {
    solver->work->kkt->nt2kkt[i] = KtoPKPt[solver->work->kkt->nt2kkt[i]];
  }
  for (QCOSInt i = 0; i < m; ++i) {
    solver->work->kkt->ntdiag2kkt[i] =
        KtoPKPt[solver->work->kkt->ntdiag2kkt[i]];
  }

  for (QCOSInt i = 0; i < solver->work->data->P->nnz; ++i) {
    solver->work->kkt->PregtoKKT[i] = KtoPKPt[solver->work->kkt->PregtoKKT[i]];
  }

  for (QCOSInt i = 0; i < solver->work->data->A->nnz; ++i) {
    solver->work->kkt->AtoKKT[i] = KtoPKPt[solver->work->kkt->AtoKKT[i]];
  }

  for (QCOSInt i = 0; i < solver->work->data->G->nnz; ++i) {
    solver->work->kkt->GtoKKT[i] = KtoPKPt[solver->work->kkt->GtoKKT[i]];
  }

  free_qcos_csc_matrix(solver->work->kkt->K);
  qcos_free(KtoPKPt);

  solver->work->kkt->K = PKPt;

  // Compute elimination tree.
  QCOSInt sumLnz =
      QDLDL_etree(Kn, solver->work->kkt->K->p, solver->work->kkt->K->i,
                  solver->work->kkt->iwork, solver->work->kkt->Lnz,
                  solver->work->kkt->etree);
  if (sumLnz < 0) {
    return QCOS_SETUP_ERROR;
  }
  solver->work->kkt->Li = qcos_malloc(sizeof(QCOSInt) * sumLnz);
  solver->work->kkt->Lx = qcos_malloc(sizeof(QCOSFloat) * sumLnz);

  // Allocate solution struct.
  solver->sol = qcos_malloc(sizeof(QCOSSolution));
  solver->sol->x = qcos_malloc(n * sizeof(QCOSFloat));
  solver->sol->s = qcos_malloc(m * sizeof(QCOSFloat));
  solver->sol->y = qcos_malloc(p * sizeof(QCOSFloat));
  solver->sol->z = qcos_malloc(m * sizeof(QCOSFloat));
  solver->sol->iters = 0;
  solver->sol->status = QCOS_UNSOLVED;

  stop_timer(&setup_timer);
  solver->sol->setup_time_sec = get_elapsed_time_sec(&setup_timer);

  return QCOS_NO_ERROR;
}

void qcos_set_csc(QCOSCscMatrix* A, QCOSInt m, QCOSInt n, QCOSInt Annz,
                  QCOSFloat* Ax, QCOSInt* Ap, QCOSInt* Ai)
{
  A->m = m;
  A->n = n;
  A->nnz = Annz;
  A->x = Ax;
  A->p = Ap;
  A->i = Ai;
}

void set_default_settings(QCOSSettings* settings)
{
  settings->max_iters = 50;
  settings->bisection_iters = 5;
  settings->ruiz_iters = 5;
  settings->iterative_refinement_iterations = 3;
  settings->verbose = 0;
  settings->abstol = 1e-7;
  settings->reltol = 1e-7;
  settings->abstol_inaccurate = 1e-5;
  settings->reltol_inaccurate = 1e-5;
  settings->reg = 1e-7;
}

QCOSInt qcos_update_settings(QCOSSolver* solver,
                             const QCOSSettings* new_settings)
{
  if (qcos_validate_settings(new_settings)) {
    return qcos_error(QCOS_SETTINGS_VALIDATION_ERROR);
  }

  solver->settings->abstol = new_settings->abstol;
  solver->settings->abstol_inaccurate = new_settings->abstol_inaccurate;
  solver->settings->bisection_iters = new_settings->bisection_iters;
  solver->settings->iterative_refinement_iterations =
      new_settings->iterative_refinement_iterations;
  solver->settings->max_iters = new_settings->max_iters;
  solver->settings->reg = new_settings->reg;
  solver->settings->reltol = new_settings->reltol;
  solver->settings->reltol_inaccurate = new_settings->reltol_inaccurate;
  solver->settings->ruiz_iters = new_settings->ruiz_iters;
  solver->settings->verbose = new_settings->verbose;

  return 0;
}

void update_vector_data(QCOSSolver* solver, QCOSFloat* cnew, QCOSFloat* bnew,
                        QCOSFloat* hnew)
{
  solver->sol->status = QCOS_UNSOLVED;
  QCOSProblemData* data = solver->work->data;

  // Update cost vector.
  if (cnew) {
    for (QCOSInt i = 0; i < data->n; ++i) {
      data->c[i] = solver->work->kkt->k * solver->work->kkt->Druiz[i] * cnew[i];
    }
  }

  // Update equality constraint vector.
  if (bnew) {
    for (QCOSInt i = 0; i < data->p; ++i) {
      data->b[i] = solver->work->kkt->Eruiz[i] * bnew[i];
    }
  }

  // Update conic constraint vector.
  if (hnew) {
    for (QCOSInt i = 0; i < data->m; ++i) {
      data->h[i] = solver->work->kkt->Fruiz[i] * hnew[i];
    }
  }
}

void update_matrix_data(QCOSSolver* solver, QCOSFloat* Pxnew, QCOSFloat* Axnew,
                        QCOSFloat* Gxnew)
{
  solver->sol->status = QCOS_UNSOLVED;
  QCOSProblemData* data = solver->work->data;
  QCOSKKT* kkt = solver->work->kkt;

  // Undo regularization.
  unregularize(data->P, solver->settings->reg);

  // Unequilibrate P.
  scale_arrayf(data->P->x, data->P->x, kkt->kinv, data->P->nnz);
  row_scale(data->P, kkt->Dinvruiz);
  col_scale(data->P, kkt->Dinvruiz);

  // Unequilibrate c.
  scale_arrayf(data->c, data->c, kkt->kinv, data->n);
  ew_product(data->c, kkt->Dinvruiz, data->c, data->n);

  // Unequilibrate A.
  row_scale(data->A, kkt->Einvruiz);
  col_scale(data->A, kkt->Dinvruiz);

  // Unequilibrate G.
  row_scale(data->G, kkt->Finvruiz);
  col_scale(data->G, kkt->Dinvruiz);

  // ScaUnequilibratele b.
  ew_product(data->b, kkt->Einvruiz, data->b, data->p);

  // Unequilibrate h.
  ew_product(data->h, kkt->Finvruiz, data->h, data->m);

  // Update P and avoid nonzeros that were added for regularization.
  if (Pxnew) {
    QCOSInt avoid =
        kkt->Pnum_nzadded > 0 ? kkt->Pnzadded_idx[0] : data->P->nnz + 1;
    QCOSInt offset = 0;
    for (QCOSInt i = 0; i < data->P->nnz - kkt->Pnum_nzadded; ++i) {
      if (i == avoid) {
        offset++;
        avoid = offset > kkt->Pnum_nzadded ? kkt->Pnzadded_idx[offset]
                                           : data->P->nnz + 1;
      }
      data->P->x[i + offset] = Pxnew[i];
    }
  }

  // Update A.
  if (Axnew) {
    for (QCOSInt i = 0; i < data->A->nnz; ++i) {
      data->A->x[i] = Axnew[i];
    }
  }

  // Update G.
  if (Gxnew) {
    for (QCOSInt i = 0; i < data->G->nnz; ++i) {
      data->G->x[i] = Gxnew[i];
    }
  }

  // Equilibrate new matrix data.
  ruiz_equilibration(solver);

  // Regularize P.
  unregularize(data->P, -solver->settings->reg);

  // Update P in KKT matrix.
  for (QCOSInt i = 0; i < data->P->nnz; ++i) {
    solver->work->kkt->K->x[solver->work->kkt->PregtoKKT[i]] = data->P->x[i];
  }

  // Update A in KKT matrix.
  for (QCOSInt i = 0; i < data->A->nnz; ++i) {
    solver->work->kkt->K->x[solver->work->kkt->AtoKKT[i]] = data->A->x[i];
  }

  // Update G in KKT matrix.
  for (QCOSInt i = 0; i < data->G->nnz; ++i) {
    solver->work->kkt->K->x[solver->work->kkt->GtoKKT[i]] = data->G->x[i];
  }
}

QCOSInt qcos_solve(QCOSSolver* solver)
{
  start_timer(&(solver->work->solve_timer));

  // Validate settings.
  if (qcos_validate_settings(solver->settings)) {
    return qcos_error(QCOS_SETTINGS_VALIDATION_ERROR);
  }

  if (solver->settings->verbose) {
    print_header(solver);
  }

  // Get initializations for primal and dual variables.
  initialize_ipm(solver);

  for (QCOSInt i = 1; i <= solver->settings->max_iters; ++i) {

    // Compute kkt residual.
    compute_kkt_residual(solver);

    // Compute mu.
    compute_mu(solver->work);

    // Check stopping criteria.
    if (check_stopping(solver)) {
      stop_timer(&(solver->work->solve_timer));
      unscale_variables(solver->work);
      copy_solution(solver);
      if (solver->settings->verbose) {
        print_footer(solver->sol, solver->sol->status);
      }
      return solver->sol->status;
    }

    // Compute Nesterov-Todd scalings.
    compute_nt_scaling(solver->work);

    // Update Nestrov-Todd block of KKT matrix.
    update_nt_block(solver);

    // Perform predictor-corrector.
    predictor_corrector(solver);

    // Update iteration count.
    solver->sol->iters = i;

    // Log solver progress to console if we are solving in verbose mode.
    if (solver->settings->verbose) {
      log_iter(solver);
    }
  }

  stop_timer(&(solver->work->solve_timer));
  unscale_variables(solver->work);
  copy_solution(solver);
  solver->sol->status = QCOS_MAX_ITER;
  return QCOS_MAX_ITER;
}

QCOSInt qcos_cleanup(QCOSSolver* solver)
{

  // Free problem data.
  free_qcos_csc_matrix(solver->work->data->P);
  free_qcos_csc_matrix(solver->work->data->A);
  free_qcos_csc_matrix(solver->work->data->G);
  qcos_free(solver->work->data->b);
  qcos_free(solver->work->data->c);
  qcos_free(solver->work->data->h);
  qcos_free(solver->work->data);

  // Free primal and dual variables.
  qcos_free(solver->work->kkt->rhs);
  qcos_free(solver->work->kkt->kktres);
  qcos_free(solver->work->kkt->xyz);
  qcos_free(solver->work->kkt->xyzbuff);
  qcos_free(solver->work->x);
  qcos_free(solver->work->s);
  qcos_free(solver->work->y);
  qcos_free(solver->work->z);

  // Free Nesterov-Todd scalings and scaled variables.
  qcos_free(solver->work->W);
  qcos_free(solver->work->Wfull);
  qcos_free(solver->work->Winv);
  qcos_free(solver->work->Winvfull);
  qcos_free(solver->work->WtW);
  qcos_free(solver->work->lambda);
  qcos_free(solver->work->sbar);
  qcos_free(solver->work->zbar);
  qcos_free(solver->work->xbuff);
  qcos_free(solver->work->ybuff);
  qcos_free(solver->work->ubuff1);
  qcos_free(solver->work->ubuff2);
  qcos_free(solver->work->ubuff3);
  qcos_free(solver->work->Ds);

  // Free KKT struct.
  free_qcos_csc_matrix(solver->work->kkt->K);
  qcos_free(solver->work->kkt->p);
  qcos_free(solver->work->kkt->pinv);
  qcos_free(solver->work->kkt->delta);
  qcos_free(solver->work->kkt->Druiz);
  qcos_free(solver->work->kkt->Eruiz);
  qcos_free(solver->work->kkt->Fruiz);
  qcos_free(solver->work->kkt->Dinvruiz);
  qcos_free(solver->work->kkt->Einvruiz);
  qcos_free(solver->work->kkt->Finvruiz);
  qcos_free(solver->work->kkt->nt2kkt);
  qcos_free(solver->work->kkt->ntdiag2kkt);
  qcos_free(solver->work->kkt->Pnzadded_idx);
  qcos_free(solver->work->kkt->PregtoKKT);
  qcos_free(solver->work->kkt->AtoKKT);
  qcos_free(solver->work->kkt->GtoKKT);
  qcos_free(solver->work->kkt->etree);
  qcos_free(solver->work->kkt->Lnz);
  qcos_free(solver->work->kkt->Lp);
  qcos_free(solver->work->kkt->D);
  qcos_free(solver->work->kkt->Dinv);
  qcos_free(solver->work->kkt->iwork);
  qcos_free(solver->work->kkt->bwork);
  qcos_free(solver->work->kkt->fwork);
  qcos_free(solver->work->kkt->Li);
  qcos_free(solver->work->kkt->Lx);
  qcos_free(solver->work->kkt);

  // Free solution struct.
  qcos_free(solver->sol->x);
  qcos_free(solver->sol->s);
  qcos_free(solver->sol->y);
  qcos_free(solver->sol->z);
  qcos_free(solver->sol);

  qcos_free(solver->work);
  qcos_free(solver->settings);
  qcos_free(solver);

  return 1;
}