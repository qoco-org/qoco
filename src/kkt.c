/**
 * @file kkt.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "kkt.h"
#include "qoco_utils.h"

QOCOCscMatrix* construct_kkt(QOCOCscMatrix* P, QOCOCscMatrix* A,
                             QOCOCscMatrix* G, QOCOCscMatrix* At,
                             QOCOCscMatrix* Gt, QOCOFloat static_reg, QOCOInt n,
                             QOCOInt m, QOCOInt p, QOCOInt l, QOCOInt nsoc,
                             QOCOInt* q, QOCOInt* PregtoKKT, QOCOInt* AttoKKT,
                             QOCOInt* GttoKKT, QOCOInt* nt2kkt,
                             QOCOInt* ntdiag2kkt, QOCOInt Wnnz)
{
  QOCOCscMatrix* KKT = qoco_malloc(sizeof(QOCOCscMatrix));

  KKT->m = n + m + p;
  KKT->n = n + m + p;
  KKT->nnz = P->nnz + A->nnz + G->nnz + Wnnz + p;

  KKT->x = qoco_calloc(KKT->nnz, sizeof(QOCOFloat));
  KKT->i = qoco_calloc(KKT->nnz, sizeof(QOCOInt));
  KKT->p = qoco_calloc((KKT->n + 1), sizeof(QOCOInt));

  QOCOInt nz = 0;
  QOCOInt col = 0;
  // Add P block
  for (QOCOInt k = 0; k < P->nnz; ++k) {
    PregtoKKT[k] = nz;
    KKT->x[nz] = P->x[k];
    KKT->i[nz] = P->i[k];
    nz += 1;
  }
  for (QOCOInt k = 0; k < P->n + 1; ++k) {
    KKT->p[col] = P->p[k];
    col += 1;
  }

  // Add A^T block
  for (QOCOInt Atcol = 0; Atcol < At->n; ++Atcol) {
    QOCOInt nzadded = 0;
    for (QOCOInt k = At->p[Atcol]; k < At->p[Atcol + 1]; ++k) {
      // If the nonzero is in row i of A then add
      if (AttoKKT) {
        AttoKKT[k] = nz;
      }
      KKT->x[nz] = At->x[k];
      KKT->i[nz] = At->i[k];
      nz += 1;
      nzadded += 1;
    }

    // Add -e * Id regularization.
    KKT->x[nz] = -static_reg;
    KKT->i[nz] = n + Atcol;
    nz += 1;
    nzadded += 1;
    KKT->p[col] = KKT->p[col - 1] + nzadded;
    col += 1;
  }

  // Add non-negative orthant part of G^T.
  QOCOInt nz_nt = 0;
  QOCOInt diag = 0;
  for (QOCOInt Gtcol = 0; Gtcol < l; ++Gtcol) {

    // Counter for number of nonzeros from G added to this column of KKT matrix
    QOCOInt nzadded = 0;
    for (QOCOInt k = Gt->p[Gtcol]; k < Gt->p[Gtcol + 1]; ++k) {
      if (GttoKKT) {
        GttoKKT[k] = nz;
      }
      KKT->x[nz] = Gt->x[k];
      KKT->i[nz] = Gt->i[k];
      nz += 1;
      nzadded += 1;
    }

    // Add -Id to NT block.
    KKT->x[nz] = -1.0;
    KKT->i[nz] = n + p + Gtcol;
    KKT->p[col] = KKT->p[col - 1] + nzadded + 1;

    // Mapping from NT matrix entries to KKT matrix entries.
    if (nt2kkt) {
      nt2kkt[nz_nt] = nz;
    }
    if (ntdiag2kkt) {
      ntdiag2kkt[diag] = nz;
    }
    diag++;
    nz_nt += 1;

    nz += 1;
    col += 1;
  }

  // Add second-order cone parts of G^T.
  QOCOInt idx = l;
  for (QOCOInt c = 0; c < nsoc; ++c) {
    for (QOCOInt Gtcol = idx; Gtcol < idx + q[c]; ++Gtcol) {
      // Loop over columns of G

      // Counter for number of nonzeros from G added to this column of KKT
      // matrix
      QOCOInt nzadded = 0;
      for (QOCOInt k = Gt->p[Gtcol]; k < Gt->p[Gtcol + 1]; ++k) {
        if (GttoKKT) {
          GttoKKT[k] = nz;
        }
        KKT->x[nz] = Gt->x[k];
        KKT->i[nz] = Gt->i[k];
        nz += 1;
        nzadded += 1;
      }

      // Add NT block.
      for (QOCOInt i = idx; i < idx + q[c]; i++) {
        // Only add upper triangular part.
        if (i + n + p <= col - 1) {
          // Add -1 if element is on main diagonal and 0 otherwise.
          if (i + n + p == col - 1) {
            KKT->x[nz] = -1.0;
            if (ntdiag2kkt) {
              ntdiag2kkt[diag] = nz;
            }
            diag++;
          }
          else {
            KKT->x[nz] = 0.0;
          }
          KKT->i[nz] = n + p + i;
          if (nt2kkt) {
            nt2kkt[nz_nt] = nz;
          }
          nz_nt += 1;
          nz += 1;
          nzadded += 1;
        }
      }
      KKT->p[col] = KKT->p[col - 1] + nzadded;
      // Mapping from NT matrix entries to KKT matrix entries.
      col += 1;
    }
    idx += q[c];
  }
  return KKT;
}

void initialize_ipm(QOCOSolver* solver)
{

  // TODO: Should we modify the KKT matrix in the kkt struct and in the
  // linsys_data struct?.

  // Set Nesterov-Todd block in KKT matrix to -I.
  for (QOCOInt i = 0; i < solver->work->data->m; ++i) {
    solver->work->kkt->K->x[solver->work->kkt->ntdiag2kkt[i]] = -1.0;
  }

  // Set Nesterov-Todd block in Wfull to -I.
  for (QOCOInt i = 0; i < solver->work->data->l; ++i) {
    solver->work->Wfull[i] = 1.0;
  }
  QOCOInt idx = solver->work->data->l;
  for (QOCOInt i = 0; i < solver->work->data->nsoc; ++i) {
    for (QOCOInt k = 0; k < solver->work->data->q[i]; ++k) {
      for (QOCOInt l = 0; l < solver->work->data->q[i]; ++l) {
        solver->work->Wfull[idx + k * solver->work->data->q[i] + k] = 1.0;
      }
    }
    idx += solver->work->data->q[i] * solver->work->data->q[i];
  }

  solver->linsys->linsys_initialize_nt(solver->linsys_data,
                                       solver->work->data->m);

  // Need to be set to 1.0 not 0.0 due to low tolerance stopping criteria checks
  // which only occur when a = 0.0. If a is set to 0.0 then the low tolerance
  // stopping criteria check would be triggered.
  solver->work->a = 1.0;

  // Construct rhs of KKT system.
  idx = 0;
  for (idx = 0; idx < solver->work->data->n; ++idx) {
    solver->work->kkt->rhs[idx] = -solver->work->data->c[idx];
  }
  for (QOCOInt i = 0; i < solver->work->data->p; ++i) {
    solver->work->kkt->rhs[idx] = solver->work->data->b[i];
    idx += 1;
  }
  for (QOCOInt i = 0; i < solver->work->data->m; ++i) {
    solver->work->kkt->rhs[idx] = solver->work->data->h[i];
    idx += 1;
  }

  // Factor KKT matrix.
  solver->linsys->linsys_factor(solver->linsys_data, solver->work->data->n,
                                solver->settings->kkt_dynamic_reg);

  // Solve KKT system.
  solver->linsys->linsys_solve(solver->linsys_data, solver->work,
                               solver->work->kkt->rhs, solver->work->kkt->xyz,
                               solver->settings->iter_ref_iters);

  // Copy x part of solution to x.
  copy_arrayf(solver->work->kkt->xyz, solver->work->x, solver->work->data->n);

  // Copy y part of solution to y.
  copy_arrayf(&solver->work->kkt->xyz[solver->work->data->n], solver->work->y,
              solver->work->data->p);

  // Copy z part of solution to z.
  copy_arrayf(
      &solver->work->kkt->xyz[solver->work->data->n + solver->work->data->p],
      solver->work->z, solver->work->data->m);

  // Copy and negate z part of solution to s.
  copy_and_negate_arrayf(
      &solver->work->kkt->xyz[solver->work->data->n + solver->work->data->p],
      solver->work->s, solver->work->data->m);

  // Bring s and z to cone C.
  bring2cone(solver->work->s, solver->work->data);
  bring2cone(solver->work->z, solver->work->data);
}

void set_nt_block_zeros(QOCOWorkspace* work)
{
  for (QOCOInt i = 0; i < work->Wnnz; ++i) {
    work->kkt->K->x[work->kkt->nt2kkt[i]] = 0.0;
  }
}

void update_nt_block(QOCOSolver* solver)
{
  for (QOCOInt i = 0; i < solver->work->Wnnz; ++i) {
    solver->work->kkt->K->x[solver->work->kkt->nt2kkt[i]] =
        -solver->work->WtW[i];
  }

  // Regularize Nesterov-Todd block of KKT matrix.
  for (QOCOInt i = 0; i < solver->work->data->m; ++i) {
    solver->work->kkt->K->x[solver->work->kkt->ntdiag2kkt[i]] -=
        solver->settings->kkt_static_reg;
  }
}

void compute_kkt_residual(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;

  // Zero out the NT scaling block.
  set_nt_block_zeros(work);
  for (QOCOInt i = 0; i < work->Wnnzfull; ++i) {
    work->Wfull[i] = 0.0;
  }

  // Set xyzbuff to [x;y;z]
  copy_arrayf(work->x, work->kkt->xyzbuff1, work->data->n);
  copy_arrayf(work->y, &work->kkt->xyzbuff1[work->data->n], work->data->p);
  copy_arrayf(work->z, &work->kkt->xyzbuff1[work->data->n + work->data->p],
              work->data->m);

  kkt_multiply(solver->work, work->kkt->xyzbuff1, work->kkt->kktres);

  // rhs += [c;-b;-h+s]
  QOCOInt idx;

  // Add c and account for regularization of P.
  for (idx = 0; idx < work->data->n; ++idx) {
    work->kkt->kktres[idx] =
        work->kkt->kktres[idx] +
        (work->data->c[idx] - solver->settings->kkt_static_reg * work->x[idx]);
  }

  // Add -b.
  for (QOCOInt i = 0; i < work->data->p; ++i) {
    work->kkt->kktres[idx] = work->kkt->kktres[idx] - work->data->b[i];
    idx += 1;
  }

  // Add -h + s.
  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->kkt->kktres[idx] += -work->data->h[i] + work->s[i];
    idx += 1;
  }

  // Compute objective.
  QOCOFloat obj = qoco_dot(work->x, work->data->c, work->data->n);
  USpMv(work->data->P, work->x, work->xbuff);

  // Correct for regularization in P.
  QOCOFloat regularization_correction = 0.0;
  for (QOCOInt i = 0; i < work->data->n; ++i) {
    regularization_correction +=
        solver->settings->kkt_static_reg * work->x[i] * work->x[i];
  }
  obj += 0.5 * (qoco_dot(work->xbuff, work->x, work->data->n) -
                regularization_correction);
  obj = safe_div(obj, work->kkt->k);
  solver->sol->obj = obj;
}

void construct_kkt_aff_rhs(QOCOWorkspace* work)
{
  // Negate the kkt residual and store in rhs.
  copy_and_negate_arrayf(work->kkt->kktres, work->kkt->rhs,
                         work->data->n + work->data->p + work->data->m);

  // Compute W*lambda
  nt_multiply(work->Wfull, work->lambda, work->ubuff1, work->data->l,
              work->data->m, work->data->nsoc, work->data->q);

  // Add W*lambda to z portion of rhs.
  qoco_axpy(work->ubuff1, &work->kkt->rhs[work->data->n + work->data->p],
            &work->kkt->rhs[work->data->n + work->data->p], 1.0, work->data->m);
}

void construct_kkt_comb_rhs(QOCOWorkspace* work)
{

  // Negate the kkt residual and store in rhs.
  copy_and_negate_arrayf(work->kkt->kktres, work->kkt->rhs,
                         work->data->n + work->data->p + work->data->m);

  /// ds = -cone_product(lambda, lambda) - settings.mehrotra *
  /// cone_product((W' \ Dsaff), (W * Dzaff), pdata) + sigma * mu * e.

  // ubuff1 = Winv * Dsaff.
  nt_multiply(work->Winvfull, work->Ds, work->ubuff1, work->data->l,
              work->data->m, work->data->nsoc, work->data->q);

  // ubuff2 = W * Dzaff.
  QOCOFloat* Dzaff = &work->kkt->xyz[work->data->n + work->data->p];
  nt_multiply(work->Wfull, Dzaff, work->ubuff2, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);

  // ubuff3 = cone_product((W' \ Dsaff), (W * Dzaff), pdata).
  cone_product(work->ubuff1, work->ubuff2, work->ubuff3, work->data->l,
               work->data->nsoc, work->data->q);

  // ubuff3 = cone_product((W' \ Dsaff), (W * Dzaff), pdata) - sigma * mu * e.
  QOCOFloat sm = work->sigma * work->mu;
  QOCOInt idx = 0;
  for (idx = 0; idx < work->data->l; ++idx) {
    work->ubuff3[idx] -= sm;
  }
  for (QOCOInt i = 0; i < work->data->nsoc; ++i) {
    work->ubuff3[idx] -= sm;
    idx += work->data->q[i];
  }
  // ubuff1 = lambda * lambda.
  cone_product(work->lambda, work->lambda, work->ubuff1, work->data->l,
               work->data->nsoc, work->data->q);

  // Ds = -cone_product(lambda, lambda) - settings.mehrotra *
  // cone_product((W' \ Dsaff), (W * Dzaff), pdata) + sigma * mu * e.

  copy_and_negate_arrayf(work->ubuff1, work->Ds, work->data->m);
  qoco_axpy(work->ubuff3, work->Ds, work->Ds, -1.0, work->data->m);

  // ubuff2 = cone_division(lambda, ds).
  cone_division(work->lambda, work->Ds, work->ubuff2, work->data->l,
                work->data->nsoc, work->data->q);

  // ubuff1 = W * cone_division(lambda, ds).
  nt_multiply(work->Wfull, work->ubuff2, work->ubuff1, work->data->l,
              work->data->m, work->data->nsoc, work->data->q);

  // rhs = [dx;dy;dz-W'*cone_division(lambda, ds, pdata)];
  qoco_axpy(work->ubuff1, &work->kkt->rhs[work->data->n + work->data->p],
            &work->kkt->rhs[work->data->n + work->data->p], -1.0,
            work->data->m);
}

void predictor_corrector(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;

  // Factor KKT matrix.
  solver->linsys->linsys_factor(solver->linsys_data, solver->work->data->n,
                                solver->settings->kkt_dynamic_reg);

  // Construct rhs for affine scaling direction.
  construct_kkt_aff_rhs(work);

  // Solve to get affine scaling direction.
  solver->linsys->linsys_solve(solver->linsys_data, solver->work,
                               solver->work->kkt->rhs, solver->work->kkt->xyz,
                               solver->settings->iter_ref_iters);

  // Compute Dsaff. Dsaff = W' * (-lambda - W * Dzaff).
  QOCOFloat* Dzaff = &work->kkt->xyz[work->data->n + work->data->p];
  nt_multiply(work->Wfull, Dzaff, work->ubuff1, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);
  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->ubuff1[i] = -work->lambda[i] - work->ubuff1[i];
  }
  nt_multiply(work->Wfull, work->ubuff1, work->Ds, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);

  // Compute centering parameter.
  compute_centering(solver);

  // Construct rhs for affine scaling direction.
  construct_kkt_comb_rhs(work);

  // Solve to get combined direction.
  solver->linsys->linsys_solve(solver->linsys_data, solver->work,
                               solver->work->kkt->rhs, solver->work->kkt->xyz,
                               solver->settings->iter_ref_iters);
  // Check if solution has NaNs. If NaNs are present, early exit and set a to
  // 0.0 to trigger reduced tolerance optimality checks.
  for (QOCOInt i = 0; i < work->kkt->K->n; ++i) {
    if (isnan(work->kkt->xyz[i])) {
      work->a = 0.0;
      return;
    }
  }

  // Compute Ds. Ds = W' * (cone_division(lambda, ds, pdata) - W * Dz). ds
  // computed in construct_kkt_comb_rhs() and stored in work->Ds.
  QOCOFloat* Dz = &work->kkt->xyz[work->data->n + work->data->p];
  cone_division(work->lambda, work->Ds, work->ubuff1, work->data->l,
                work->data->nsoc, work->data->q);
  nt_multiply(work->Wfull, Dz, work->ubuff2, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);

  qoco_axpy(work->ubuff2, work->ubuff1, work->ubuff3, -1.0, work->data->m);
  nt_multiply(work->Wfull, work->ubuff3, work->Ds, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);

  // Compute step-size.
  QOCOFloat a = qoco_min(linesearch(work->s, work->Ds, 0.99, solver),
                         linesearch(work->z, Dz, 0.99, solver));

  // Save step-size.
  work->a = a;

  // Update iterates.
  QOCOFloat* Dx = work->kkt->xyz;
  QOCOFloat* Dy = &work->kkt->xyz[work->data->n];

  qoco_axpy(Dx, work->x, work->x, a, work->data->n);
  qoco_axpy(work->Ds, work->s, work->s, a, work->data->m);
  qoco_axpy(Dy, work->y, work->y, a, work->data->p);
  qoco_axpy(Dz, work->z, work->z, a, work->data->m);
}

void kkt_multiply(QOCOWorkspace* work, QOCOFloat* x, QOCOFloat* y)
{
  QOCOProblemData* data = work->data;

  // Compute y[1:n] = P * x[1:n] + A^T * x[n+1:n+p] + G^T * x[n+p+1:n+p+m].
  USpMv(data->P, x, y);

  if (data->p > 0) {
    SpMtv(data->A, &x[data->n], work->xbuff);
    qoco_axpy(y, work->xbuff, y, 1.0, data->n);
    SpMv(data->A, x, &y[data->n]);
  }

  if (data->m > 0) {
    SpMtv(data->G, &x[data->n + data->p], work->xbuff);
    qoco_axpy(y, work->xbuff, y, 1.0, data->n);
    SpMv(data->G, x, &y[data->n + data->p]);
  }

  nt_multiply(work->Wfull, &x[data->n + data->p], work->ubuff1, data->l,
              data->m, data->nsoc, data->q);
  nt_multiply(work->Wfull, work->ubuff1, work->ubuff2, data->l, data->m,
              data->nsoc, data->q);
  qoco_axpy(work->ubuff2, &y[data->n + data->p], &y[data->n + data->p], -1.0,
            data->m);
}