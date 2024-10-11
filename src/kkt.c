/**
 * @file kkt.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 */

#include "kkt.h"
#include "utils.h"

void allocate_kkt(QOCOWorkspace* work)
{
  work->kkt->K = qoco_malloc(sizeof(QOCOCscMatrix));

  // Number of nonzeros in second-order cone part of NT scaling.
  QOCOInt Wsoc_nnz = 0;
  for (QOCOInt i = 0; i < work->data->nsoc; ++i) {
    Wsoc_nnz += work->data->q[i] * work->data->q[i] - work->data->q[i];
  }
  Wsoc_nnz /= 2;

  work->Wnnz = work->data->m + Wsoc_nnz;
  work->kkt->K->m = work->data->n + work->data->m + work->data->p;
  work->kkt->K->n = work->data->n + work->data->m + work->data->p;
  work->kkt->K->nnz = work->data->P->nnz + work->data->A->nnz +
                      work->data->G->nnz + work->Wnnz + work->data->p;

  work->kkt->K->x = qoco_calloc(work->kkt->K->nnz, sizeof(QOCOFloat));
  work->kkt->K->i = qoco_calloc(work->kkt->K->nnz, sizeof(QOCOInt));
  work->kkt->K->p = qoco_calloc((work->kkt->K->n + 1), sizeof(QOCOInt));
}

void construct_kkt(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOInt nz = 0;
  QOCOInt col = 0;
  // Add P block
  for (QOCOInt k = 0; k < work->data->P->nnz; ++k) {
    work->kkt->PregtoKKT[k] = nz;
    work->kkt->K->x[nz] = work->data->P->x[k];
    work->kkt->K->i[nz] = work->data->P->i[k];
    nz += 1;
  }
  for (QOCOInt k = 0; k < work->data->P->n + 1; ++k) {
    work->kkt->K->p[col] = work->data->P->p[k];
    col += 1;
  }

  // Add A^T block
  for (QOCOInt Atcol = 0; Atcol < work->data->At->n; ++Atcol) {
    QOCOInt nzadded = 0;
    for (QOCOInt k = work->data->At->p[Atcol]; k < work->data->At->p[Atcol + 1];
         ++k) {
      // If the nonzero is in row i of A then add
      work->kkt->AtoKKT[k] = nz;
      work->kkt->K->x[nz] = work->data->At->x[k];
      work->kkt->K->i[nz] = work->data->At->i[k];
      nz += 1;
      nzadded += 1;
    }

    // Add -e * Id regularization.
    work->kkt->K->x[nz] = -solver->settings->static_reg;
    work->kkt->K->i[nz] = work->data->n + Atcol;
    nz += 1;
    nzadded += 1;
    work->kkt->K->p[col] = work->kkt->K->p[col - 1] + nzadded;
    col += 1;
  }

  // Add non-negative orthant part of G^T.
  QOCOInt nz_nt = 0;
  QOCOInt diag = 0;
  for (QOCOInt Gtcol = 0; Gtcol < work->data->l; ++Gtcol) {

    // Counter for number of nonzeros from G added to this column of KKT matrix
    QOCOInt nzadded = 0;
    for (QOCOInt k = work->data->Gt->p[Gtcol]; k < work->data->Gt->p[Gtcol + 1];
         ++k) {
      work->kkt->GtoKKT[k] = nz;
      work->kkt->K->x[nz] = work->data->Gt->x[k];
      work->kkt->K->i[nz] = work->data->Gt->i[k];
      nz += 1;
      nzadded += 1;
    }

    // Add -Id to NT block.
    work->kkt->K->x[nz] = -1.0;
    work->kkt->K->i[nz] = work->data->n + work->data->p + Gtcol;
    work->kkt->K->p[col] = work->kkt->K->p[col - 1] + nzadded + 1;

    // Mapping from NT matrix entries to KKT matrix entries.
    work->kkt->nt2kkt[nz_nt] = nz;
    work->kkt->ntdiag2kkt[diag] = nz;
    diag++;
    nz_nt += 1;

    nz += 1;
    col += 1;
  }

  // Add second-order cone parts of G^T.
  QOCOInt idx = work->data->l;
  for (QOCOInt c = 0; c < work->data->nsoc; ++c) {
    for (QOCOInt Gtcol = idx; Gtcol < idx + work->data->q[c]; ++Gtcol) {
      // Loop over columns of G

      // Counter for number of nonzeros from G added to this column of KKT
      // matrix
      QOCOInt nzadded = 0;
      for (QOCOInt k = work->data->Gt->p[Gtcol];
           k < work->data->Gt->p[Gtcol + 1]; ++k) {
        work->kkt->GtoKKT[k] = nz;
        work->kkt->K->x[nz] = work->data->Gt->x[k];
        work->kkt->K->i[nz] = work->data->Gt->i[k];
        nz += 1;
        nzadded += 1;
      }

      // Add NT block.
      for (QOCOInt i = idx; i < idx + work->data->q[c]; i++) {
        // Only add upper triangular part.
        if (i + work->data->n + work->data->p <= col - 1) {
          // Add -1 if element is on main diagonal and 0 otherwise.
          if (i + work->data->n + work->data->p == col - 1) {
            work->kkt->K->x[nz] = -1.0;
            work->kkt->ntdiag2kkt[diag] = nz;
            diag++;
          }
          else {
            work->kkt->K->x[nz] = 0.0;
          }
          work->kkt->K->i[nz] = work->data->n + work->data->p + i;
          work->kkt->nt2kkt[nz_nt] = nz;
          nz_nt += 1;
          nz += 1;
          nzadded += 1;
        }
      }
      work->kkt->K->p[col] = work->kkt->K->p[col - 1] + nzadded;
      // Mapping from NT matrix entries to KKT matrix entries.
      col += 1;
    }
    idx += work->data->q[c];
  }
}

void initialize_ipm(QOCOSolver* solver)
{
  // Set Nesterov-Todd block to -I.
  for (QOCOInt i = 0; i < solver->work->data->m; ++i) {
    solver->work->kkt->K->x[solver->work->kkt->ntdiag2kkt[i]] = -1.0;
  }

  // Need to be set to 1.0 not 0.0 due to low tolerance stopping criteria checks
  // which only occur when a = 0.0. If a is set to 0.0 then the low tolerance
  // stopping criteria check would be triggered.
  solver->work->a = 1.0;

  // Construct rhs of KKT system..
  QOCOInt idx;
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
  QDLDL_factor(
      solver->work->kkt->K->n, solver->work->kkt->K->p, solver->work->kkt->K->i,
      solver->work->kkt->K->x, solver->work->kkt->Lp, solver->work->kkt->Li,
      solver->work->kkt->Lx, solver->work->kkt->D, solver->work->kkt->Dinv,
      solver->work->kkt->Lnz, solver->work->kkt->etree,
      solver->work->kkt->bwork, solver->work->kkt->iwork,
      solver->work->kkt->fwork, solver->work->kkt->p, solver->work->data->n,
      solver->settings->dyn_reg);

  // Solve KKT system.
  kkt_solve(solver->work->kkt, solver->work->kkt->rhs,
            solver->settings->iterative_refinement_iterations);

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
        solver->settings->static_reg;
  }
}

void compute_kkt_residual(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;

  // Zero out the NT scaling block.
  set_nt_block_zeros(work);

  // Set xyzbuff to [x;y;z]
  copy_arrayf(work->x, work->kkt->xyzbuff, work->data->n);
  copy_arrayf(work->y, &work->kkt->xyzbuff[work->data->n], work->data->p);
  copy_arrayf(work->z, &work->kkt->xyzbuff[work->data->n + work->data->p],
              work->data->m);

  // Permute xyzbuff and store into xyz.
  for (QOCOInt i = 0; i < work->kkt->K->n; ++i) {
    work->kkt->xyz[i] = work->kkt->xyzbuff[work->kkt->p[i]];
  }

  // rhs = K * [x;y;z]
  USpMv(work->kkt->K, work->kkt->xyz, work->kkt->xyzbuff);

  // Permute again to get xyz.
  for (QOCOInt i = 0; i < work->kkt->K->n; ++i) {
    work->kkt->kktres[work->kkt->p[i]] = work->kkt->xyzbuff[i];
  }

  // rhs += [c;-b;-h+s]
  QOCOInt idx;

  // Add c and account for regularization of P.
  for (idx = 0; idx < work->data->n; ++idx) {
    work->kkt->kktres[idx] =
        work->kkt->kktres[idx] +
        (work->data->c[idx] - solver->settings->static_reg * work->x[idx]);
  }

  // Add -b and account for regularization.
  for (QOCOInt i = 0; i < work->data->p; ++i) {
    work->kkt->kktres[idx] =
        work->kkt->kktres[idx] +
        (-work->data->b[i] + solver->settings->static_reg * work->y[i]);
    idx += 1;
  }

  // Add -h + s.
  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->kkt->kktres[idx] += -work->data->h[i] + work->s[i];
    idx += 1;
  }

  // Compute objective.
  QOCOFloat obj = dot(work->x, work->data->c, work->data->n);
  USpMv(work->data->P, work->x, work->xbuff);

  // Correct for regularization in P.
  QOCOFloat regularization_correction = 0.0;
  for (QOCOInt i = 0; i < work->data->n; ++i) {
    regularization_correction +=
        solver->settings->static_reg * work->x[i] * work->x[i];
  }
  obj += 0.5 *
         (dot(work->xbuff, work->x, work->data->n) - regularization_correction);
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
  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->kkt->rhs[work->data->n + work->data->p + i] += work->ubuff1[i];
  }
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

  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->Ds[i] = -work->ubuff1[i] - work->ubuff3[i];
  }

  // ubuff2 = cone_division(lambda, ds).
  cone_division(work->lambda, work->Ds, work->ubuff2, work->data->l,
                work->data->nsoc, work->data->q);

  // ubuff1 = W * cone_division(lambda, ds).
  nt_multiply(work->Wfull, work->ubuff2, work->ubuff1, work->data->l,
              work->data->m, work->data->nsoc, work->data->q);

  // rhs = [dx;dy;dz-W'*cone_division(lambda, ds, pdata)];
  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->kkt->rhs[work->data->n + work->data->p + i] -= work->ubuff1[i];
  }
}

void predictor_corrector(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;

  // Factor KKT matrix.
  QDLDL_factor(work->kkt->K->n, work->kkt->K->p, work->kkt->K->i,
               work->kkt->K->x, work->kkt->Lp, work->kkt->Li, work->kkt->Lx,
               work->kkt->D, work->kkt->Dinv, work->kkt->Lnz, work->kkt->etree,
               work->kkt->bwork, work->kkt->iwork, work->kkt->fwork,
               solver->work->kkt->p, solver->work->data->n,
               solver->settings->dyn_reg);

  // Construct rhs for affine scaling direction.
  construct_kkt_aff_rhs(work);

  // Solve to get affine scaling direction.
  kkt_solve(work->kkt, work->kkt->rhs,
            solver->settings->iterative_refinement_iterations);

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
  kkt_solve(work->kkt, work->kkt->rhs,
            solver->settings->iterative_refinement_iterations);

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
  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->ubuff3[i] = work->ubuff1[i] - work->ubuff2[i];
  }
  nt_multiply(work->Wfull, work->ubuff3, work->Ds, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);

  // Compute step-size.
  nt_multiply(work->Winvfull, work->Ds, work->ubuff3, work->data->l,
              work->data->m, work->data->nsoc, work->data->q);
  nt_multiply(work->Wfull, Dz, work->ubuff2, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);
  QOCOFloat a = qoco_min(linesearch(work->lambda, work->ubuff3, 0.99, solver),
                         linesearch(work->lambda, work->ubuff2, 0.99, solver));

  // Save step-size.
  work->a = a;

  // Update iterates.
  QOCOFloat* Dx = work->kkt->xyz;
  QOCOFloat* Dy = &work->kkt->xyz[work->data->n];

  for (QOCOInt i = 0; i < work->data->n; ++i) {
    work->x[i] = work->x[i] + a * Dx[i];
  }
  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->s[i] = work->s[i] + a * work->Ds[i];
  }
  for (QOCOInt i = 0; i < work->data->p; ++i) {
    work->y[i] = work->y[i] + a * Dy[i];
  }
  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->z[i] = work->z[i] + a * Dz[i];
  }
}

void kkt_solve(QOCOKKT* kkt, QOCOFloat* b, QOCOInt iters)
{
  // Permute b and store in xyzbuff.
  for (QOCOInt i = 0; i < kkt->K->n; ++i) {
    kkt->xyzbuff[i] = b[kkt->p[i]];
  }

  // Copy permuted b into b.
  copy_arrayf(kkt->xyzbuff, b, kkt->K->n);

  // Triangular solve.
  QDLDL_solve(kkt->K->n, kkt->Lp, kkt->Li, kkt->Lx, kkt->Dinv, kkt->xyzbuff);

  // Iterative refinement.
  for (QOCOInt i = 0; i < iters; ++i) {
    // r = b - K * x
    USpMv(kkt->K, kkt->xyzbuff, kkt->xyz);
    for (QOCOInt j = 0; j < kkt->K->n; ++j) {
      kkt->xyz[j] = b[j] - kkt->xyz[j];
    }

    // dx = K \ r
    QDLDL_solve(kkt->K->n, kkt->Lp, kkt->Li, kkt->Lx, kkt->Dinv, kkt->xyz);

    // x = x + dx.
    axpy(kkt->xyzbuff, kkt->xyz, kkt->xyzbuff, 1.0, kkt->K->n);
  }

  for (QOCOInt i = 0; i < kkt->K->n; ++i) {
    kkt->xyz[kkt->p[i]] = kkt->xyzbuff[i];
  }
}
