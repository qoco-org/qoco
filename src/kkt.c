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
  QOCOInt Pnnz = P ? P->nnz : 0;
  KKT->nnz = Pnnz + A->nnz + G->nnz + Wnnz + p;

  KKT->x = qoco_calloc(KKT->nnz, sizeof(QOCOFloat));
  KKT->i = qoco_calloc(KKT->nnz, sizeof(QOCOInt));
  KKT->p = qoco_calloc((KKT->n + 1), sizeof(QOCOInt));

  QOCOInt nz = 0;
  QOCOInt col = 0;
  // Add P block
  if (P) {
    for (QOCOInt k = 0; k < P->nnz; ++k) {
      if (PregtoKKT) {
        PregtoKKT[k] = nz;
      }
      KKT->x[nz] = P->x[k];
      KKT->i[nz] = P->i[k];
      nz += 1;
    }
    for (QOCOInt k = 0; k < P->n + 1; ++k) {
      KKT->p[col] = P->p[k];
      col += 1;
    }
  }
  else {
    // P is NULL, so just set column pointers for n columns
    for (QOCOInt k = 0; k < n + 1; ++k) {
      KKT->p[col] = 0;
      col += 1;
    }
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

  // Set Nesterov-Todd block in Wfull to -I (need for kkt_multiply in iterative
  // refinement).
  for (QOCOInt i = 0; i < solver->work->Wnnzfull; ++i) {
    solver->work->Wfull[i] = 0.0;
  }
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
  // Construct rhs of KKT system = [-c;b;h].
  QOCOProblemData* data = solver->work->data;
  QOCOFloat* cdata = get_data_vectorf(data->c);
  QOCOFloat* bdata = get_data_vectorf(data->b);
  QOCOFloat* hdata = get_data_vectorf(data->h);
  QOCOFloat* rhs = get_data_vectorf(solver->work->rhs);
  QOCOFloat* xyz = get_data_vectorf(solver->work->xyz);
  copy_and_negate_arrayf(cdata, rhs, data->n);
  copy_arrayf(bdata, &rhs[data->n], data->p);
  copy_arrayf(hdata, &rhs[data->n + data->p], data->m);

  // Factor KKT matrix.
  solver->linsys->linsys_factor(solver->linsys_data, solver->work->data->n,
                                solver->settings->kkt_dynamic_reg);

  // Solve KKT system.
  solver->linsys->linsys_solve(solver->linsys_data, solver->work, rhs, xyz,
                               solver->settings->iter_ref_iters);

  // Copy x part of solution to x (GPU-to-GPU copy).
  // During solve phase, get_data_vectorf returns device pointers, so
  // copy_arrayf will detect both xyz and work->x are on device and use a CUDA
  // kernel.
  copy_arrayf(xyz, get_data_vectorf(solver->work->x), solver->work->data->n);

  // Copy y part of solution to y (GPU-to-GPU copy).
  copy_arrayf(&xyz[solver->work->data->n], get_data_vectorf(solver->work->y),
              solver->work->data->p);

  // Copy z part of solution to z (GPU-to-GPU copy).
  copy_arrayf(&xyz[solver->work->data->n + solver->work->data->p],
              get_data_vectorf(solver->work->z), solver->work->data->m);

  // Copy and negate z part of solution to s (GPU-to-GPU copy).
  copy_and_negate_arrayf(&xyz[solver->work->data->n + solver->work->data->p],
                         get_data_vectorf(solver->work->s),
                         solver->work->data->m);

  // Bring s and z to cone C.
  bring2cone(get_data_vectorf(solver->work->s), solver->work->data);
  bring2cone(get_data_vectorf(solver->work->z), solver->work->data);
}

void compute_kkt_residual(QOCOProblemData* data, QOCOFloat* x, QOCOFloat* y,
                          QOCOFloat* s, QOCOFloat* z, QOCOFloat* kktres,
                          QOCOFloat static_reg, QOCOFloat* xyzbuff,
                          QOCOFloat* nbuff, QOCOFloat* mbuff1,
                          QOCOFloat* mbuff2)
{

  // Set xyzbuff to [x;y;z]
  copy_arrayf(x, xyzbuff, data->n);
  copy_arrayf(y, &xyzbuff[data->n], data->p);
  copy_arrayf(z, &xyzbuff[data->n + data->p], data->m);

  // Compute K*[x;y;z] with a zero'd out NT block.
  kkt_multiply(xyzbuff, kktres, data, NULL, nbuff, mbuff1, mbuff1);

  // rhs += [c;-b;-h+s]
  // Add c and account for regularization of P.
  QOCOFloat* cdata = get_data_vectorf(data->c);
  QOCOFloat* bdata = get_data_vectorf(data->b);
  QOCOFloat* hdata = get_data_vectorf(data->h);
  qoco_axpy(cdata, kktres, kktres, 1.0, data->n);
  qoco_axpy(x, kktres, kktres, -static_reg, data->n);

  // Add -b.
  qoco_axpy(bdata, &kktres[data->n], &kktres[data->n], -1.0, data->p);

  // Add -h + s.
  qoco_axpy(hdata, &kktres[data->n + data->p], &kktres[data->n + data->p], -1.0,
            data->m);
  qoco_axpy(s, &kktres[data->n + data->p], &kktres[data->n + data->p], 1.0,
            data->m);
}

QOCOFloat compute_objective(QOCOProblemData* data, QOCOFloat* x,
                            QOCOFloat* nbuff, QOCOFloat static_reg, QOCOFloat k)
{
  QOCOFloat* cdata = get_data_vectorf(data->c);
  QOCOFloat obj = qoco_dot(x, cdata, data->n);
  if (data->P) {
    USpMv_matrix(data->P, x, nbuff);
  }
  else {
    for (QOCOInt i = 0; i < data->n; ++i) {
      nbuff[i] = 0.0;
    }
  }

  // Correct for regularization in P.
  QOCOFloat regularization_correction = 0.0;
  for (QOCOInt i = 0; i < data->n; ++i) {
    regularization_correction += static_reg * x[i] * x[i];
  }
  obj += 0.5 * (qoco_dot(nbuff, x, data->n) - regularization_correction);
  obj = safe_div(obj, k);
  return obj;
}
void construct_kkt_aff_rhs(QOCOWorkspace* work)
{
  // During solve phase, get_data_vectorf returns device pointer automatically
  QOCOFloat* rhs = get_data_vectorf(work->rhs);

  // Negate the kkt residual and store in rhs.
  copy_and_negate_arrayf(work->kktres, rhs,
                         work->data->n + work->data->p + work->data->m);

  // Compute W*lambda
  nt_multiply(work->Wfull, work->lambda, work->ubuff1, work->data->l,
              work->data->m, work->data->nsoc, work->data->q);

  // Add W*lambda to z portion of rhs.
  qoco_axpy(work->ubuff1, &rhs[work->data->n + work->data->p],
            &rhs[work->data->n + work->data->p], 1.0, work->data->m);
}

void construct_kkt_comb_rhs(QOCOWorkspace* work)
{
  // During solve phase, get_data_vectorf returns device pointer automatically
  QOCOFloat* rhs = get_data_vectorf(work->rhs);
  QOCOFloat* xyz = get_data_vectorf(work->xyz);

  // Negate the kkt residual and store in rhs.
  copy_and_negate_arrayf(work->kktres, rhs,
                         work->data->n + work->data->p + work->data->m);

  /// ds = -cone_product(lambda, lambda) - settings.mehrotra *
  /// cone_product((W' \ Dsaff), (W * Dzaff), pdata) + sigma * mu * e.

  // ubuff1 = Winv * Dsaff.
  nt_multiply(work->Winvfull, work->Ds, work->ubuff1, work->data->l,
              work->data->m, work->data->nsoc, work->data->q);

  // ubuff2 = W * Dzaff.
  QOCOFloat* Dzaff = &xyz[work->data->n + work->data->p];
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
  qoco_axpy(work->ubuff1, &rhs[work->data->n + work->data->p],
            &rhs[work->data->n + work->data->p], -1.0, work->data->m);
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
  // During solve phase, get_data_vectorf returns device pointers automatically
  QOCOFloat* rhs = get_data_vectorf(work->rhs);
  QOCOFloat* xyz = get_data_vectorf(work->xyz);
  solver->linsys->linsys_solve(solver->linsys_data, solver->work, rhs, xyz,
                               solver->settings->iter_ref_iters);

  // Compute Dsaff. Dsaff = W' * (-lambda - W * Dzaff).
  QOCOFloat* Dzaff = &xyz[work->data->n + work->data->p];
  nt_multiply(work->Wfull, Dzaff, work->ubuff1, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);
  for (QOCOInt i = 0; i < work->data->m; ++i) {
    work->ubuff1[i] = -work->lambda[i] - work->ubuff1[i];
  }
  nt_multiply(work->Wfull, work->ubuff1, work->Ds, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);

  // Compute centering parameter.
  compute_centering(solver);

  // Construct rhs for combined direction.
  construct_kkt_comb_rhs(work);

  // Solve to get combined direction.
  // During solve phase, get_data_vectorf returns device pointers automatically
  rhs = get_data_vectorf(work->rhs);
  xyz = get_data_vectorf(work->xyz);
  solver->linsys->linsys_solve(solver->linsys_data, solver->work, rhs, xyz,
                               solver->settings->iter_ref_iters);

// Check if solution has NaNs. If NaNs are present, early exit and set a to
// 0.0 to trigger reduced tolerance optimality checks.
// During solve phase, xyz is on device, so we need to sync to host for NaN
// check But this is error checking, not part of normal solve flow
#ifdef QOCO_ALGEBRA_BACKEND_CUDA
  extern int get_solve_phase(void);
  if (get_solve_phase()) {
    // Temporarily sync xyz from device to host for NaN check
    // This creates a temporary copy, but it's only for error checking
    QOCOInt n = work->data->n + work->data->m + work->data->p;
    QOCOFloat* xyz_host = (QOCOFloat*)malloc(n * sizeof(QOCOFloat));
    extern void* cudaMemcpy(void* dst, const void* src, size_t count, int kind);
#define cudaMemcpyDeviceToHost 2
    cudaMemcpy(xyz_host, xyz, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost);
    for (QOCOInt i = 0; i < n; ++i) {
      if (isnan(xyz_host[i])) {
        free(xyz_host);
        work->a = 0.0;
        return;
      }
    }
    free(xyz_host);
  }
  else {
    // Not in solve phase, use host pointer directly
    QOCOFloat* xyz_host = get_data_vectorf(work->xyz);
    for (QOCOInt i = 0; i < work->data->n + work->data->p + work->data->m;
         ++i) {
      if (isnan(xyz_host[i])) {
        work->a = 0.0;
        return;
      }
    }
  }
#else
  // Builtin backend - use host pointer directly
  QOCOFloat* xyz_host = get_data_vectorf(work->xyz);
  for (QOCOInt i = 0; i < work->data->n + work->data->p + work->data->m; ++i) {
    if (isnan(xyz_host[i])) {
      work->a = 0.0;
      return;
    }
  }
#endif

  // Compute Ds. Ds = W' * (cone_division(lambda, ds, pdata) - W * Dz). ds
  // computed in construct_kkt_comb_rhs() and stored in work->Ds.
  // Use device pointer during solve phase
  QOCOFloat* Dz = &xyz[work->data->n + work->data->p];
  cone_division(work->lambda, work->Ds, work->ubuff1, work->data->l,
                work->data->nsoc, work->data->q);
  nt_multiply(work->Wfull, Dz, work->ubuff2, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);

  qoco_axpy(work->ubuff2, work->ubuff1, work->ubuff3, -1.0, work->data->m);
  nt_multiply(work->Wfull, work->ubuff3, work->Ds, work->data->l, work->data->m,
              work->data->nsoc, work->data->q);

  // Compute step-size.
  QOCOFloat a =
      qoco_min(linesearch(get_data_vectorf(work->s), work->Ds, 0.99, solver),
               linesearch(get_data_vectorf(work->z), Dz, 0.99, solver));

  // Save step-size.
  work->a = a;

  // Update iterates.
  // During solve phase, get_data_vectorf returns device pointer automatically
  QOCOFloat* Dx = xyz;
  QOCOFloat* Dy = &xyz[work->data->n];
  QOCOFloat* Ds = work->Ds;

  qoco_axpy(Dx, get_data_vectorf(work->x), get_data_vectorf(work->x), a,
            work->data->n);
  qoco_axpy(Ds, get_data_vectorf(work->s), get_data_vectorf(work->s), a,
            work->data->m);
  qoco_axpy(Dy, get_data_vectorf(work->y), get_data_vectorf(work->y), a,
            work->data->p);
  qoco_axpy(Dz, get_data_vectorf(work->z), get_data_vectorf(work->z), a,
            work->data->m);

  // Note: No sync needed here - get_data_vectorf returns device pointer during
  // solve phase, so qoco_axpy operates directly on device memory
}

void kkt_multiply(QOCOFloat* x, QOCOFloat* y, QOCOProblemData* data,
                  QOCOFloat* Wfull, QOCOFloat* nbuff, QOCOFloat* mbuff1,
                  QOCOFloat* mbuff2)
{

  // Compute y[1:n] = P * x[1:n] + A^T * x[n+1:n+p] + G^T * x[n+p+1:n+p+m].
  if (data->P) {
    USpMv_matrix(data->P, x, y);
  }
  else {
    for (QOCOInt i = 0; i < data->n; ++i) {
      y[i] = 0.0;
    }
  }

  if (data->p > 0) {
    SpMtv_matrix(data->A, &x[data->n], nbuff);
    qoco_axpy(y, nbuff, y, 1.0, data->n);
    SpMv_matrix(data->A, x, &y[data->n]);
  }

  if (data->m > 0) {
    SpMtv_matrix(data->G, &x[data->n + data->p], nbuff);
    qoco_axpy(y, nbuff, y, 1.0, data->n);
    SpMv_matrix(data->G, x, &y[data->n + data->p]);
  }

  if (Wfull) {
    nt_multiply(Wfull, &x[data->n + data->p], mbuff1, data->l, data->m,
                data->nsoc, data->q);
    nt_multiply(Wfull, mbuff1, mbuff2, data->l, data->m, data->nsoc, data->q);
    qoco_axpy(mbuff2, &y[data->n + data->p], &y[data->n + data->p], -1.0,
              data->m);
  }
}