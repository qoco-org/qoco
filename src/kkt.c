#include "kkt.h"
#include "utils.h"

void allocate_kkt(QCOSWorkspace* work)
{
  work->kkt->K = qcos_malloc(sizeof(QCOSCscMatrix));

  // Number of nonzeros in KKT matrix is sum of nnz for P, A, G, and m (for the
  // NT scaling matrix block).

  // Number of nonzeros in second-order cone part of NT scaling.
  QCOSInt Wsoc_nnz = 0;
  for (QCOSInt i = 0; i < work->data->ncones; ++i) {
    Wsoc_nnz += work->data->q[i] * work->data->q[i] - work->data->q[i];
  }
  Wsoc_nnz /= 2;

  work->Wnnz = work->data->m + Wsoc_nnz;
  work->kkt->K->m = work->data->n + work->data->m + work->data->p;
  work->kkt->K->n = work->data->n + work->data->m + work->data->p;
  work->kkt->K->nnz =
      work->data->P->nnz + work->data->A->nnz + work->data->G->nnz + work->Wnnz;

  work->kkt->K->x = qcos_calloc(work->kkt->K->nnz, sizeof(QCOSFloat));
  work->kkt->K->i = qcos_calloc(work->kkt->K->nnz, sizeof(QCOSInt));
  work->kkt->K->p = qcos_calloc((work->kkt->K->n + 1), sizeof(QCOSInt));
}

void construct_kkt(QCOSWorkspace* work)
{
  QCOSInt nz = 0;
  QCOSInt col = 0;
  // Add P block
  for (QCOSInt k = 0; k < work->data->P->nnz; ++k) {
    work->kkt->K->x[nz] = work->data->P->x[k];
    work->kkt->K->i[nz] = work->data->P->i[k];
    nz += 1;
  }
  for (QCOSInt k = 0; k < work->data->P->n + 1; ++k) {
    work->kkt->K->p[col] = work->data->P->p[k];
    col += 1;
  }

  // Add A^T block
  for (QCOSInt row = 0; row < work->data->A->m; ++row) {
    // Loop over columns of A
    // Counter for number of nonzeros from A added to this column of KKT matrix
    QCOSInt nzadded = 0;
    for (QCOSInt j = 0; j < work->data->A->n; ++j) {
      // Loop over all nonzeros in column j
      for (QCOSInt k = work->data->A->p[j]; k < work->data->A->p[j + 1]; ++k) {
        // If the nonzero is in row i of A then add
        if (work->data->A->i[k] == row) {
          work->kkt->K->x[nz] = work->data->A->x[k];
          work->kkt->K->i[nz] = j;
          nz += 1;
          nzadded += 1;
        }
      }
    }
    work->kkt->K->p[col] = work->kkt->K->p[col - 1] + nzadded;
    col += 1;
  }

  // Add non-negative orthant part of G^T.
  QCOSInt nz_nt = 0;
  for (QCOSInt row = 0; row < work->data->l; ++row) {
    // Loop over columns of G
    // Counter for number of nonzeros from G added to this column of KKT matrix
    QCOSInt nzadded = 0;
    for (QCOSInt j = 0; j < work->data->G->n; ++j) {
      // Loop over all nonzeros in column j
      for (QCOSInt k = work->data->G->p[j]; k < work->data->G->p[j + 1]; ++k) {
        // If the nonzero is in row i of G then add.
        if (work->data->G->i[k] == row) {
          work->kkt->K->x[nz] = work->data->G->x[k];
          work->kkt->K->i[nz] = j;
          nz += 1;
          nzadded += 1;
        }
      }
    }

    // Add -Id to NT block.
    work->kkt->K->x[nz] = -1.0;
    work->kkt->K->i[nz] = work->data->n + work->data->p + row;
    work->kkt->K->p[col] = work->kkt->K->p[col - 1] + nzadded + 1;

    // Mapping from NT matrix entries to KKT matrix entries.
    work->kkt->nt2kkt[nz_nt] = nz;
    nz_nt += 1;

    nz += 1;
    col += 1;
  }

  // Add second-order cone parts of G^T.
  QCOSInt idx = work->data->l;
  for (QCOSInt c = 0; c < work->data->ncones; ++c) {
    for (QCOSInt row = idx; row < idx + work->data->q[c]; ++row) {
      // Loop over columns of G

      // Counter for number of nonzeros from G added to this column of KKT
      // matrix
      QCOSInt nzadded = 0;
      for (QCOSInt j = 0; j < work->data->G->n; ++j) {
        // Loop over all nonzeros in column j
        for (QCOSInt k = work->data->G->p[j]; k < work->data->G->p[j + 1];
             ++k) {
          // If the nonzero is in row i of G then add.
          if (work->data->G->i[k] == row) {
            work->kkt->K->x[nz] = work->data->G->x[k];
            work->kkt->K->i[nz] = j;
            nz += 1;
            nzadded += 1;
          }
        }
      }

      // Add NT block.
      for (QCOSInt i = idx; i < idx + work->data->q[c]; i++) {
        // Only add upper triangular part.
        if (i + work->data->n + work->data->p <= col - 1) {
          // Add -1 if element is on main diagonal and 0 otherwise.
          if (i + work->data->n + work->data->p == col - 1) {
            work->kkt->K->x[nz] = -1.0;
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

void initialize_ipm(QCOSSolver* solver)
{
  // Construct rhs of KKT system and initialize the solution variable to the
  // right hand side (needed by qdldl).
  QCOSInt idx;
  for (idx = 0; idx < solver->work->data->n; ++idx) {
    solver->work->kkt->rhs[idx] = -solver->work->data->c[idx];
    solver->work->kkt->xyz[idx] = -solver->work->data->c[idx];
  }
  for (QCOSInt i = 0; i < solver->work->data->p; ++i) {
    solver->work->kkt->rhs[idx] = solver->work->data->b[i];
    solver->work->kkt->xyz[idx] = solver->work->data->b[i];
    idx += 1;
  }
  for (QCOSInt i = 0; i < solver->work->data->m; ++i) {
    solver->work->kkt->rhs[idx] = solver->work->data->h[i];
    solver->work->kkt->xyz[idx] = solver->work->data->h[i];
    idx += 1;
  }

  // Factor KKT matrix.
  QDLDL_factor(solver->work->kkt->K->n, solver->work->kkt->K->p,
               solver->work->kkt->K->i, solver->work->kkt->K->x,
               solver->work->kkt->Lp, solver->work->kkt->Li,
               solver->work->kkt->Lx, solver->work->kkt->D,
               solver->work->kkt->Dinv, solver->work->kkt->Lnz,
               solver->work->kkt->etree, solver->work->kkt->bwork,
               solver->work->kkt->iwork, solver->work->kkt->fwork);

  // Solve KKT system.
  QDLDL_solve(solver->work->kkt->K->n, solver->work->kkt->Lp,
              solver->work->kkt->Li, solver->work->kkt->Lx,
              solver->work->kkt->Dinv, solver->work->kkt->xyz);

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

void set_nt_block_zeros(QCOSWorkspace* work)
{
  for (QCOSInt i = 0; i < work->Wnnz; ++i) {
    work->kkt->K->x[work->kkt->nt2kkt[i]] = 0.0;
  }
}

void update_nt_block(QCOSWorkspace* work)
{
  for (QCOSInt i = 0; i < work->Wnnz; ++i) {
    work->kkt->K->x[work->kkt->nt2kkt[i]] = -work->WtW[i];
  }
}
void compute_kkt_residual(QCOSWorkspace* work)
{
  // Zero out the NT scaling block.
  set_nt_block_zeros(work);

  // Set xyz to [x;y;z]
  copy_arrayf(work->x, work->kkt->xyz, work->data->n);
  copy_arrayf(work->y, &work->kkt->xyz[work->data->n], work->data->p);
  copy_arrayf(work->z, &work->kkt->xyz[work->data->n + work->data->p],
              work->data->m);

  // rhs = K * [x;y;z]
  qcos_USpMv(work->kkt->K, work->kkt->xyz, work->kkt->rhs);

  // rhs += [c;-b;-h+s]
  QCOSInt idx;

  // Add c.
  for (idx = 0; idx < work->data->n; ++idx) {
    work->kkt->rhs[idx] += work->data->c[idx];
  }

  // Add -b;
  for (QCOSInt i = 0; i < work->data->p; ++i) {
    work->kkt->rhs[idx] += -work->data->b[i];
    idx += 1;
  }

  // Add -h + s.
  for (QCOSInt i = 0; i < work->data->m; ++i) {
    work->kkt->rhs[idx] += -work->data->h[i] + work->s[i];
    idx += 1;
  }
}