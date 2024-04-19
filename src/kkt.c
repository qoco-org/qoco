#include "kkt.h"

QCOSCscMatrix* allocate_kkt(QCOSProblemData* data)
{
  QCOSCscMatrix* K = qcos_malloc(sizeof(QCOSCscMatrix));

  // Number of nonzeros in KKT matrix is sum of nnz for P, A, G, and m (for the
  // NT scaling matrix block).

  K->m = data->n + data->m + data->p;
  K->n = data->n + data->m + data->p;
  K->nnz = data->P->nnz + data->A->nnz + data->G->nnz + data->m;
  K->x = qcos_calloc(K->nnz, sizeof(QCOSFloat));
  K->i = qcos_calloc(K->nnz, sizeof(QCOSInt));
  K->p = qcos_calloc((K->n + 1), sizeof(QCOSInt));

  return K;
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
  for (QCOSInt i = 0; i < work->data->A->m; ++i) {
    // Loop over columns of A
    // Counter for number of nonzeros from A added to this column of KKT matrix
    QCOSInt nzadded = 0;
    for (QCOSInt j = 0; j < work->data->A->n; ++j) {
      // Loop over all nonzeros in column j
      for (QCOSInt k = work->data->A->p[j]; k < work->data->A->p[j + 1]; ++k) {
        // If the nonzero is in row i of A then add
        if (work->data->A->i[k] == i) {
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

  // Add G^T block
  for (QCOSInt i = 0; i < work->data->G->m; ++i) {
    // Loop over columns of A
    // Counter for number of nonzeros from A added to this column of KKT matrix
    QCOSInt nzadded = 0;
    for (QCOSInt j = 0; j < work->data->G->n; ++j) {
      // Loop over all nonzeros in column j
      for (QCOSInt k = work->data->G->p[j]; k < work->data->G->p[j + 1]; ++k) {
        // If the nonzero is in row i of A then add
        if (work->data->G->i[k] == i) {
          work->kkt->K->x[nz] = work->data->G->x[k];
          work->kkt->K->i[nz] = j;
          nz += 1;
          nzadded += 1;
        }
      }
    }

    // Add -Id to NT block.
    work->kkt->K->x[nz] = -1.0;
    work->kkt->K->i[nz] = work->data->n + work->data->p + i;
    work->kkt->K->p[col] = work->kkt->K->p[col - 1] + nzadded + 1;

    // Mapping from NT matrix entries to KKT matrix entries.
    work->kkt->nt2kkt[i] = nz;

    nz += 1;
    col += 1;
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