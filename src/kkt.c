#include "kkt.h"

QCOSCscMatrix* initialize_kkt(QCOSProblemData* data)
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