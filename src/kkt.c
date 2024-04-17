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