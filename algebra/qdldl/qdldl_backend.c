/**
 * @file qdldl_backend.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "qdldl_backend.h"
#include <stdio.h>

// Contains data for linear system.
struct LinSysData {
  /** KKT matrix in CSC form. */
  QOCOCscMatrix* K;

  /** Diagonal of scaling matrix. */
  QOCOFloat* delta;

  /** Diagonal of scaling matrix. */
  QOCOFloat* Druiz;

  /** Diagonal of scaling matrix. */
  QOCOFloat* Eruiz;

  /** Diagonal of scaling matrix. */
  QOCOFloat* Fruiz;

  /** Inverse of Druiz. */
  QOCOFloat* Dinvruiz;

  /** Inverse of Eruiz. */
  QOCOFloat* Einvruiz;

  /** Inverse of Fruiz. */
  QOCOFloat* Finvruiz;

  /** Cost scaling factor. */
  QOCOFloat k;

  /** Inverse of cost scaling factor. */
  QOCOFloat kinv;

  /** Permutation vector. */
  QOCOInt* p;

  /** Inverse of permutation vector. */
  QOCOInt* pinv;

  /** Elimination tree for LDL factorization of K. */
  QOCOInt* etree;

  QOCOInt* Lnz;

  QOCOFloat* Lx;

  QOCOInt* Lp;

  QOCOInt* Li;

  QOCOFloat* D;

  QOCOFloat* Dinv;

  QOCOInt* iwork;

  unsigned char* bwork;

  QOCOFloat* fwork;

  /** RHS of KKT system. */
  QOCOFloat* rhs;

  /** Solution of KKT system. */
  QOCOFloat* xyz;

  /** Buffer of size n + m + p. */
  QOCOFloat* xyzbuff1;

  /** Buffer of size n + m + p. */
  QOCOFloat* xyzbuff2;

  /** Residual of KKT condition. */
  QOCOFloat* kktres;

  /** Mapping from elements in the Nesterov-Todd scaling matrix to elements in
   * the KKT matrix. */
  QOCOInt* nt2kkt;

  /** Mapping from elements on the main diagonal of the Nesterov-Todd scaling
   * matrices to elements in the KKT matrix. Used for regularization.*/
  QOCOInt* ntdiag2kkt;

  /** Mapping from elements in regularized P to elements in the KKT matrix. */
  QOCOInt* PregtoKKT;

  /** Indices of P->x that were added due to regularization. */
  QOCOInt* Pnzadded_idx;

  /** Number of elements of P->x that were added due to regularization. */
  QOCOInt Pnum_nzadded;

  /** Mapping from elements in At to elements in the KKT matrix. */
  QOCOInt* AttoKKT;

  /** Mapping from elements in Gt to elements in the KKT matrix. */
  QOCOInt* GttoKKT;

  QOCOInt* KtoPKPt;
};

static LinSysData* qdldl_setup(QOCOKKT* kkt, QOCOProblemData* data)
{
  // Number of columns of KKT matrix.
  QOCOInt Kn = kkt->K->n;

  LinSysData* linsys_data = malloc(sizeof(LinSysData));

  // Allocate memory for QDLDL.
  linsys_data->etree = qoco_malloc(sizeof(QOCOInt) * Kn);
  linsys_data->Lnz = qoco_malloc(sizeof(QOCOInt) * Kn);
  linsys_data->Lp = qoco_malloc(sizeof(QOCOInt) * (Kn + 1));
  linsys_data->D = qoco_malloc(sizeof(QOCOFloat) * Kn);
  linsys_data->Dinv = qoco_malloc(sizeof(QOCOFloat) * Kn);
  linsys_data->iwork = qoco_malloc(sizeof(QOCOInt) * 3 * Kn);
  linsys_data->bwork = qoco_malloc(sizeof(unsigned char) * Kn);
  linsys_data->fwork = qoco_malloc(sizeof(QOCOFloat) * Kn);

  // Compute AMD ordering.
  linsys_data->K = new_qoco_csc_matrix(kkt->K);
  linsys_data->p = qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
  linsys_data->pinv = qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
  QOCOInt amd_status =
      amd_order(linsys_data->K->n, linsys_data->K->p, linsys_data->K->i,
                linsys_data->p, (double*)NULL, (double*)NULL);
  if (amd_status < 0) {
    printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,,, %d", amd_status);
    return NULL;
  }
  invert_permutation(linsys_data->p, linsys_data->pinv, linsys_data->K->n);

  // Permute KKT matrix.
  linsys_data->KtoPKPt = qoco_malloc(linsys_data->K->nnz * sizeof(QOCOInt));
  QOCOCscMatrix* PKPt =
      csc_symperm(linsys_data->K, linsys_data->pinv, linsys_data->KtoPKPt);

  // Update mappings to permuted matrix.
  linsys_data->nt2kkt = qoco_calloc(kkt->Wnnz, sizeof(QOCOInt));
  linsys_data->ntdiag2kkt = qoco_calloc(data->m, sizeof(QOCOInt));
  linsys_data->PregtoKKT = qoco_calloc(data->P->nnz, sizeof(QOCOInt));
  linsys_data->AttoKKT = qoco_calloc(data->A->nnz, sizeof(QOCOInt));
  linsys_data->GttoKKT = qoco_calloc(data->G->nnz, sizeof(QOCOInt));
  for (QOCOInt i = 0; i < kkt->Wnnz; ++i) {
    linsys_data->nt2kkt[i] = linsys_data->KtoPKPt[kkt->nt2kkt[i]];
  }
  for (QOCOInt i = 0; i < data->m; ++i) {
    linsys_data->ntdiag2kkt[i] = linsys_data->KtoPKPt[kkt->ntdiag2kkt[i]];
  }

  for (QOCOInt i = 0; i < data->P->nnz; ++i) {
    linsys_data->PregtoKKT[i] = linsys_data->KtoPKPt[kkt->PregtoKKT[i]];
  }

  for (QOCOInt i = 0; i < data->A->nnz; ++i) {
    linsys_data->AttoKKT[i] = linsys_data->KtoPKPt[kkt->AttoKKT[i]];
  }

  for (QOCOInt i = 0; i < data->G->nnz; ++i) {
    linsys_data->GttoKKT[i] = linsys_data->KtoPKPt[kkt->GttoKKT[i]];
  }

  free_qoco_csc_matrix(linsys_data->K);

  linsys_data->K = PKPt;

  // Compute elimination tree.
  QOCOInt sumLnz =
      QDLDL_etree(Kn, linsys_data->p, linsys_data->K->i, linsys_data->iwork,
                  linsys_data->Lnz, linsys_data->etree);
  if (sumLnz < 0) {
    printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,,, %d", sumLnz);
    return NULL;
  }

  linsys_data->Li = qoco_malloc(sizeof(QOCOInt) * sumLnz);
  linsys_data->Lx = qoco_malloc(sizeof(QOCOFloat) * sumLnz);
  return linsys_data;
}

static void qdldl_factor() {}

static void qdldl_solve() {}

static void qdldl_cleanup() {}

// TODO: Update data

// Export the backend struct
LinSysBackend qdldl_backend = {.linsys_setup = qdldl_setup,
                               .linsys_factor = qdldl_factor,
                               .linsys_solve = qdldl_solve,
                               .linsys_cleanup = qdldl_cleanup};
