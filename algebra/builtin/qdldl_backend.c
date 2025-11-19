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

// Contains data for linear system.
struct LinSysData {
  /** KKT matrix in CSC form. */
  QOCOCscMatrix* K;

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

  /** Buffer of size n + m + p. */
  QOCOFloat* xyzbuff1;

  /** Buffer of size n + m + p. */
  QOCOFloat* xyzbuff2;

  /** Mapping from elements in the Nesterov-Todd scaling matrix to elements in
   * the KKT matrix. */
  QOCOInt* nt2kkt;

  /** Mapping from elements on the main diagonal of the Nesterov-Todd scaling
   * matrices to elements in the KKT matrix. Used for regularization.*/
  QOCOInt* ntdiag2kkt;

  /** Mapping from elements in regularized P to elements in the KKT matrix. */
  QOCOInt* PregtoKKT;

  /** Mapping from elements in At to elements in the KKT matrix. */
  QOCOInt* AttoKKT;

  /** Mapping from elements in Gt to elements in the KKT matrix. */
  QOCOInt* GttoKKT;

  QOCOInt Wnnz;
};

static LinSysData* qdldl_setup(QOCOProblemData* data, QOCOSettings* settings,
                               QOCOInt Wnnz)
{
  // Number of columns of KKT matrix.
  QOCOInt Kn = data->n + data->m + data->p;

  LinSysData* linsys_data = malloc(sizeof(LinSysData));

  // Allocate vector buffers.
  linsys_data->xyzbuff1 = qoco_malloc(sizeof(QOCOFloat) * Kn);
  linsys_data->xyzbuff2 = qoco_malloc(sizeof(QOCOFloat) * Kn);
  linsys_data->Wnnz = Wnnz;

  // Allocate memory for QDLDL.
  linsys_data->etree = qoco_malloc(sizeof(QOCOInt) * Kn);
  linsys_data->Lnz = qoco_malloc(sizeof(QOCOInt) * Kn);
  linsys_data->Lp = qoco_malloc(sizeof(QOCOInt) * (Kn + 1));
  linsys_data->D = qoco_malloc(sizeof(QOCOFloat) * Kn);
  linsys_data->Dinv = qoco_malloc(sizeof(QOCOFloat) * Kn);
  linsys_data->iwork = qoco_malloc(sizeof(QOCOInt) * 3 * Kn);
  linsys_data->bwork = qoco_malloc(sizeof(unsigned char) * Kn);
  linsys_data->fwork = qoco_malloc(sizeof(QOCOFloat) * Kn);

  // Allocate memory for mappings to KKT matrix.
  linsys_data->nt2kkt = qoco_calloc(Wnnz, sizeof(QOCOInt));
  linsys_data->ntdiag2kkt = qoco_calloc(data->m, sizeof(QOCOInt));
  linsys_data->PregtoKKT = qoco_calloc(get_nnz(data->P), sizeof(QOCOInt));
  linsys_data->AttoKKT = qoco_calloc(get_nnz(data->A), sizeof(QOCOInt));
  linsys_data->GttoKKT = qoco_calloc(get_nnz(data->G), sizeof(QOCOInt));

  QOCOInt* nt2kkt_temp = qoco_calloc(Wnnz, sizeof(QOCOInt));
  QOCOInt* ntdiag2kkt_temp = qoco_calloc(data->m, sizeof(QOCOInt));
  QOCOInt* PregtoKKT_temp = data->P ? qoco_calloc(get_nnz(data->P), sizeof(QOCOInt)) : NULL;
  QOCOInt* AttoKKT_temp = qoco_calloc(get_nnz(data->A), sizeof(QOCOInt));
  QOCOInt* GttoKKT_temp = qoco_calloc(get_nnz(data->G), sizeof(QOCOInt));

  linsys_data->K = construct_kkt(
      data->P ? get_csc_matrix(data->P) : NULL, get_csc_matrix(data->A), get_csc_matrix(data->G), 
      get_csc_matrix(data->At), get_csc_matrix(data->Gt),
      settings->kkt_static_reg, data->n, data->m, data->p, data->l, data->nsoc,
      data->q, PregtoKKT_temp, AttoKKT_temp, GttoKKT_temp, nt2kkt_temp,
      ntdiag2kkt_temp, Wnnz);

  // Compute AMD ordering.
  linsys_data->p = qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
  linsys_data->pinv = qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
  QOCOInt amd_status =
      amd_order(linsys_data->K->n, linsys_data->K->p, linsys_data->K->i,
                linsys_data->p, (double*)NULL, (double*)NULL);
  if (amd_status < 0) {
    return NULL;
  }
  invert_permutation(linsys_data->p, linsys_data->pinv, linsys_data->K->n);

  // Permute KKT matrix.
  QOCOInt* KtoPKPt = qoco_malloc(linsys_data->K->nnz * sizeof(QOCOInt));
  QOCOCscMatrix* PKPt = csc_symperm(linsys_data->K, linsys_data->pinv, KtoPKPt);

  // Update mappings to permuted matrix.
  for (QOCOInt i = 0; i < Wnnz; ++i) {
    linsys_data->nt2kkt[i] = KtoPKPt[nt2kkt_temp[i]];
  }
  for (QOCOInt i = 0; i < data->m; ++i) {
    linsys_data->ntdiag2kkt[i] = KtoPKPt[ntdiag2kkt_temp[i]];
  }

  if (data->P && PregtoKKT_temp) {
    for (QOCOInt i = 0; i < get_nnz(data->P); ++i) {
      linsys_data->PregtoKKT[i] = KtoPKPt[PregtoKKT_temp[i]];
    }
  }

  for (QOCOInt i = 0; i < get_nnz(data->A); ++i) {
    linsys_data->AttoKKT[i] = KtoPKPt[AttoKKT_temp[i]];
  }

  for (QOCOInt i = 0; i < get_nnz(data->G); ++i) {
    linsys_data->GttoKKT[i] = KtoPKPt[GttoKKT_temp[i]];
  }

  free_qoco_csc_matrix(linsys_data->K);
  qoco_free(KtoPKPt);
  qoco_free(nt2kkt_temp);
  qoco_free(ntdiag2kkt_temp);
  qoco_free(PregtoKKT_temp);
  qoco_free(AttoKKT_temp);
  qoco_free(GttoKKT_temp);
  linsys_data->K = PKPt;

  // Compute elimination tree.
  QOCOInt sumLnz =
      QDLDL_etree(Kn, linsys_data->K->p, linsys_data->K->i, linsys_data->iwork,
                  linsys_data->Lnz, linsys_data->etree);
  if (sumLnz < 0) {
    return NULL;
  }

  linsys_data->Li = qoco_malloc(sizeof(QOCOInt) * sumLnz);
  linsys_data->Lx = qoco_malloc(sizeof(QOCOFloat) * sumLnz);
  return linsys_data;
}

static void qdldl_factor(LinSysData* linsys_data, QOCOInt n,
                         QOCOFloat kkt_dynamic_reg)
{
  QDLDL_factor(linsys_data->K->n, linsys_data->K->p, linsys_data->K->i,
               linsys_data->K->x, linsys_data->Lp, linsys_data->Li,
               linsys_data->Lx, linsys_data->D, linsys_data->Dinv,
               linsys_data->Lnz, linsys_data->etree, linsys_data->bwork,
               linsys_data->iwork, linsys_data->fwork, linsys_data->p, n,
               kkt_dynamic_reg);
}

static void qdldl_solve(LinSysData* linsys_data, QOCOWorkspace* work,
                        QOCOFloat* b, QOCOFloat* x, QOCOInt iter_ref_iters)
{
  // Permute b and store in xyzbuff.
  for (QOCOInt i = 0; i < linsys_data->K->n; ++i) {
    linsys_data->xyzbuff1[i] = b[linsys_data->p[i]];
  }

  // Copy permuted b into b.
  copy_arrayf(linsys_data->xyzbuff1, b, linsys_data->K->n);

  // Triangular solve.
  QDLDL_solve(linsys_data->K->n, linsys_data->Lp, linsys_data->Li,
              linsys_data->Lx, linsys_data->Dinv, linsys_data->xyzbuff1);

  // Iterative refinement.
  for (QOCOInt i = 0; i < iter_ref_iters; ++i) {
    // r = b - K * x

    for (QOCOInt k = 0; k < linsys_data->K->n; ++k) {
      x[linsys_data->p[k]] = linsys_data->xyzbuff1[k];
    }

    kkt_multiply(x, linsys_data->xyzbuff2, work->data, work->Wfull, work->xbuff,
                 work->ubuff1, work->ubuff2);
    for (QOCOInt k = 0; k < linsys_data->K->n; ++k) {
      x[k] = linsys_data->xyzbuff2[linsys_data->p[k]];
    }

    for (QOCOInt j = 0; j < linsys_data->K->n; ++j) {
      x[j] = b[j] - x[j];
    }

    // dx = K \ r
    QDLDL_solve(linsys_data->K->n, linsys_data->Lp, linsys_data->Li,
                linsys_data->Lx, linsys_data->Dinv, x);

    // x = x + dx.
    qoco_axpy(linsys_data->xyzbuff1, x, linsys_data->xyzbuff1, 1.0,
              linsys_data->K->n);
  }

  for (QOCOInt i = 0; i < linsys_data->K->n; ++i) {
    x[linsys_data->p[i]] = linsys_data->xyzbuff1[i];
  }
}

static void qdldl_initialize_nt(LinSysData* linsys_data, QOCOInt m)
{
  for (QOCOInt i = 0; i < linsys_data->Wnnz; ++i) {
    linsys_data->K->x[linsys_data->nt2kkt[i]] = 0.0;
  }

  // Set Nesterov-Todd block in KKT matrix to -I.
  for (QOCOInt i = 0; i < m; ++i) {
    linsys_data->K->x[linsys_data->ntdiag2kkt[i]] = -1.0;
  }
}

static void qdldl_update_nt(LinSysData* linsys_data, QOCOFloat* WtW,
                            QOCOFloat kkt_static_reg, QOCOInt m)
{
  for (QOCOInt i = 0; i < linsys_data->Wnnz; ++i) {
    linsys_data->K->x[linsys_data->nt2kkt[i]] = -WtW[i];
  }

  // Regularize Nesterov-Todd block of KKT matrix.
  for (QOCOInt i = 0; i < m; ++i) {
    linsys_data->K->x[linsys_data->ntdiag2kkt[i]] -= kkt_static_reg;
  }
}

static void qdldl_update_data(LinSysData* linsys_data, QOCOProblemData* data)
{
  // Update P in KKT matrix.
  if (data->P && linsys_data->PregtoKKT) {
    QOCOCscMatrix* Pcsc = get_csc_matrix(data->P);
    for (QOCOInt i = 0; i < get_nnz(data->P); ++i) {
      linsys_data->K->x[linsys_data->PregtoKKT[i]] = Pcsc->x[i];
    }
  }

  // Update A in KKT matrix.
  QOCOCscMatrix* Acsc = get_csc_matrix(data->A);
  for (QOCOInt i = 0; i < get_nnz(data->A); ++i) {
    linsys_data->K->x[linsys_data->AttoKKT[data->AtoAt[i]]] =
        Acsc->x[i];
  }

  // Update G in KKT matrix.
  QOCOCscMatrix* Gcsc = get_csc_matrix(data->G);
  for (QOCOInt i = 0; i < get_nnz(data->G); ++i) {
    linsys_data->K->x[linsys_data->GttoKKT[data->GtoGt[i]]] =
        Gcsc->x[i];
  }
}

static void qdldl_cleanup(LinSysData* linsys_data)
{
  free_qoco_csc_matrix(linsys_data->K);
  qoco_free(linsys_data->p);
  qoco_free(linsys_data->pinv);
  qoco_free(linsys_data->etree);
  qoco_free(linsys_data->Lnz);
  qoco_free(linsys_data->Lx);
  qoco_free(linsys_data->Lp);
  qoco_free(linsys_data->Li);
  qoco_free(linsys_data->D);
  qoco_free(linsys_data->Dinv);
  qoco_free(linsys_data->iwork);
  qoco_free(linsys_data->bwork);
  qoco_free(linsys_data->fwork);
  qoco_free(linsys_data->xyzbuff1);
  qoco_free(linsys_data->xyzbuff2);
  qoco_free(linsys_data->nt2kkt);
  qoco_free(linsys_data->ntdiag2kkt);
  qoco_free(linsys_data->PregtoKKT);
  qoco_free(linsys_data->AttoKKT);
  qoco_free(linsys_data->GttoKKT);
  qoco_free(linsys_data);
}

// Export the backend struct
LinSysBackend backend = {.linsys_setup = qdldl_setup,
                         .linsys_initialize_nt = qdldl_initialize_nt,
                         .linsys_update_nt = qdldl_update_nt,
                         .linsys_update_data = qdldl_update_data,
                         .linsys_factor = qdldl_factor,
                         .linsys_solve = qdldl_solve,
                         .linsys_cleanup = qdldl_cleanup};
