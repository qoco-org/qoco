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
  /** KKT matrix in CSC form (size N_exp x N_exp). */
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

  /** Positive diagonal mask in the unpermuted expanded KKT ordering. */
  unsigned char* positive_diag;

  QOCOFloat* fwork;

  /** Buffer of size N_exp. */
  QOCOFloat* xyzbuff1;

  /** Buffer of size N_exp. */
  QOCOFloat* xyzbuff2;

  /** Mapping from WtW entries (LP + dense SOC + sparse SOC diagonal) to KKT. */
  QOCOInt* nt2kkt;

  /** Mapping from all m diagonal entries to KKT (for regularization). */
  QOCOInt* ntdiag2kkt;

  /** Mapping from regularized P to KKT. */
  QOCOInt* PregtoKKT;

  /** Mapping from At to KKT. */
  QOCOInt* AttoKKT;

  /** Mapping from Gt to KKT. */
  QOCOInt* GttoKKT;

  /** Number of entries in nt2kkt: l + dense SOC upper-tri + sparse SOC
   * diagonal. */
  QOCOInt Wnnz;

  /** Number of sparse SOC cones. */
  QOCOInt nsoc_sparse;

  /** Per-SOC sparse flag (length nsoc). */
  QOCOInt* soc_is_sparse;

  /** Expanded system size: n + p + m + 2*nsoc_sparse. */
  QOCOInt N_exp;

  /** Total elements in u/v vectors: sum q[i] for sparse SOCs. */
  QOCOInt nt_sparse_nnz;

  /** Index into u/v arrays for each sparse SOC (length nsoc_sparse). */
  QOCOInt* sparse_soc_nt_idx;

  /** Mapping from u vector entries to KKT (length nt_sparse_nnz). */
  QOCOInt* nt_u2kkt;

  /** Mapping from v vector entries to KKT (length nt_sparse_nnz). */
  QOCOInt* nt_v2kkt;

  /** Mapping for extra 2x2 diagonal per sparse SOC (length 2*nsoc_sparse). */
  QOCOInt* nt_uvdiag2kkt;

  /** Static regularization for the (1,1) P block. */
  QOCOFloat kkt_static_reg_P;

  /** Static regularization for the (2,2) A block. */
  QOCOFloat kkt_static_reg_A;

  /** Static regularization for the (3,3) G block. */
  QOCOFloat kkt_static_reg_G;
};

static LinSysData* qdldl_setup(QOCOProblemData* data, QOCOSettings* settings,
                               QOCOInt Wnnz, QOCOInt nsoc_sparse,
                               QOCOInt* soc_is_sparse, QOCOInt nt_sparse_nnz,
                               QOCOInt* sparse_soc_nt_idx)
{
  QOCOInt N = data->n + data->m + data->p;
  QOCOInt N_exp = N + 2 * nsoc_sparse;

  // Compute Wnnz for QDLDL: LP + dense SOC upper-tri + sparse SOC diagonal.
  QOCOInt* q = get_data_vectori(data->q);
  QOCOInt Wnnz_qdldl = Wnnz;
  for (QOCOInt i = 0; i < data->nsoc; ++i) {
    if (soc_is_sparse && soc_is_sparse[i]) {
      QOCOInt qi = q[i];
      Wnnz_qdldl -= (qi * qi - qi) / 2;
    }
  }

  LinSysData* linsys_data = malloc(sizeof(LinSysData));

  linsys_data->N_exp = N_exp;
  linsys_data->Wnnz = Wnnz_qdldl;
  linsys_data->nsoc_sparse = nsoc_sparse;
  linsys_data->nt_sparse_nnz = nt_sparse_nnz;
  linsys_data->kkt_static_reg_P = settings->kkt_static_reg_P;
  linsys_data->kkt_static_reg_A = settings->kkt_static_reg_A;
  linsys_data->kkt_static_reg_G = settings->kkt_static_reg_G;

  linsys_data->soc_is_sparse =
      qoco_calloc(data->nsoc > 0 ? data->nsoc : 1, sizeof(QOCOInt));
  if (soc_is_sparse) {
    for (QOCOInt i = 0; i < data->nsoc; ++i)
      linsys_data->soc_is_sparse[i] = soc_is_sparse[i];
  }

  linsys_data->sparse_soc_nt_idx =
      qoco_calloc(nsoc_sparse > 0 ? nsoc_sparse : 1, sizeof(QOCOInt));
  if (sparse_soc_nt_idx) {
    for (QOCOInt i = 0; i < nsoc_sparse; ++i)
      linsys_data->sparse_soc_nt_idx[i] = sparse_soc_nt_idx[i];
  }

  linsys_data->xyzbuff1 = qoco_malloc(sizeof(QOCOFloat) * N_exp);
  linsys_data->xyzbuff2 = qoco_malloc(sizeof(QOCOFloat) * N_exp);

  linsys_data->etree = qoco_malloc(sizeof(QOCOInt) * N_exp);
  linsys_data->Lnz = qoco_malloc(sizeof(QOCOInt) * N_exp);
  linsys_data->Lp = qoco_malloc(sizeof(QOCOInt) * (N_exp + 1));
  linsys_data->D = qoco_malloc(sizeof(QOCOFloat) * N_exp);
  linsys_data->Dinv = qoco_malloc(sizeof(QOCOFloat) * N_exp);
  linsys_data->iwork = qoco_malloc(sizeof(QOCOInt) * 3 * N_exp);
  linsys_data->bwork = qoco_malloc(sizeof(unsigned char) * N_exp);
  linsys_data->positive_diag = qoco_calloc(N_exp, sizeof(unsigned char));
  linsys_data->fwork = qoco_malloc(sizeof(QOCOFloat) * N_exp);
  for (QOCOInt i = 0; i < data->n; ++i) {
    linsys_data->positive_diag[i] = 1;
  }
  for (QOCOInt i = 0; i < nsoc_sparse; ++i) {
    QOCOInt aux_idx = N + 2 * i;
    linsys_data->positive_diag[aux_idx] = 1;
  }

  linsys_data->nt2kkt = qoco_calloc(Wnnz_qdldl, sizeof(QOCOInt));
  linsys_data->ntdiag2kkt = qoco_calloc(data->m, sizeof(QOCOInt));
  linsys_data->PregtoKKT = qoco_calloc(get_nnz(data->P), sizeof(QOCOInt));
  linsys_data->AttoKKT = qoco_calloc(get_nnz(data->A), sizeof(QOCOInt));
  linsys_data->GttoKKT = qoco_calloc(get_nnz(data->G), sizeof(QOCOInt));
  linsys_data->nt_u2kkt =
      qoco_calloc(nt_sparse_nnz > 0 ? nt_sparse_nnz : 1, sizeof(QOCOInt));
  linsys_data->nt_v2kkt =
      qoco_calloc(nt_sparse_nnz > 0 ? nt_sparse_nnz : 1, sizeof(QOCOInt));
  linsys_data->nt_uvdiag2kkt =
      qoco_calloc(nsoc_sparse > 0 ? 2 * nsoc_sparse : 1, sizeof(QOCOInt));

  QOCOInt* nt2kkt_temp = qoco_calloc(Wnnz_qdldl, sizeof(QOCOInt));
  QOCOInt* ntdiag2kkt_temp = qoco_calloc(data->m, sizeof(QOCOInt));
  QOCOInt* PregtoKKT_temp =
      data->P ? qoco_calloc(get_nnz(data->P), sizeof(QOCOInt)) : NULL;
  QOCOInt* AttoKKT_temp = qoco_calloc(get_nnz(data->A), sizeof(QOCOInt));
  QOCOInt* GttoKKT_temp = qoco_calloc(get_nnz(data->G), sizeof(QOCOInt));
  QOCOInt* nt_u2kkt_temp =
      qoco_calloc(nt_sparse_nnz > 0 ? nt_sparse_nnz : 1, sizeof(QOCOInt));
  QOCOInt* nt_v2kkt_temp =
      qoco_calloc(nt_sparse_nnz > 0 ? nt_sparse_nnz : 1, sizeof(QOCOInt));
  QOCOInt* nt_uvdiag2kkt_temp =
      qoco_calloc(nsoc_sparse > 0 ? 2 * nsoc_sparse : 1, sizeof(QOCOInt));

  linsys_data->K = construct_kkt(
      data->P ? get_csc_matrix(data->P) : NULL, get_csc_matrix(data->A),
      get_csc_matrix(data->G), get_csc_matrix(data->At),
      get_csc_matrix(data->Gt), settings->kkt_static_reg_A, data->n, data->m,
      data->p, data->l, data->nsoc, q, PregtoKKT_temp, AttoKKT_temp,
      GttoKKT_temp, nt2kkt_temp, ntdiag2kkt_temp, Wnnz_qdldl, soc_is_sparse,
      nsoc_sparse, nt_sparse_nnz, sparse_soc_nt_idx, nt_u2kkt_temp,
      nt_v2kkt_temp, nt_uvdiag2kkt_temp);

  linsys_data->p = qoco_malloc(N_exp * sizeof(QOCOInt));
  linsys_data->pinv = qoco_malloc(N_exp * sizeof(QOCOInt));
  QOCOInt amd_status = amd_order(N_exp, linsys_data->K->p, linsys_data->K->i,
                                 linsys_data->p, (double*)NULL, (double*)NULL);
  if (amd_status < 0)
    return NULL;
  invert_permutation(linsys_data->p, linsys_data->pinv, N_exp);

  QOCOInt* KtoPKPt = qoco_malloc(linsys_data->K->nnz * sizeof(QOCOInt));
  QOCOCscMatrix* PKPt = csc_symperm(linsys_data->K, linsys_data->pinv, KtoPKPt);

  for (QOCOInt i = 0; i < Wnnz_qdldl; ++i)
    linsys_data->nt2kkt[i] = KtoPKPt[nt2kkt_temp[i]];
  for (QOCOInt i = 0; i < data->m; ++i)
    linsys_data->ntdiag2kkt[i] = KtoPKPt[ntdiag2kkt_temp[i]];
  if (data->P && PregtoKKT_temp) {
    for (QOCOInt i = 0; i < get_nnz(data->P); ++i)
      linsys_data->PregtoKKT[i] = KtoPKPt[PregtoKKT_temp[i]];
  }
  for (QOCOInt i = 0; i < get_nnz(data->A); ++i)
    linsys_data->AttoKKT[i] = KtoPKPt[AttoKKT_temp[i]];
  for (QOCOInt i = 0; i < get_nnz(data->G); ++i)
    linsys_data->GttoKKT[i] = KtoPKPt[GttoKKT_temp[i]];
  for (QOCOInt i = 0; i < nt_sparse_nnz; ++i) {
    linsys_data->nt_u2kkt[i] = KtoPKPt[nt_u2kkt_temp[i]];
    linsys_data->nt_v2kkt[i] = KtoPKPt[nt_v2kkt_temp[i]];
  }
  QOCOInt n_uvdiag = 2 * nsoc_sparse;
  for (QOCOInt i = 0; i < n_uvdiag; ++i)
    linsys_data->nt_uvdiag2kkt[i] = KtoPKPt[nt_uvdiag2kkt_temp[i]];

  free_qoco_csc_matrix(linsys_data->K);
  qoco_free(KtoPKPt);
  qoco_free(nt2kkt_temp);
  qoco_free(ntdiag2kkt_temp);
  qoco_free(PregtoKKT_temp);
  qoco_free(AttoKKT_temp);
  qoco_free(GttoKKT_temp);
  qoco_free(nt_u2kkt_temp);
  qoco_free(nt_v2kkt_temp);
  qoco_free(nt_uvdiag2kkt_temp);
  linsys_data->K = PKPt;

  QOCOInt sumLnz =
      QDLDL_etree(N_exp, linsys_data->K->p, linsys_data->K->i,
                  linsys_data->iwork, linsys_data->Lnz, linsys_data->etree);
  if (sumLnz < 0)
    return NULL;

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
               kkt_dynamic_reg, linsys_data->positive_diag);
}

/**
 * @brief Computes norm(K_true*x - b, inf) against the unregularized KKT
 * matrix using the current solution in linsys_data->xyzbuff1 (permuted space).
 * Stores the residual vector in x_scratch (size N = n+p+m) and returns the
 * inf-norm.
 */
static QOCOFloat compute_linsys_residual(LinSysData* linsys_data,
                                         QOCOWorkspace* work, QOCOFloat* b,
                                         QOCOFloat* x_scratch)
{
  QOCOInt N = linsys_data->K->n;
  (void)work;

  for (QOCOInt i = 0; i < N; ++i) {
    linsys_data->xyzbuff2[i] = 0.0;
  }

  for (QOCOInt col = 0; col < N; ++col) {
    for (QOCOInt p = linsys_data->K->p[col]; p < linsys_data->K->p[col + 1];
         ++p) {
      QOCOInt row = linsys_data->K->i[p];
      QOCOFloat val = linsys_data->K->x[p];
      linsys_data->xyzbuff2[row] += val * linsys_data->xyzbuff1[col];
      if (row != col) {
        linsys_data->xyzbuff2[col] += val * linsys_data->xyzbuff1[row];
      }
    }
  }

  for (QOCOInt k = 0; k < N; ++k) {
    x_scratch[k] = b[k] - linsys_data->xyzbuff2[k];
  }

  return inf_norm(x_scratch, N);
}

#ifdef QOCO_LOGGING
static void log_linsys_error(LinSysData* linsys_data, QOCOWorkspace* work,
                             QOCOFloat* b, QOCOFloat* x_scratch,
                             const char* label, FILE* f)
{
  QOCOFloat res = compute_linsys_residual(linsys_data, work, b, x_scratch);
  fprintf(f, "  (%s): %.4e\n", label, res);
}
#endif

static void qdldl_solve(LinSysData* linsys_data, QOCOWorkspace* work,
                        QOCOVectorf* b_vec, QOCOVectorf* x_vec,
                        QOCOFloat ir_tol, QOCOInt max_ir_iters)
{
  QOCOFloat* b = get_data_vectorf(b_vec);
  QOCOFloat* x = get_data_vectorf(x_vec);
  QOCOInt N_base = work->data->n + work->data->p + work->data->m;
  for (QOCOInt i = N_base; i < linsys_data->K->n; ++i) {
    b[i] = 0.0;
  }

  // Permute b and store in xyzbuff.
  for (QOCOInt i = 0; i < linsys_data->K->n; ++i) {
    linsys_data->xyzbuff1[i] = b[linsys_data->p[i]];
  }

  // Copy permuted b into b.
  copy_arrayf(linsys_data->xyzbuff1, b, linsys_data->K->n);

  // Triangular solve.
  QDLDL_solve(linsys_data->K->n, linsys_data->Lp, linsys_data->Li,
              linsys_data->Lx, linsys_data->Dinv, linsys_data->xyzbuff1);

#ifdef QOCO_LOGGING
  FILE* log_f = fopen("qoco_log.txt", "a");
  if (log_f) {
    log_linsys_error(linsys_data, work, b, x, "initial solve", log_f);
  }
#endif

  // Adaptive iterative refinement with best-solution tracking.
  // x is used as scratch for residual computation (size K->n).
  // best_sol (work->xyzbuff1) saves the permuted solution with the lowest
  // residual seen so far; if a step worsens the residual we restore it.
  QOCOFloat* best_sol = get_data_vectorf(work->xyzbuff1);
  QOCOFloat best_res = compute_linsys_residual(linsys_data, work, b, x);
  copy_arrayf(linsys_data->xyzbuff1, best_sol, linsys_data->K->n);

  QOCOInt ir_count = 0;
  QOCOFloat res = best_res;

  for (QOCOInt i = 0; i < max_ir_iters; ++i) {
    if (res < ir_tol) {
      break;
    }

    // r = b - K*x is already in permuted space in x from
    // compute_linsys_residual.
    copy_arrayf(x, linsys_data->xyzbuff2, linsys_data->K->n);

    // dx = K \ r
    QDLDL_solve(linsys_data->K->n, linsys_data->Lp, linsys_data->Li,
                linsys_data->Lx, linsys_data->Dinv, linsys_data->xyzbuff2);

    // x_new = x_old + dx (permuted space).
    qoco_axpy(linsys_data->xyzbuff1, linsys_data->xyzbuff2,
              linsys_data->xyzbuff1, 1.0, linsys_data->K->n);

    QOCOFloat new_res = compute_linsys_residual(linsys_data, work, b, x);

#ifdef QOCO_LOGGING
    if (log_f) {
      log_linsys_error(linsys_data, work, b, x, "refinement", log_f);
    }
#endif

    if (new_res >= best_res) {
      // Residual worsened; restore best solution and stop.
      copy_arrayf(best_sol, linsys_data->xyzbuff1, linsys_data->K->n);
      break;
    }

    ir_count++;
    best_res = new_res;
    copy_arrayf(linsys_data->xyzbuff1, best_sol, linsys_data->K->n);
    res = new_res;
  }

  work->ir_iters += ir_count;

#ifdef QOCO_LOGGING
  if (log_f)
    fclose(log_f);
#endif

  for (QOCOInt i = 0; i < linsys_data->K->n; ++i) {
    x[linsys_data->p[i]] = linsys_data->xyzbuff1[i];
  }
}

static void qdldl_set_nt_identity(LinSysData* linsys_data, QOCOWorkspace* work)
{
  QOCOInt m = work->data->m;
  for (QOCOInt i = 0; i < linsys_data->Wnnz; ++i) {
    linsys_data->K->x[linsys_data->nt2kkt[i]] = 0;
  }

  for (QOCOInt i = 0; i < m; ++i) {
    linsys_data->K->x[linsys_data->ntdiag2kkt[i]] = -1.0;
  }

  for (QOCOInt i = 0; i < linsys_data->nt_sparse_nnz; ++i) {
    linsys_data->K->x[linsys_data->nt_u2kkt[i]] = 0.0;
    linsys_data->K->x[linsys_data->nt_v2kkt[i]] = 0.0;
  }
  for (QOCOInt i = 0; i < linsys_data->nsoc_sparse; ++i) {
    QOCOInt uvdiag_idx = 2 * i;
    linsys_data->K->x[linsys_data->nt_uvdiag2kkt[uvdiag_idx]] = 1.0;
    linsys_data->K->x[linsys_data->nt_uvdiag2kkt[uvdiag_idx + 1]] = -1.0;
  }
}

static void qdldl_update_nt(LinSysData* linsys_data, QOCOWorkspace* work,
                            QOCOFloat kkt_static_reg_G)
{
  QOCOFloat* WtW = get_data_vectorf(work->WtW);
  QOCOFloat* eta2 = get_data_vectorf(work->nt_eta2_sparse);
  QOCOFloat* d = get_data_vectorf(work->nt_d_sparse);
  QOCOFloat* u = get_data_vectorf(work->nt_u_sparse);
  QOCOFloat* v = get_data_vectorf(work->nt_v_sparse);
  QOCOInt* q = get_data_vectori(work->data->q);

  QOCOInt nt_src = 0;
  QOCOInt nt_dst = 0;
  for (QOCOInt i = 0; i < work->data->l; ++i) {
    linsys_data->K->x[linsys_data->nt2kkt[nt_dst++]] = -WtW[nt_src++];
  }

  QOCOInt sp_cone = 0;
  for (QOCOInt c = 0; c < work->data->nsoc; ++c) {
    QOCOInt qi = q[c];
    if (work->soc_is_sparse && work->soc_is_sparse[c]) {
      for (QOCOInt col = 0; col < qi; ++col) {
        for (QOCOInt row = 0; row <= col; ++row) {
          if (row == col) {
            QOCOFloat diag = eta2[sp_cone] * (row == 0 ? d[sp_cone] : 1.0);
            linsys_data->K->x[linsys_data->nt2kkt[nt_dst++]] = -diag;
          }
          nt_src++;
        }
      }
      QOCOInt sidx = get_element_vectori(work->sparse_soc_nt_idx, sp_cone);
      for (QOCOInt j = 0; j < qi; ++j) {
        linsys_data->K->x[linsys_data->nt_u2kkt[sidx + j]] =
            -eta2[sp_cone] * u[sidx + j];
        linsys_data->K->x[linsys_data->nt_v2kkt[sidx + j]] =
            -eta2[sp_cone] * v[sidx + j];
      }
      QOCOInt uvdiag_idx = 2 * sp_cone;
      linsys_data->K->x[linsys_data->nt_uvdiag2kkt[uvdiag_idx]] = eta2[sp_cone];
      linsys_data->K->x[linsys_data->nt_uvdiag2kkt[uvdiag_idx + 1]] =
          -eta2[sp_cone];
      sp_cone++;
    }
    else {
      QOCOInt ntri = (qi * qi + qi) / 2;
      for (QOCOInt j = 0; j < ntri; ++j) {
        linsys_data->K->x[linsys_data->nt2kkt[nt_dst++]] = -WtW[nt_src++];
      }
    }
  }

  // Regularize Nesterov-Todd block of KKT matrix.
  QOCOInt m = work->data->m;
  for (QOCOInt i = 0; i < m; ++i) {
    linsys_data->K->x[linsys_data->ntdiag2kkt[i]] -= kkt_static_reg_G;
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
    linsys_data->K->x[linsys_data->AttoKKT[data->AtoAt[i]]] = Acsc->x[i];
  }

  // Update G in KKT matrix.
  QOCOCscMatrix* Gcsc = get_csc_matrix(data->G);
  for (QOCOInt i = 0; i < get_nnz(data->G); ++i) {
    linsys_data->K->x[linsys_data->GttoKKT[data->GtoGt[i]]] = Gcsc->x[i];
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
  qoco_free(linsys_data->positive_diag);
  qoco_free(linsys_data->fwork);
  qoco_free(linsys_data->xyzbuff1);
  qoco_free(linsys_data->xyzbuff2);
  qoco_free(linsys_data->nt2kkt);
  qoco_free(linsys_data->ntdiag2kkt);
  qoco_free(linsys_data->PregtoKKT);
  qoco_free(linsys_data->AttoKKT);
  qoco_free(linsys_data->GttoKKT);
  qoco_free(linsys_data->soc_is_sparse);
  qoco_free(linsys_data->sparse_soc_nt_idx);
  qoco_free(linsys_data->nt_u2kkt);
  qoco_free(linsys_data->nt_v2kkt);
  qoco_free(linsys_data->nt_uvdiag2kkt);
  qoco_free(linsys_data);
}

static const char* qdldl_name() { return "builtin/qdldl"; }

// Export the backend struct
LinSysBackend backend = {.linsys_name = qdldl_name,
                         .linsys_setup = qdldl_setup,
                         .linsys_set_nt_identity = qdldl_set_nt_identity,
                         .linsys_update_nt = qdldl_update_nt,
                         .linsys_update_data = qdldl_update_data,
                         .linsys_factor = qdldl_factor,
                         .linsys_solve = qdldl_solve,
                         .linsys_cleanup = qdldl_cleanup};
