/**
 * @file cone.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "cone.h"
#include "qoco_utils.h"

void set_nt_scaling_identity(QOCOVectorf* nt_scaling, QOCOInt nt_scaling_nnz,
                             QOCOVectori* nt_scaling_soc_idx,
                             QOCOProblemData* data)
{
  (void)nt_scaling_soc_idx;
  QOCOFloat* nt_scaling_data = get_data_vectorf(nt_scaling);
  for (QOCOInt i = 0; i < nt_scaling_nnz; ++i) {
    nt_scaling_data[i] = 0.0;
  }
  for (QOCOInt i = 0; i < data->l; ++i) {
    nt_scaling_data[i] = 1.0;
  }
  QOCOInt idx = data->l;
  for (QOCOInt i = 0; i < data->nsoc; ++i) {
    QOCOInt qi = get_element_vectori(data->q, i);
    nt_scaling_data[idx] = 1.0;
    nt_scaling_data[idx + 1] = 1.0;
    for (QOCOInt k = 2; k <= qi; ++k) {
      nt_scaling_data[idx + k] = 0.0;
    }
    idx += qi + 1;
  }
}

/**
 * @brief Computes second-order cone product u * v = p.
 *
 * @param u u = (u0, u1) is a vector in second-order cone of dimension n.
 * @param v v = (v0, v1) is a vector in second-order cone of dimension n.
 * @param p Cone product of u and v.
 * @param n Dimension of second-order cone.
 */
void soc_product(const QOCOFloat* u, const QOCOFloat* v, QOCOFloat* p,
                 QOCOInt n)
{
  p[0] = qoco_dot(u, v, n);
  for (QOCOInt i = 1; i < n; ++i) {
    p[i] = u[0] * v[i] + v[0] * u[i];
  }
}

/**
 * @brief Commpues second-order cone division lambda # v = d
 *
 * @param lam lam = (lam0, lam1) is a vector in second-order cone of dimension
 * n.
 * @param v v = (v0, v1) is a vector in second-order cone of dimension n.
 * @param d Cone divisin of lam and v.
 * @param n Dimension of second-order cone.
 */
void soc_division(const QOCOFloat* lam, const QOCOFloat* v, QOCOFloat* d,
                  QOCOInt n)
{
  QOCOFloat f = lam[0] * lam[0] - qoco_dot(&lam[1], &lam[1], n - 1);
  QOCOFloat finv = safe_div(1.0, f);
  QOCOFloat lam0inv = safe_div(1.0, lam[0]);
  QOCOFloat lam1dv1 = qoco_dot(&lam[1], &v[1], n - 1);

  d[0] = finv * (lam[0] * v[0] - qoco_dot(&lam[1], &v[1], n - 1));
  for (QOCOInt i = 1; i < n; ++i) {
    d[i] = finv *
           (-lam[i] * v[0] + lam0inv * f * v[i] + lam0inv * lam1dv1 * lam[i]);
  }
}

/**
 * @brief Computes residual of vector u with respect to the second order cone of
 * dimension n.
 *
 * @param u u = (u0, u1) is a vector in second-order cone of dimension n.
 * @param n Dimension of second order cone.
 * @return Residual: norm(u1) - u0. Negative if the vector is in the cone and
 * positive otherwise.
 */
QOCOFloat soc_residual(const QOCOFloat* u, QOCOInt n)
{
  QOCOFloat res = 0;
  for (QOCOInt i = 1; i < n; ++i) {
    res += u[i] * u[i];
  }
  res = qoco_sqrt(res) - u[0];
  return res;
}

/**
 * @brief Computes u0^2 - u1'*u1 of vector u with respect to the second order
 * cone of dimension n.
 *
 * @param u u = (u0, u1) is a vector in second order cone of dimension n.
 * @param n Dimension of second order cone.
 * @return Residual: u0^2 - u1'*u1.
 */
QOCOFloat soc_residual2(const QOCOFloat* u, QOCOInt n)
{
  QOCOFloat res = u[0] * u[0];
  for (QOCOInt i = 1; i < n; ++i) {
    res -= u[i] * u[i];
  }
  return res;
}

void cone_product(const QOCOFloat* u, const QOCOFloat* v, QOCOFloat* p,
                  QOCOInt l, QOCOInt nsoc, const QOCOInt* q,
                  const QOCOInt* soc_idx)
{
  (void)soc_idx;
  QOCOInt idx;
  // Compute LP cone product.
  for (idx = 0; idx < l; ++idx) {
    p[idx] = u[idx] * v[idx];
  }

  // Compute second-order cone product.
  for (QOCOInt i = 0; i < nsoc; ++i) {
    soc_product(&u[idx], &v[idx], &p[idx], q[i]);
    idx += q[i];
  }
}

void cone_division(const QOCOFloat* lambda, const QOCOFloat* v, QOCOFloat* d,
                   QOCOInt l, QOCOInt nsoc, const QOCOInt* q,
                   const QOCOInt* soc_idx)
{
  (void)soc_idx;
  QOCOInt idx;
  // Compute LP cone division.
  for (idx = 0; idx < l; ++idx) {
    d[idx] = safe_div(v[idx], lambda[idx]);
  }

  // Compute second-order cone division.
  for (QOCOInt i = 0; i < nsoc; ++i) {
    soc_division(&lambda[idx], &v[idx], &d[idx], q[i]);
    idx += q[i];
  }
}

/**
 * @brief Computes residual of vector u with respect to cone C.
 *
 * @param u Vector to be tested.
 * @param l Dimension of LP cone.
 * @param nsoc Number of second-order cones.
 * @param q Dimension of each second-order cone.
 * @return Residual: Negative if the vector is in the cone and positive
 * otherwise.
 */
QOCOFloat cone_residual(const QOCOFloat* u, QOCOInt l, QOCOInt nsoc,
                        const QOCOInt* q)
{
  QOCOFloat res = -1e7;

  // Compute LP cone residuals.
  QOCOInt idx;
  for (idx = 0; idx < l; ++idx) {
    res = qoco_max(-u[idx], res);
    // If res is positive can just return here.
  }

  // Compute second-order cone residual.
  for (QOCOInt i = 0; i < nsoc; ++i) {
    res = qoco_max(res, soc_residual(&u[idx], q[i]));
    idx += q[i];
  }

  return res;
}

void bring2cone(QOCOFloat* u, QOCOInt* soc_idx, QOCOProblemData* data)
{
  (void)soc_idx;
  if (cone_residual(u, data->l, data->nsoc, get_data_vectori(data->q)) >= 0) {
    QOCOFloat a = 0.0;

    // Get a for for LP cone.
    QOCOInt idx;
    for (idx = 0; idx < data->l; ++idx) {
      a = qoco_max(a, -u[idx]);
    }
    a = qoco_max(a, 0.0);

    // Get a for second-order cone.
    for (QOCOInt i = 0; i < data->nsoc; ++i) {
      QOCOInt qi = get_element_vectori(data->q, i);
      QOCOFloat soc_res = soc_residual(&u[idx], qi);
      if (soc_res > 0 && soc_res > a) {
        a = soc_res;
      }
      idx += qi;
    }

    // Compute u + (1 + a) * e for LP cone.
    for (idx = 0; idx < data->l; ++idx) {
      u[idx] += (1 + a);
    }

    // Compute u + (1 + a) * e for second-order cones.
    for (QOCOInt i = 0; i < data->nsoc; ++i) {
      u[idx] += (1 + a);
      idx += get_element_vectori(data->q, i);
    }
  }
}

static void nt_multiply_impl(QOCOFloat* W, QOCOInt* nt_scaling_soc_idx,
                             QOCOInt* soc_idx, QOCOFloat* x, QOCOFloat* z,
                             QOCOInt l, QOCOInt m, QOCOInt nsoc, QOCOInt* q,
                             QOCOInt inverse)
{
  (void)nt_scaling_soc_idx;
  (void)soc_idx;
  (void)m;
  // Compute product for LP cone part of W.
  for (QOCOInt i = 0; i < l; ++i) {
    z[i] = inverse ? safe_div(x[i], W[i]) : (W[i] * x[i]);
  }

  // Compute product for second-order cones using fast O(m) operations
  // from equations (14) and (15) in the ECOS paper.
  QOCOInt nt_idx = l;
  QOCOInt idx = l;

  for (QOCOInt i = 0; i < nsoc; ++i) {
    QOCOFloat scale = inverse ? safe_div(1.0, W[nt_idx]) : W[nt_idx];
    QOCOFloat w0 = W[nt_idx + 1];
    QOCOFloat* w1 = &W[nt_idx + 2];
    QOCOFloat x0 = x[idx];
    QOCOFloat zeta = qoco_dot(w1, &x[idx + 1], q[i] - 1);
    QOCOFloat w0p1_inv = safe_div(1.0, 1.0 + w0);

    if (inverse) {
      z[idx] = scale * (w0 * x0 - zeta);
      QOCOFloat coeff = -x0 + zeta * w0p1_inv;
      for (QOCOInt j = 1; j < q[i]; ++j) {
        z[idx + j] = scale * (x[idx + j] + coeff * w1[j - 1]);
      }
    }
    else {
      z[idx] = scale * (w0 * x0 + zeta);
      QOCOFloat coeff = x0 + zeta * w0p1_inv;
      for (QOCOInt j = 1; j < q[i]; ++j) {
        z[idx + j] = scale * (x[idx + j] + coeff * w1[j - 1]);
      }
    }
    idx += q[i];
    nt_idx += q[i] + 1;
  }
}

void nt_multiply(QOCOFloat* W, QOCOInt* nt_scaling_soc_idx, QOCOInt* soc_idx,
                 QOCOFloat* x, QOCOFloat* z, QOCOInt l, QOCOInt m, QOCOInt nsoc,
                 QOCOInt* q)
{
  nt_multiply_impl(W, nt_scaling_soc_idx, soc_idx, x, z, l, m, nsoc, q, 0);
}

void nt_multiply_inv(QOCOFloat* W, QOCOInt* nt_scaling_soc_idx,
                     QOCOInt* soc_idx, QOCOFloat* x, QOCOFloat* z, QOCOInt l,
                     QOCOInt m, QOCOInt nsoc, QOCOInt* q)
{
  nt_multiply_impl(W, nt_scaling_soc_idx, soc_idx, x, z, l, m, nsoc, q, 1);
}

void compute_nt_scaling(QOCOWorkspace* work)
{
  QOCOFloat* W = get_data_vectorf(work->W);
  QOCOFloat* WtW = get_data_vectorf(work->WtW);
  QOCOFloat* nt_scaling = get_data_vectorf(work->nt_scaling);
  QOCOFloat* Winv = get_data_vectorf(work->Winv);
  QOCOFloat* sbar = get_data_vectorf(work->sbar);
  QOCOFloat* zbar = get_data_vectorf(work->zbar);
  QOCOFloat* lambda = get_data_vectorf(work->lambda);
  // Compute Nesterov-Todd scaling for LP cone.
  QOCOInt idx;
  for (idx = 0; idx < work->data->l; ++idx) {
    WtW[idx] = safe_div(get_element_vectorf(work->s, idx),
                        get_element_vectorf(work->z, idx));
    W[idx] = qoco_sqrt(WtW[idx]);
    nt_scaling[idx] = W[idx];
    Winv[idx] = safe_div(1.0, W[idx]);
  }

  // Compute Nesterov-Todd scaling for second-order cones.
  QOCOInt nt_idx = idx;
  QOCOInt nt_idx_full = idx;
  for (QOCOInt i = 0; i < work->data->nsoc; ++i) {
    QOCOInt qi = get_element_vectori(work->data->q, i);
    // Compute normalized vectors.
    QOCOFloat s_scal = soc_residual2(get_pointer_vectorf(work->s, idx), qi);
    s_scal = qoco_sqrt(s_scal);
    QOCOFloat f = safe_div(1.0, s_scal);
    scale_arrayf(get_pointer_vectorf(work->s, idx), sbar, f, qi);

    QOCOFloat z_scal = soc_residual2(get_pointer_vectorf(work->z, idx), qi);
    z_scal = qoco_sqrt(z_scal);
    f = safe_div(1.0, z_scal);
    scale_arrayf(get_pointer_vectorf(work->z, idx), zbar, f, qi);

    QOCOFloat gamma = qoco_sqrt(0.5 * (1 + qoco_dot(sbar, zbar, qi)));

    f = safe_div(1.0, (2 * gamma));

    // Overwrite sbar with wbar.
    sbar[0] = f * (sbar[0] + zbar[0]);
    for (QOCOInt j = 1; j < qi; ++j) {
      sbar[j] = f * (sbar[j] - zbar[j]);
    }

    // eta = sqrt(s_scal / z_scal)
    f = qoco_sqrt(safe_div(s_scal, z_scal));

    // Store fast scaling parameters: [eta, w0, w1[0], ..., w1[qi-2]]
    nt_scaling[nt_idx_full] = f;
    nt_scaling[nt_idx_full + 1] = sbar[0];
    for (QOCOInt j = 1; j < qi; ++j) {
      nt_scaling[nt_idx_full + 1 + j] = sbar[j];
    }

    // Compute W (upper triangular) and Winv for the sparse KKT update.
    // Also compute WtW = eta^2 * (2*w*w' - J) in upper triangular storage.
    QOCOFloat finv = safe_div(1.0, f);
    QOCOFloat eta2 = f * f;

    // Overwrite zbar with v (= Clarabel's normalized w vector), needed for W.
    QOCOFloat fv = safe_div(1.0, qoco_sqrt(2 * (sbar[0] + 1)));
    zbar[0] = fv * (sbar[0] + 1.0);
    for (QOCOInt j = 1; j < qi; ++j) {
      zbar[j] = fv * sbar[j];
    }

    QOCOInt shift = 0;
    for (QOCOInt j = 0; j < qi; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        W[nt_idx + shift] = 2 * (zbar[k] * zbar[j]);
        if (j != 0 && k == 0) {
          Winv[nt_idx + shift] = -W[nt_idx + shift];
        }
        else {
          Winv[nt_idx + shift] = W[nt_idx + shift];
        }
        if (j == k && j == 0) {
          W[nt_idx + shift] -= 1;
          Winv[nt_idx + shift] -= 1;
        }
        else if (j == k) {
          W[nt_idx + shift] += 1;
          Winv[nt_idx + shift] += 1;
        }
        W[nt_idx + shift] *= f;
        Winv[nt_idx + shift] *= finv;

        QOCOFloat val = eta2 * 2.0 * sbar[j] * sbar[k];
        if (j == k && j == 0) {
          val -= eta2;
        }
        else if (j == k) {
          val += eta2;
        }
        WtW[nt_idx + shift] = val;
        shift += 1;
      }
    }

    idx += qi;
    nt_idx += (qi * qi + qi) / 2;
    nt_idx_full += qi + 1;
  }

  // Compute scaled variable lambda. lambda = W * z.
  nt_multiply(nt_scaling, NULL, NULL, get_pointer_vectorf(work->z, 0), lambda,
              work->data->l, work->data->m, work->data->nsoc,
              get_data_vectori(work->data->q));
}

void compute_centering(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOFloat* xyz = get_data_vectorf(work->xyz);
  QOCOFloat* Ds = get_data_vectorf(work->Ds);
  QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);
  QOCOFloat* ubuff2 = get_data_vectorf(work->ubuff2);
  QOCOFloat* Dzaff = &xyz[work->data->n + work->data->p];
  QOCOFloat a =
      qoco_min(linesearch(get_pointer_vectorf(work->z, 0), Dzaff, 1.0, solver),
               linesearch(get_pointer_vectorf(work->s, 0), Ds, 1.0, solver));

  // Compute rho. rho = ((s + a * Ds)'*(z + a * Dz)) / (s'*z).
  qoco_axpy(Dzaff, get_pointer_vectorf(work->z, 0), ubuff1, a, work->data->m);
  qoco_axpy(Ds, get_pointer_vectorf(work->s, 0), ubuff2, a, work->data->m);
  QOCOFloat rho = qoco_dot(ubuff1, ubuff2, work->data->m) /
                  qoco_dot(get_pointer_vectorf(work->z, 0),
                           get_pointer_vectorf(work->s, 0), work->data->m);

  // Compute sigma. sigma = max(0, min(1, rho))^3.
  QOCOFloat sigma = qoco_min(1.0, rho);
  sigma = qoco_max(0.0, sigma);
  work->sigma = sigma * sigma * sigma;
}

/**
 * @brief Compute maximum step length alpha >= 0 such that
 * x + alpha * dx remains in the nonnegative orthant.
 */
QOCOFloat lp_step_length(const QOCOFloat* x, const QOCOFloat* dx, QOCOInt n,
                         QOCOFloat alpha_max)
{
  QOCOFloat alpha = alpha_max;

  for (QOCOInt i = 0; i < n; ++i) {
    if (dx[i] < 0.0) {
      alpha = qoco_min(alpha, -x[i] / dx[i]);
    }
  }

  return alpha;
}

/**
 * @brief Compute maximum step length alpha >= 0 such that
 * x + alpha * dx remains in the second-order cone.
 */
QOCOFloat soc_step_length(const QOCOFloat* x, const QOCOFloat* dx, QOCOInt n,
                          QOCOFloat alpha_max)
{
  const QOCOFloat two = 2.0;
  const QOCOFloat four = 4.0;

  QOCOFloat alpha = alpha_max;

  // ----------------------------------
  // Scalar safeguard: x0 + alpha dx0 >= 0
  // ----------------------------------
  if (x[0] >= 0.0 && dx[0] < 0.0) {
    QOCOFloat a = -x[0] / dx[0];
    if (a < alpha)
      alpha = a;
  }

  // ----------------------------------
  // Compute quadratic coefficients
  // ----------------------------------

  // a = dx0^2 - ||dx1||^2
  QOCOFloat dx1_norm2 = 0.0;
  for (QOCOInt i = 1; i < n; ++i) {
    dx1_norm2 += dx[i] * dx[i];
  }
  QOCOFloat a = dx[0] * dx[0] - dx1_norm2;

  // b = 2*(x0*dx0 - x1^T dx1)
  QOCOFloat x1dx1 = 0.0;
  for (QOCOInt i = 1; i < n; ++i) {
    x1dx1 += x[i] * dx[i];
  }
  QOCOFloat b = two * (x[0] * dx[0] - x1dx1);

  // c = x0^2 - ||x1||^2
  QOCOFloat x1_norm2 = 0.0;
  for (QOCOInt i = 1; i < n; ++i) {
    x1_norm2 += x[i] * x[i];
  }
  QOCOFloat c = x[0] * x[0] - x1_norm2;

  if (c < 0.0)
    c = 0.0; // numerical safeguard

  // ----------------------------------
  // Discriminant
  // ----------------------------------
  QOCOFloat d = b * b - four * a * c;

  // ----------------------------------
  // Case analysis (same as Clarabel)
  // ----------------------------------

  // No positive root → no restriction
  if ((a > 0.0 && b > 0.0) || d < 0.0) {
    return alpha;
  }

  // Linear case
  if (qoco_abs(a) < 1e-14) {
    return alpha;
  }

  // Boundary case
  if (c == 0.0) {
    if (a >= 0.0)
      return alpha;
    else
      return 0.0;
  }

  // ----------------------------------
  // Stable quadratic root computation
  // ----------------------------------

  QOCOFloat sqrt_d = qoco_sqrt(d);

  QOCOFloat t;
  if (b >= 0.0) {
    t = -b - sqrt_d;
  }
  else {
    t = -b + sqrt_d;
  }

  QOCOFloat r1 = (two * c) / t;
  QOCOFloat r2 = t / (two * a);

  // Keep only positive roots
  if (r1 < 0.0)
    r1 = QOCOFloat_MAX;
  if (r2 < 0.0)
    r2 = QOCOFloat_MAX;

  QOCOFloat r = qoco_min(r1, r2);

  if (r < alpha)
    alpha = r;

  return alpha;
}

QOCOFloat cone_step_length(QOCOFloat* u, QOCOFloat* Du, QOCOProblemData* data)
{
  QOCOFloat alpha = 1.0;

  QOCOInt idx = 0;

  // ---------------------------
  // LP cone
  // ---------------------------
  alpha = lp_step_length(&u[idx], &Du[idx], data->l, alpha);
  idx += data->l;

  // ---------------------------
  // SOC cones
  // ---------------------------
  for (QOCOInt i = 0; i < data->nsoc; ++i) {
    QOCOInt qi = get_element_vectori(data->q, i);

    QOCOFloat a = soc_step_length(&u[idx], &Du[idx], qi, alpha);

    if (a < alpha)
      alpha = a;

    idx += qi;
  }

  return alpha;
}

QOCOFloat linesearch(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                     QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;

  QOCOFloat alpha = cone_step_length(u, Du, work->data);

  // Safeguard against step-sizes which are too small. If alpha is very small,
  // it is likely that the step direction is of low quality, so stay at current
  // iterate (force alpha=0).
  if (alpha < 1e-12) {
    return 0;
  }

  return f * alpha;
}

void add_e(QOCOFloat* x, QOCOFloat a, QOCOInt l, QOCOInt nsoc, QOCOVectori* q)
{
  QOCOInt idx = 0;
  for (idx = 0; idx < l; ++idx) {
    x[idx] -= a;
  }
  for (QOCOInt i = 0; i < nsoc; ++i) {
    x[idx] -= a;
    idx += get_element_vectori(q, i);
  }
}