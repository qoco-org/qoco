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

void set_Wfull_identity(QOCOVectorf* Wfull, QOCOInt Wnnzfull,
                        QOCOVectori* Wsoc_idx, QOCOProblemData* data)
{
  (void)Wsoc_idx;
  QOCOFloat* Wfull_data = get_data_vectorf(Wfull);
  for (QOCOInt i = 0; i < Wnnzfull; ++i) {
    Wfull_data[i] = 0.0;
  }
  for (QOCOInt i = 0; i < data->l; ++i) {
    Wfull_data[i] = 1.0;
  }
  QOCOInt idx = data->l;
  for (QOCOInt i = 0; i < data->nsoc; ++i) {
    QOCOInt qi = get_element_vectori(data->q, i);
    for (QOCOInt k = 0; k < qi; ++k) {
      Wfull_data[idx + k * qi + k] = 1.0;
    }
    idx += qi * qi;
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

void nt_multiply(QOCOFloat* W, QOCOInt* Wsoc_idx, QOCOInt* soc_idx,
                 QOCOFloat* x, QOCOFloat* z, QOCOInt l, QOCOInt m, QOCOInt nsoc,
                 QOCOInt* q)
{
  (void)Wsoc_idx;
  (void)soc_idx;
  // Compute product for LP cone part of W.
  for (QOCOInt i = 0; i < l; ++i) {
    z[i] = (W[i] * x[i]);
  }

  // Compute product for second-order cones.
  QOCOInt nt_idx = l;
  QOCOInt idx = l;

  // Zero out second-order cone block of result z.
  for (QOCOInt i = l; i < m; ++i) {
    z[i] = 0;
  }

  // Loop over all second-order cones.
  for (QOCOInt i = 0; i < nsoc; ++i) {
    // Loop over elements within a second-order cone.
    for (QOCOInt j = 0; j < q[i]; ++j) {
      z[idx + j] += qoco_dot(&W[nt_idx + j * q[i]], &x[idx], q[i]);
    }
    idx += q[i];
    nt_idx += q[i] * q[i];
  }
}

void compute_nt_scaling(QOCOWorkspace* work)
{
  QOCOFloat* W = get_data_vectorf(work->W);
  QOCOFloat* WtW = get_data_vectorf(work->WtW);
  QOCOFloat* Wfull = get_data_vectorf(work->Wfull);
  QOCOFloat* Winv = get_data_vectorf(work->Winv);
  QOCOFloat* Winvfull = get_data_vectorf(work->Winvfull);
  QOCOFloat* sbar = get_data_vectorf(work->sbar);
  QOCOFloat* zbar = get_data_vectorf(work->zbar);
  QOCOFloat* lambda = get_data_vectorf(work->lambda);
  // Compute Nesterov-Todd scaling for LP cone.
  QOCOInt idx;
  for (idx = 0; idx < work->data->l; ++idx) {
    WtW[idx] = safe_div(get_element_vectorf(work->s, idx),
                        get_element_vectorf(work->z, idx));
    W[idx] = qoco_sqrt(WtW[idx]);
    Wfull[idx] = W[idx];
    Winv[idx] = safe_div(1.0, W[idx]);
    Winvfull[idx] = Winv[idx];
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

    // Overwrite zbar with v.
    f = safe_div(1.0, qoco_sqrt(2 * (sbar[0] + 1)));
    zbar[0] = f * (sbar[0] + 1.0);
    for (QOCOInt j = 1; j < qi; ++j) {
      zbar[j] = f * sbar[j];
    }

    // Compute W for second-order cones.
    QOCOInt shift = 0;
    f = qoco_sqrt(safe_div(s_scal, z_scal));
    QOCOFloat finv = safe_div(1.0, f);
    for (QOCOInt j = 0; j < qi; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        QOCOInt full_idx1 = nt_idx_full + j * qi + k;
        QOCOInt full_idx2 = nt_idx_full + k * qi + j;
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
        Wfull[full_idx1] = W[nt_idx + shift];
        Wfull[full_idx2] = W[nt_idx + shift];
        Winvfull[full_idx1] = Winv[nt_idx + shift];
        Winvfull[full_idx2] = Winv[nt_idx + shift];
        shift += 1;
      }
    }

    // Compute WtW for second-order cones.
    shift = 0;
    for (QOCOInt j = 0; j < qi; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        WtW[nt_idx + shift] = qoco_dot(&Wfull[nt_idx_full + j * qi],
                                       &Wfull[nt_idx_full + k * qi], qi);
        shift += 1;
      }
    }

    idx += qi;
    nt_idx += (qi * qi + qi) / 2;
    nt_idx_full += qi * qi;
  }

  // Compute scaled variable lambda. lambda = W * z.
  nt_multiply(Wfull, NULL, NULL, get_pointer_vectorf(work->z, 0), lambda,
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