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

void soc_product(const QOCOFloat* u, const QOCOFloat* v, QOCOFloat* p,
                 QOCOInt n)
{
  p[0] = qoco_dot(u, v, n);
  for (QOCOInt i = 1; i < n; ++i) {
    p[i] = u[0] * v[i] + v[0] * u[i];
  }
}

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

QOCOFloat soc_residual(const QOCOFloat* u, QOCOInt n)
{
  QOCOFloat res = 0;
  for (QOCOInt i = 1; i < n; ++i) {
    res += u[i] * u[i];
  }
  res = qoco_sqrt(res) - u[0];
  return res;
}

QOCOFloat soc_residual2(const QOCOFloat* u, QOCOInt n)
{
  QOCOFloat res = u[0] * u[0];
  for (QOCOInt i = 1; i < n; ++i) {
    res -= u[i] * u[i];
  }
  return res;
}

void cone_product(const QOCOFloat* u, const QOCOFloat* v, QOCOFloat* p,
                  QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
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
                   QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
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

void bring2cone(QOCOFloat* u, QOCOProblemData* data)
{
  if (cone_residual(u, data->l, data->nsoc, data->q) >= 0) {
    QOCOFloat a = 0.0;

    // Get a for for LP cone.
    QOCOInt idx;
    for (idx = 0; idx < data->l; ++idx) {
      a = qoco_max(a, -u[idx]);
    }
    a = qoco_max(a, 0.0);

    // Get a for second-order cone.
    for (QOCOInt i = 0; i < data->nsoc; ++i) {
      QOCOFloat soc_res = soc_residual(&u[idx], data->q[i]);
      if (soc_res > 0 && soc_res > a) {
        a = soc_res;
      }
      idx += data->q[i];
    }

    // Compute u + (1 + a) * e for LP cone.
    for (idx = 0; idx < data->l; ++idx) {
      u[idx] += (1 + a);
    }

    // Compute u + (1 + a) * e for second-order cones.
    for (QOCOInt i = 0; i < data->nsoc; ++i) {
      u[idx] += (1 + a);
      idx += data->q[i];
    }
  }
}

void nt_multiply(QOCOFloat* W, QOCOFloat* x, QOCOFloat* z, QOCOInt l, QOCOInt m,
                 QOCOInt nsoc, QOCOInt* q)
{
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
    WtW[idx] =
        safe_div(get_element_vectorf(work->s, idx), get_element_vectorf(work->z, idx));
    W[idx] = qoco_sqrt(WtW[idx]);
    Wfull[idx] = W[idx];
    Winv[idx] = safe_div(1.0, W[idx]);
    Winvfull[idx] = Winv[idx];
  }

  // Compute Nesterov-Todd scaling for second-order cones.
  QOCOInt nt_idx = idx;
  QOCOInt nt_idx_full = idx;
  for (QOCOInt i = 0; i < work->data->nsoc; ++i) {
    // Compute normalized vectors.
    QOCOFloat s_scal =
        soc_residual2(get_pointer_vectorf(work->s, idx), work->data->q[i]);
    s_scal = qoco_sqrt(s_scal);
    QOCOFloat f = safe_div(1.0, s_scal);
    scale_arrayf(get_pointer_vectorf(work->s, idx), sbar, f,
                 work->data->q[i]);

    QOCOFloat z_scal =
        soc_residual2(get_pointer_vectorf(work->z, idx), work->data->q[i]);
    z_scal = qoco_sqrt(z_scal);
    f = safe_div(1.0, z_scal);
    scale_arrayf(get_pointer_vectorf(work->z, idx), zbar, f,
                 work->data->q[i]);

    QOCOFloat gamma = qoco_sqrt(
        0.5 * (1 + qoco_dot(sbar, zbar, work->data->q[i])));

    f = safe_div(1.0, (2 * gamma));

    // Overwrite sbar with wbar.
    sbar[0] = f * (sbar[0] + zbar[0]);
    for (QOCOInt j = 1; j < work->data->q[i]; ++j) {
      sbar[j] = f * (sbar[j] - zbar[j]);
    }

    // Overwrite zbar with v.
    f = safe_div(1.0, qoco_sqrt(2 * (sbar[0] + 1)));
    zbar[0] = f * (sbar[0] + 1.0);
    for (QOCOInt j = 1; j < work->data->q[i]; ++j) {
      zbar[j] = f * sbar[j];
    }

    // Compute W for second-order cones.
    QOCOInt shift = 0;
    f = qoco_sqrt(safe_div(s_scal, z_scal));
    QOCOFloat finv = safe_div(1.0, f);
    for (QOCOInt j = 0; j < work->data->q[i]; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        QOCOInt full_idx1 = nt_idx_full + j * work->data->q[i] + k;
        QOCOInt full_idx2 = nt_idx_full + k * work->data->q[i] + j;
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
    for (QOCOInt j = 0; j < work->data->q[i]; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        WtW[nt_idx + shift] = qoco_dot(
            &Wfull[nt_idx_full + j * work->data->q[i]],
            &Wfull[nt_idx_full + k * work->data->q[i]], work->data->q[i]);
        shift += 1;
      }
    }

    idx += work->data->q[i];
    nt_idx += (work->data->q[i] * work->data->q[i] + work->data->q[i]) / 2;
    nt_idx_full += work->data->q[i] * work->data->q[i];
  }

  // Compute scaled variable lambda. lambda = W * z.
  nt_multiply(Wfull, get_pointer_vectorf(work->z, 0), lambda,
              work->data->l, work->data->m, work->data->nsoc, work->data->q);
}

void compute_centering(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOFloat* xyz = get_data_vectorf(work->xyz);
  QOCOFloat* Ds = get_data_vectorf(work->Ds);
  QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);
  QOCOFloat* ubuff2 = get_data_vectorf(work->ubuff2);
  QOCOFloat* Dzaff = &xyz[work->data->n + work->data->p];
  QOCOFloat a = qoco_min(
      linesearch(get_pointer_vectorf(work->z, 0), Dzaff, 1.0, solver),
      linesearch(get_pointer_vectorf(work->s, 0), Ds, 1.0, solver));

  // Compute rho. rho = ((s + a * Ds)'*(z + a * Dz)) / (s'*z).
  qoco_axpy(Dzaff, get_pointer_vectorf(work->z, 0), ubuff1, a,
            work->data->m);
  qoco_axpy(Ds, get_pointer_vectorf(work->s, 0), ubuff2, a,
            work->data->m);
  QOCOFloat rho = qoco_dot(ubuff1, ubuff2, work->data->m) /
                  qoco_dot(get_pointer_vectorf(work->z, 0),
                           get_pointer_vectorf(work->s, 0), work->data->m);

  // Compute sigma. sigma = max(0, min(1, rho))^3.
  QOCOFloat sigma = qoco_min(1.0, rho);
  sigma = qoco_max(0.0, sigma);
  sigma = sigma * sigma * sigma;
  solver->work->sigma = sigma;
}

QOCOFloat linesearch(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                     QOCOSolver* solver)
{
  if (solver->work->data->nsoc == 0) {
    return exact_linesearch(u, Du, f, solver);
  }
  else {
    return bisection_search(u, Du, f, solver);
  }
}

QOCOFloat bisection_search(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                           QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);

  QOCOFloat al = 0.0;
  QOCOFloat au = 1.0;
  QOCOFloat a = 0.0;
  for (QOCOInt i = 0; i < solver->settings->bisect_iters; ++i) {
    a = 0.5 * (al + au);
    qoco_axpy(Du, u, ubuff1, safe_div(a, f), work->data->m);
    if (cone_residual(ubuff1, work->data->l, work->data->nsoc,
                      work->data->q) >= 0) {
      au = a;
    }
    else {
      al = a;
    }
  }
  return al;
}

QOCOFloat exact_linesearch(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                           QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;

  QOCOFloat a = 1.0;
  QOCOFloat minval = 0;

  // Compute a for LP cones.
  for (QOCOInt i = 0; i < work->data->l; ++i) {
    if (Du[i] < minval * u[i])
      minval = Du[i] / u[i];
  }

  if (-f < minval)
    a = f;
  else
    a = -f / minval;

  return a;
}
