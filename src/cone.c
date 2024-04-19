#include "cone.h"

void cone_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p,
                  QCOSProblemData* data)
{
  QCOSInt idx;
  // Compute LP cone product.
  for (idx = 0; idx < data->l; ++idx) {
    p[idx] = u[idx] * v[idx];
  }

  // Compute second-order cone product.
  for (QCOSInt i = 0; i < data->ncones; ++i) {
    soc_product(&u[idx], &v[idx], &p[idx], data->q[i]);
    idx += data->q[i];
  }
}

void soc_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p, QCOSInt n)
{
  p[0] = dot(u, v, n);
  for (QCOSInt i = 1; i < n; ++i) {
    p[i] = u[0] * v[i] + v[0] * u[i];
  }
}

QCOSFloat soc_residual(QCOSFloat* u, QCOSInt n)
{
  QCOSFloat res = 0;
  for (QCOSInt i = 1; i < n; ++i) {
    res += u[i] * u[i];
  }
  res = qcos_sqrt(res) - u[0];
  return res;
}

QCOSFloat cone_residual(QCOSFloat* u, QCOSProblemData* data)
{
  QCOSFloat res = -1e7;

  // Compute LP cone residuals.
  QCOSInt idx;
  for (idx = 0; idx < data->l; ++idx) {
    res = qcos_max(-u[idx], res);
    // If res is positive can just return here.
  }

  // Compute second-order cone residual.
  for (QCOSInt i = 0; i < data->ncones; ++i) {
    res = qcos_max(res, soc_residual(&u[idx], data->q[i]));
    idx += data->q[i];
  }

  return res;
}

void bring2cone(QCOSFloat* u, QCOSProblemData* data)
{
  if (cone_residual(u, data) >= 0) {
    QCOSFloat a = 0.0;

    // Get a for for LP cone.
    QCOSInt idx;
    for (idx = 0; idx < data->l; ++idx) {
      a = qcos_max(a, -u[idx]);
    }
    a = qcos_max(a, 0.0);

    // Get a for second-order cone.
    for (QCOSInt i = 0; i < data->ncones; ++i) {
      QCOSFloat soc_res = soc_residual(&u[idx], data->q[i]);
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
    for (QCOSInt i = 0; i < data->ncones; ++i) {
      u[idx] += (1 + a);
      idx += data->q[i];
    }
  }
}