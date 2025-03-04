//------------------------------------------------------------------------------
// AMD/Source/amd_order: user-callable AMD ordering method
//------------------------------------------------------------------------------

// AMD, Copyright (c) 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and
// Iain S. Duff.  All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

/* User-callable AMD minimum degree ordering routine.  See amd.h for
 * documentation.
 */

#include "amd_internal.h"

/* ========================================================================= */
/* === AMD_order =========================================================== */
/* ========================================================================= */

int AMD_order(Int n, const Int Ap[], const Int Ai[], Int P[], double Control[],
              double Info[])
{
  Int *Len, *S, nz, i, *Pinv, info, status, *Rp, *Ri, *Cp, *Ci, ok;
  size_t nzaat, slen;
  double mem = 0;

#ifndef NDEBUG
  AMD_debug_init("amd");
#endif

  /* clear the Info array, if it exists */
  info = Info != (double*)NULL;
  if (info) {
    for (i = 0; i < AMD_INFO; i++) {
      Info[i] = EMPTY;
    }
    Info[AMD_N] = n;
    Info[AMD_STATUS] = AMD_OK;
  }

  /* make sure inputs exist and n is >= 0 */
  if (Ai == (Int*)NULL || Ap == (Int*)NULL || P == (Int*)NULL || n < 0) {
    if (info)
      Info[AMD_STATUS] = AMD_INVALID;
    return (AMD_INVALID); /* arguments are invalid */
  }

  if (n == 0) {
    return (AMD_OK); /* n is 0 so there's nothing to do */
  }

  nz = Ap[n];
  if (info) {
    Info[AMD_NZ] = nz;
  }
  if (nz < 0) {
    if (info)
      Info[AMD_STATUS] = AMD_INVALID;
    return (AMD_INVALID);
  }

  /* check if n or nz will cause integer overflow */
  if (((size_t)n) >= Int_MAX / sizeof(Int) ||
      ((size_t)nz) >= Int_MAX / sizeof(Int)) {
    if (info)
      Info[AMD_STATUS] = AMD_OUT_OF_MEMORY;
    return (AMD_OUT_OF_MEMORY); /* problem too large */
  }

  /* check the input matrix:	AMD_OK, AMD_INVALID, or AMD_OK_BUT_JUMBLED */
  status = AMD_valid(n, n, Ap, Ai);

  if (status == AMD_INVALID) {
    if (info)
      Info[AMD_STATUS] = AMD_INVALID;
    return (AMD_INVALID); /* matrix is invalid */
  }

  /* allocate two size-n integer workspaces */
  size_t nn = (size_t)n;
  Len = SuiteSparse_malloc(nn, sizeof(Int));
  Pinv = SuiteSparse_malloc(nn, sizeof(Int));
  mem += n;
  mem += n;
  if (!Len || !Pinv) {
    /* :: out of memory :: */
    SuiteSparse_free(Len);
    SuiteSparse_free(Pinv);
    if (info)
      Info[AMD_STATUS] = AMD_OUT_OF_MEMORY;
    return (AMD_OUT_OF_MEMORY);
  }

  if (status == AMD_OK_BUT_JUMBLED) {
    /* sort the input matrix and remove duplicate entries */
    AMD_DEBUG1(("Matrix is jumbled\n"));
    Rp = SuiteSparse_malloc(nn + 1, sizeof(Int));
    Ri = SuiteSparse_malloc(nz, sizeof(Int));
    mem += (n + 1);
    mem += MAX(nz, 1);
    if (!Rp || !Ri) {
      /* :: out of memory :: */
      SuiteSparse_free(Rp);
      SuiteSparse_free(Ri);
      SuiteSparse_free(Len);
      SuiteSparse_free(Pinv);
      if (info)
        Info[AMD_STATUS] = AMD_OUT_OF_MEMORY;
      return (AMD_OUT_OF_MEMORY);
    }
    /* use Len and Pinv as workspace to create R = A' */
    AMD_preprocess(n, Ap, Ai, Rp, Ri, Len, Pinv);
    Cp = Rp;
    Ci = Ri;
  }
  else {
    /* order the input matrix as-is.  No need to compute R = A' first */
    Rp = NULL;
    Ri = NULL;
    Cp = (Int*)Ap;
    Ci = (Int*)Ai;
  }

  /* --------------------------------------------------------------------- */
  /* determine the symmetry and count off-diagonal nonzeros in A+A' */
  /* --------------------------------------------------------------------- */

  nzaat = AMD_aat(n, Cp, Ci, Len, P, Info);
  AMD_DEBUG1(("nzaat: %g\n", (double)nzaat));
  ASSERT((MAX(nz - n, 0) <= nzaat) && (nzaat <= 2 * (size_t)nz));

  /* --------------------------------------------------------------------- */
  /* allocate workspace for matrix, elbow room, and 6 size-n vectors */
  /* --------------------------------------------------------------------- */

  S = NULL;
  slen = nzaat;                      /* space for matrix */
  ok = ((slen + nzaat / 5) >= slen); /* check for size_t overflow */
  slen += nzaat / 5;                 /* add elbow room */
  for (i = 0; ok && i < 7; i++) {
    ok = ((slen + nn) > slen); /* check for size_t overflow */
    slen += nn;                /* size-n elbow room, 6 size-n work */
  }
  mem += slen;
  ok = ok && (slen < SIZE_T_MAX / sizeof(Int)); /* check for overflow */
  if (ok) {
    S = SuiteSparse_malloc(slen, sizeof(Int));
  }
  AMD_DEBUG1(("slen %g\n", (double)slen));
  if (!S) {
    /* :: out of memory :: (or problem too large) */
    SuiteSparse_free(Rp);
    SuiteSparse_free(Ri);
    SuiteSparse_free(Len);
    SuiteSparse_free(Pinv);
    if (info)
      Info[AMD_STATUS] = AMD_OUT_OF_MEMORY;
    return (AMD_OUT_OF_MEMORY);
  }
  if (info) {
    /* memory usage, in bytes. */
    Info[AMD_MEMORY] = mem * sizeof(Int);
  }

  /* --------------------------------------------------------------------- */
  /* order the matrix */
  /* --------------------------------------------------------------------- */

  AMD_1(n, Cp, Ci, P, Pinv, Len, slen, S, Control, Info);

  /* --------------------------------------------------------------------- */
  /* free the workspace */
  /* --------------------------------------------------------------------- */

  SuiteSparse_free(Rp);
  SuiteSparse_free(Ri);
  SuiteSparse_free(Len);
  SuiteSparse_free(Pinv);
  SuiteSparse_free(S);
  if (info)
    Info[AMD_STATUS] = status;
  return (status); /* successful ordering */
}
