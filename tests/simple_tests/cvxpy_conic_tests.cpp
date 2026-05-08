// Replicates the TestQOCO class in cvxpy/cvxpy/tests/test_conic_solvers.py.
// Each test embeds the QOCO-canonicalized problem (n, m, p, l, nsoc, q, c, b,
// h, P, A, G) extracted from the corresponding SolverTestHelper in
// cvxpy/cvxpy/tests/solver_test_helpers.py. Expected objectives match the
// values asserted by those helpers (places=4, except socp_3 which uses
// places=3).

#include "test_utils.h"
#include "gtest/gtest.h"

#include "qoco.h"

// test_qoco_lp_0: min |x|_1 + 1 s.t. x == 0.
TEST(cvxpy_qoco, lp_0)
{
  QOCOInt n = 4;
  QOCOInt m = 4;
  QOCOInt p = 2;
  QOCOInt l = 4;
  QOCOInt nsoc = 0;

  QOCOFloat c[] = {1.0, 1.0, 0.0, 0.0};
  QOCOFloat b[] = {0.0, 0.0};
  QOCOFloat h[] = {0.0, 0.0, 0.0, 0.0};

  QOCOFloat Ax[] = {1.0, 1.0};
  QOCOInt Annz = 2;
  QOCOInt Ai[] = {0, 1};
  QOCOInt Ap[] = {0, 0, 0, 1, 2};

  QOCOFloat Gx[] = {-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
  QOCOInt Gnnz = 8;
  QOCOInt Gi[] = {0, 2, 1, 3, 0, 2, 1, 3};
  QOCOInt Gp[] = {0, 2, 4, 6, 8};

  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, A, b, G, h, l, nsoc,
                            nullptr, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  // QOCO canonicalization adds the constant via offset; assert obj relative to
  // the canonical objective value (the +1 constant is folded by cvxpy).
  ASSERT_EQ(exit, QOCO_SOLVED);
  ASSERT_NEAR(solver->sol->obj, 0.0, 1e-4);

  qoco_cleanup(solver);
  free(settings);
  free(A);
  free(G);
}

// test_qoco_lp_1: cvxopt LP example.
TEST(cvxpy_qoco, lp_1)
{
  QOCOInt n = 2;
  QOCOInt m = 4;
  QOCOInt p = 0;
  QOCOInt l = 4;
  QOCOInt nsoc = 0;

  QOCOFloat c[] = {-4.0, -5.0};
  QOCOFloat h[] = {3.0, 3.0, 0.0, 0.0};

  QOCOFloat Gx[] = {2.0, 1.0, -1.0, 1.0, 2.0, -1.0};
  QOCOInt Gnnz = 6;
  QOCOInt Gi[] = {0, 1, 2, 0, 1, 3};
  QOCOInt Gp[] = {0, 3, 6};

  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, nullptr, nullptr, G, h,
                            l, nsoc, nullptr, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  ASSERT_EQ(exit, QOCO_SOLVED);
  ASSERT_NEAR(solver->sol->obj, -9.0, 1e-4);

  qoco_cleanup(solver);
  free(settings);
  free(G);
}

// test_qoco_lp_2.
TEST(cvxpy_qoco, lp_2)
{
  QOCOInt n = 2;
  QOCOInt m = 2;
  QOCOInt p = 1;
  QOCOInt l = 2;
  QOCOInt nsoc = 0;

  QOCOFloat c[] = {1.0, 0.5};
  QOCOFloat b[] = {1.0};
  QOCOFloat h[] = {100.0, -10.0};

  QOCOFloat Ax[] = {1.0};
  QOCOInt Annz = 1;
  QOCOInt Ai[] = {0};
  QOCOInt Ap[] = {0, 0, 1};

  QOCOFloat Gx[] = {-1.0, 1.0};
  QOCOInt Gnnz = 2;
  QOCOInt Gi[] = {0, 1};
  QOCOInt Gp[] = {0, 2, 2};

  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, A, b, G, h, l, nsoc,
                            nullptr, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  ASSERT_EQ(exit, QOCO_SOLVED);
  ASSERT_NEAR(solver->sol->obj, -99.5, 1e-4);

  qoco_cleanup(solver);
  free(settings);
  free(A);
  free(G);
}

// test_qoco_lp_5: LP with redundant equality constraints.
// Optimal obj is exactly c.dot(x0) where x0 = [0,1,0,2,0,4,0,5,6,7].
// On precision-edge platforms (Apple Silicon Release with FMA) the IPM may
// terminate with QOCO_SOLVED_INACCURATE — accept either.
TEST(cvxpy_qoco, lp_5)
{
  QOCOInt n = 10;
  QOCOInt m = 10;
  QOCOInt p = 6;
  QOCOInt l = 10;
  QOCOInt nsoc = 0;

  QOCOFloat c[] = {-4.926446816771104,    4.673285093099831,
                   1.6919194550096637,    -5.670356153465352,
                   5.638138441328613,     -2.4313726050645545,
                   5.202043552859098,     1.6174640297730951,
                   3.3440177545046414,    1.3595594267149234};
  QOCOFloat b[] = {2.470922448583264,  -2.0937344345899698, 11.89809328155599,
                   -1.386215454982766, 11.05577431529512,   0.9709149473650798};
  QOCOFloat h[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  QOCOFloat Ax[] = {
      1.764052345967664,    0.144043571160878,    -2.5529898158340787,
      0.1549474256969163,   -1.8523781256493193,  0.46925769898897884,
      0.4001572083672233,   1.454273506962975,    0.6536185954403606,
      0.37816251960217356,  1.6804518654936917,   0.9110876474879132,
      0.9787379841057392,   0.7610377251469934,   0.8644361988595057,
      -0.8877857476301128,  1.4116112779102117,   0.6347727821822979,
      2.240893199201458,    0.12167501649282841,  -0.7421650204064419,
      -1.980796468223927,   -0.37105888259305064, 0.47314842709726285,
      1.8675579901499675,   0.44386323274542566,  2.2697546239876076,
      -0.3479121493261526,  3.012865446913048,    1.049425182005064,
      -0.977277879876411,   0.33367432737426683,  -1.4543656745987648,
      0.15634896910398005,  -1.5228004444634997,  -0.33228631932002595,
      0.9500884175255894,   1.4940790731576061,   0.04575851730144607,
      1.2302906807277207,   1.4538095328489122,   1.2039897493259242,
      -0.1513572082976979,  -0.20515826376580087, -0.1871838500258336,
      1.2023798487844113,   -0.09858135190248023, 0.02982120126641364,
      -0.10321885179355784, 0.31306770165090136,  1.5327792143584575,
      -0.3873268174079523,  1.5800073460221271,   0.19370282226632146,
      0.41059850193837233,  -0.8540957393017248,  1.469358769900285,
      -0.30230275057533557, 1.0316435251744525,   -0.12409245956675032};
  QOCOInt Annz = 60;
  QOCOInt Ai[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
                  2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
                  4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
  QOCOInt Ap[] = {0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60};

  QOCOFloat Gx[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  QOCOInt Gnnz = 10;
  QOCOInt Gi[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  QOCOInt Gp[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, A, b, G, h, l, nsoc,
                            nullptr, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  ASSERT_TRUE(exit == QOCO_SOLVED || exit == QOCO_SOLVED_INACCURATE);
  ASSERT_NEAR(solver->sol->obj, 21.275425028808698, 1e-4);

  qoco_cleanup(solver);
  free(settings);
  free(A);
  free(G);
}

// test_qoco_qp_0: min x^2 s.t. x >= 1.
TEST(cvxpy_qoco, qp_0)
{
  QOCOInt n = 1;
  QOCOInt m = 1;
  QOCOInt p = 0;
  QOCOInt l = 1;
  QOCOInt nsoc = 0;

  QOCOFloat c[] = {0.0};
  QOCOFloat h[] = {-1.0};

  QOCOFloat Px[] = {2.0};
  QOCOInt Pnnz = 1;
  QOCOInt Pi[] = {0};
  QOCOInt Pp[] = {0, 1};

  QOCOFloat Gx[] = {-1.0};
  QOCOInt Gnnz = 1;
  QOCOInt Gi[] = {0};
  QOCOInt Gp[] = {0, 1};

  QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, P, c, nullptr, nullptr, G, h, l,
                            nsoc, nullptr, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  ASSERT_EQ(exit, QOCO_SOLVED);
  ASSERT_NEAR(solver->sol->obj, 1.0, 1e-4);

  qoco_cleanup(solver);
  free(settings);
  free(P);
  free(G);
}

// test_qoco_socp_0: min |x|_2 + 1 s.t. x == 0.
TEST(cvxpy_qoco, socp_0)
{
  QOCOInt n = 3;
  QOCOInt m = 3;
  QOCOInt p = 2;
  QOCOInt l = 0;
  QOCOInt nsoc = 1;
  QOCOInt q[] = {3};

  QOCOFloat c[] = {1.0, 0.0, 0.0};
  QOCOFloat b[] = {0.0, 0.0};
  QOCOFloat h[] = {0.0, 0.0, 0.0};

  QOCOFloat Ax[] = {1.0, 1.0};
  QOCOInt Annz = 2;
  QOCOInt Ai[] = {0, 1};
  QOCOInt Ap[] = {0, 0, 1, 2};

  QOCOFloat Gx[] = {-1.0, -1.0, -1.0};
  QOCOInt Gnnz = 3;
  QOCOInt Gi[] = {0, 1, 2};
  QOCOInt Gp[] = {0, 1, 2, 3};

  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, A, b, G, h, l, nsoc,
                            q, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  // Canonical objective is the variable epigraphing |x|_2; cvxpy folds the +1
  // as a constant offset so the canonical solver obj is 0.
  ASSERT_EQ(exit, QOCO_SOLVED);
  ASSERT_NEAR(solver->sol->obj, 0.0, 1e-4);

  qoco_cleanup(solver);
  free(settings);
  free(A);
  free(G);
}

// test_qoco_socp_1.
TEST(cvxpy_qoco, socp_1)
{
  QOCOInt n = 4;
  QOCOInt m = 6;
  QOCOInt p = 0;
  QOCOInt l = 2;
  QOCOInt nsoc = 1;
  QOCOInt q[] = {4};

  QOCOFloat c[] = {3.0, 2.0, 1.0, 0.0};
  QOCOFloat h[] = {-1.0, 5.0, 0.0, 0.0, 0.0, 0.0};

  QOCOFloat Gx[] = {-1.0, -1.0, -1.0, -1.0, -3.0, -1.0, 1.0, -1.0};
  QOCOInt Gnnz = 8;
  QOCOInt Gi[] = {0, 3, 0, 4, 0, 5, 1, 2};
  QOCOInt Gp[] = {0, 2, 4, 6, 8};

  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  // Mirror cvxpy's reltol=1e-8 override for this test.
  settings->reltol = 1e-8;
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, nullptr, nullptr, G, h,
                            l, nsoc, q, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  ASSERT_EQ(exit, QOCO_SOLVED);
  ASSERT_NEAR(solver->sol->obj, -13.548638904065102, 1e-4);

  qoco_cleanup(solver);
  free(settings);
  free(G);
}

// test_qoco_socp_2: SOCP-based reformulation of LP_1.
TEST(cvxpy_qoco, socp_2)
{
  QOCOInt n = 2;
  QOCOInt m = 5;
  QOCOInt p = 0;
  QOCOInt l = 3;
  QOCOInt nsoc = 1;
  QOCOInt q[] = {2};

  QOCOFloat c[] = {-4.0, -5.0};
  QOCOFloat h[] = {3.0, 0.0, 0.0, 3.0, 0.0};

  QOCOFloat Gx[] = {2.0, -1.0, -1.0, 1.0, -1.0, -2.0};
  QOCOInt Gnnz = 6;
  QOCOInt Gi[] = {0, 1, 4, 0, 2, 4};
  QOCOInt Gp[] = {0, 3, 6};

  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, nullptr, nullptr, G, h,
                            l, nsoc, q, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  ASSERT_EQ(exit, QOCO_SOLVED);
  ASSERT_NEAR(solver->sol->obj, -9.0, 1e-4);

  qoco_cleanup(solver);
  free(settings);
  free(G);
}

// test_qoco_socp_3 axis=0. cvxpy uses places=3 here.
TEST(cvxpy_qoco, socp_3ax0)
{
  QOCOInt n = 2;
  QOCOInt m = 9;
  QOCOInt p = 0;
  QOCOInt l = 0;
  QOCOInt nsoc = 3;
  QOCOInt q[] = {3, 3, 3};

  QOCOFloat c[] = {-1.0, 2.0};
  QOCOFloat h[] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0};

  QOCOFloat Gx[] = {-1.0,
                    0.4999999999999999,
                    -1.0,
                    -0.2,
                    -1.0,
                    -0.4999999999999999,
                    -1.0,
                    -1.8};
  QOCOInt Gnnz = 8;
  QOCOInt Gi[] = {1, 2, 4, 7, 1, 2, 5, 8};
  QOCOInt Gp[] = {0, 4, 8};

  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, nullptr, nullptr, G, h,
                            l, nsoc, q, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  ASSERT_EQ(exit, QOCO_SOLVED);
  ASSERT_NEAR(solver->sol->obj, -1.932105, 1e-3);

  qoco_cleanup(solver);
  free(settings);
  free(G);
}

// test_qoco_socp_3 axis=1. Canonicalizes to the same standard form as ax0.
TEST(cvxpy_qoco, socp_3ax1)
{
  QOCOInt n = 2;
  QOCOInt m = 9;
  QOCOInt p = 0;
  QOCOInt l = 0;
  QOCOInt nsoc = 3;
  QOCOInt q[] = {3, 3, 3};

  QOCOFloat c[] = {-1.0, 2.0};
  QOCOFloat h[] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0};

  QOCOFloat Gx[] = {-1.0,
                    0.4999999999999999,
                    -1.0,
                    -0.2,
                    -1.0,
                    -0.4999999999999999,
                    -1.0,
                    -1.8};
  QOCOInt Gnnz = 8;
  QOCOInt Gi[] = {1, 2, 4, 7, 1, 2, 5, 8};
  QOCOInt Gp[] = {0, 4, 8};

  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, nullptr, nullptr, G, h,
                            l, nsoc, q, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  ASSERT_EQ(exit, QOCO_SOLVED);
  ASSERT_NEAR(solver->sol->obj, -1.932105, 1e-3);

  qoco_cleanup(solver);
  free(settings);
  free(G);
}
