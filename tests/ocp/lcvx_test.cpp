#include "test_utils.h"
#include "gtest/gtest.h"
#include "lcvx_data.h"

TEST(ocp_test, lcvx)
{
    // Allocate and set sparse matrix data.
    QOCOCscMatrix* P;
    QOCOCscMatrix* A;
    QOCOCscMatrix* G;
    if(lcvx_P_nnz > 0) {
        P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(P, lcvx_n, lcvx_n, lcvx_P_nnz, lcvx_P_x, lcvx_P_p, lcvx_P_i);
    }
    else {
        P = nullptr;
    }
    if(lcvx_A_nnz > 0) {
        A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(A, lcvx_p, lcvx_n, lcvx_A_nnz, lcvx_A_x, lcvx_A_p, lcvx_A_i);
    }
    else {
        A = nullptr;
    }
    if(lcvx_G_nnz > 0) {
        G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(G, lcvx_m, lcvx_n, lcvx_G_nnz, lcvx_G_x, lcvx_G_p, lcvx_G_i);
    }
    else {
        G = nullptr;
    }
    QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
    set_default_settings(settings);
    settings->verbose = 1;
    QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));

    QOCOInt exit = qoco_setup(solver, lcvx_n, lcvx_m, lcvx_p, P, lcvx_c, A, lcvx_b, G, lcvx_h, lcvx_l, lcvx_nsoc, lcvx_q, settings);
    ASSERT_EQ(exit, QOCO_NO_ERROR);

    exit = qoco_solve(solver);
    ASSERT_EQ(exit, QOCO_SOLVED);

    // Expect relative error of objective to be less than tolerance.
    expect_rel_error(solver->sol->obj, lcvx_objopt, 0.0001);

    // Cleanup memory allocations. 
    qoco_cleanup(solver);
    free(settings);
    free(P);
    free(A);
    free(G);
}
