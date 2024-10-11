#include "test_utils.h"
#include "gtest/gtest.h"
#include "lcvx_bad_scaling_data.h"

TEST(ocp_test, lcvx_bad_scaling)
{
    // Allocate and set sparse matrix data.
    QOCOCscMatrix* P;
    QOCOCscMatrix* A;
    QOCOCscMatrix* G;
    if(lcvx_bad_scaling_P_nnz > 0) {
        P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(P, lcvx_bad_scaling_n, lcvx_bad_scaling_n, lcvx_bad_scaling_P_nnz, lcvx_bad_scaling_P_x, lcvx_bad_scaling_P_p, lcvx_bad_scaling_P_i);
    }
    else {
        P = nullptr;
    }
    if(lcvx_bad_scaling_A_nnz > 0) {
        A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(A, lcvx_bad_scaling_p, lcvx_bad_scaling_n, lcvx_bad_scaling_A_nnz, lcvx_bad_scaling_A_x, lcvx_bad_scaling_A_p, lcvx_bad_scaling_A_i);
    }
    else {
        A = nullptr;
    }
    if(lcvx_bad_scaling_G_nnz > 0) {
        G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(G, lcvx_bad_scaling_m, lcvx_bad_scaling_n, lcvx_bad_scaling_G_nnz, lcvx_bad_scaling_G_x, lcvx_bad_scaling_G_p, lcvx_bad_scaling_G_i);
    }
    else {
        G = nullptr;
    }
    QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
    set_default_settings(settings);
    settings->verbose = 1;
    QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));

    QOCOInt exit = qoco_setup(solver, lcvx_bad_scaling_n, lcvx_bad_scaling_m, lcvx_bad_scaling_p, P, lcvx_bad_scaling_c, A, lcvx_bad_scaling_b, G, lcvx_bad_scaling_h, lcvx_bad_scaling_l, lcvx_bad_scaling_nsoc, lcvx_bad_scaling_q, settings);
    ASSERT_EQ(exit, QOCO_NO_ERROR);

    exit = qoco_solve(solver);
    ASSERT_EQ(exit, QOCO_SOLVED);

    // Expect relative error of objective to be less than tolerance.
    expect_rel_error(solver->sol->obj, lcvx_bad_scaling_objopt, 0.0005);

    // Cleanup memory allocations. 
    qoco_cleanup(solver);
    free(settings);
    free(P);
    free(A);
    free(G);
}
