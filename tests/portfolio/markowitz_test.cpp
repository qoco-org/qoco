#include "test_utils.h"
#include "gtest/gtest.h"
#include "markowitz_data.h"

TEST(portfolio_test, markowitz)
{
    // Allocate and set sparse matrix data.
    QOCOCscMatrix* P;
    QOCOCscMatrix* A;
    QOCOCscMatrix* G;
    if(markowitz_P_nnz > 0) {
        P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(P, markowitz_n, markowitz_n, markowitz_P_nnz, markowitz_P_x, markowitz_P_p, markowitz_P_i);
    }
    else {
        P = nullptr;
    }
    if(markowitz_A_nnz > 0) {
        A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(A, markowitz_p, markowitz_n, markowitz_A_nnz, markowitz_A_x, markowitz_A_p, markowitz_A_i);
    }
    else {
        A = nullptr;
    }
    if(markowitz_G_nnz > 0) {
        G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(G, markowitz_m, markowitz_n, markowitz_G_nnz, markowitz_G_x, markowitz_G_p, markowitz_G_i);
    }
    else {
        G = nullptr;
    }
    QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
    set_default_settings(settings);
    settings->verbose = 1;
    QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));

    QOCOInt exit = qoco_setup(solver, markowitz_n, markowitz_m, markowitz_p, P, markowitz_c, A, markowitz_b, G, markowitz_h, markowitz_l, markowitz_nsoc, markowitz_q, settings);
    ASSERT_EQ(exit, QOCO_NO_ERROR);

    exit = qoco_solve(solver);
    ASSERT_EQ(exit, QOCO_SOLVED);

    // Expect relative error of objective to be less than tolerance.
    expect_rel_error(solver->sol->obj, markowitz_objopt, 0.0001);

    // Cleanup memory allocations. 
    qoco_cleanup(solver);
    free(settings);
    free(P);
    free(A);
    free(G);
}
