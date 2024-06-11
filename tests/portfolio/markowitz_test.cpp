#include "test_utils.h"
#include "gtest/gtest.h"
#include "markowitz_data.h"

TEST(portfolio_test, markowitz)
{
    // Allocate and set sparse matrix data.
    QCOSCscMatrix* P;
    QCOSCscMatrix* A;
    QCOSCscMatrix* G;
    if(markowitz_P_nnz > 0) {
        P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(P, markowitz_n, markowitz_n, markowitz_P_nnz, markowitz_P_x, markowitz_P_p, markowitz_P_i);
    }
    else {
        P = nullptr;
    }
    if(markowitz_A_nnz > 0) {
        A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(A, markowitz_p, markowitz_n, markowitz_A_nnz, markowitz_A_x, markowitz_A_p, markowitz_A_i);
    }
    else {
        A = nullptr;
    }
    if(markowitz_G_nnz > 0) {
        G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(G, markowitz_m, markowitz_n, markowitz_G_nnz, markowitz_G_x, markowitz_G_p, markowitz_G_i);
    }
    else {
        G = nullptr;
    }
    QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
    set_default_settings(settings);
    settings->verbose = 1;
    QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

    QCOSInt exit = qcos_setup(solver, markowitz_n, markowitz_m, markowitz_p, P, markowitz_c, A, markowitz_b, G, markowitz_h, markowitz_l, markowitz_nsoc, markowitz_q, settings);
    ASSERT_EQ(exit, QCOS_NO_ERROR);

    exit = qcos_solve(solver);
    ASSERT_EQ(exit, QCOS_SOLVED);

    // Expect relative error of objective to be less that 0.01%
    expect_rel_error(solver->sol->obj, markowitz_objopt, 1e-4);

    // Cleanup memory allocations. 
    qcos_cleanup(solver);
    free(P);
    free(A);
    free(G);
}
