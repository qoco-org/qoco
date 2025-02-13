from scipy import sparse
from datetime import datetime


def write_int(f, x, name):
    f.write("QOCOInt %s = %i;\n" % (name, x))


def write_float(f, x, name):
    f.write("QOCOFloat %s = %.17g;\n" % (name, x))


def write_vector_int(f, x, name):
    if x is None or len(x) == 0:
        f.write("QOCOInt* %s = nullptr;\n" % name)
    else:
        f.write("QOCOInt %s[%i] = { \n" % (name, len(x)))
        for i in x:
            f.write("    %i,\n" % i)
        f.write("};\n")


def write_vector_float(f, x, name):
    if x is None or len(x) == 0:
        f.write("QOCOFloat* %s = nullptr;\n" % name)
    else:
        f.write("QOCOFloat %s[%i] = { \n" % (name, len(x)))
        for i in x:
            f.write("    %.17g,\n" % i)
        f.write("};\n")


def write_csc_matrix(f, M, name):
    if M is None:
        write_int(f, 0, name + "_nnz")
        write_vector_float(f, None, name + "_x")
        write_vector_int(f, None, name + "_i")
        write_vector_int(f, None, name + "_p")
    else:
        write_int(f, M.nnz, name + "_nnz")
        write_vector_float(f, M.data, name + "_x")
        write_vector_int(f, M.indices, name + "_i")
        write_vector_int(f, M.indptr, name + "_p")


def generate_test(problem_name, test_name, tol, **kwargs):
    # Create test file.
    f = open(problem_name + "/" + test_name + "_test.cpp", "w")

    # Print header.
    f.write('#include "test_utils.h"\n')
    f.write('#include "gtest/gtest.h"\n')
    f.write('#include "' + test_name + "_" + 'data.h"\n\n')
    f.write("TEST(%s, %s)\n" % (problem_name + "_test", test_name))
    f.write("{\n")
    # Allocate and set sparse matrix data.
    f.write("    // Allocate and set sparse matrix data.\n")
    f.write("    QOCOCscMatrix* P;\n")
    f.write("    QOCOCscMatrix* A;\n")
    f.write("    QOCOCscMatrix* G;\n")
    f.write("    if(" + test_name + "_P_nnz > 0) {\n")
    f.write("        P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));\n")
    f.write(
        "        qoco_set_csc(P, "
        + test_name
        + "_n, "
        + test_name
        + "_n, "
        + test_name
        + "_P_nnz, "
        + test_name
        + "_P_x, "
        + test_name
        + "_P_p, "
        + test_name
        + "_P_i);\n"
    )
    f.write("    }\n")
    f.write("    else {\n")
    f.write("        P = nullptr;\n")
    f.write("    }\n")
    f.write("    if(" + test_name + "_A_nnz > 0) {\n")
    f.write("        A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));\n")
    f.write(
        "        qoco_set_csc(A, "
        + test_name
        + "_p, "
        + test_name
        + "_n, "
        + test_name
        + "_A_nnz, "
        + test_name
        + "_A_x, "
        + test_name
        + "_A_p, "
        + test_name
        + "_A_i);\n"
    )
    f.write("    }\n")
    f.write("    else {\n")
    f.write("        A = nullptr;\n")
    f.write("    }\n")
    f.write("    if(" + test_name + "_G_nnz > 0) {\n")
    f.write("        G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));\n")
    f.write(
        "        qoco_set_csc(G, "
        + test_name
        + "_m, "
        + test_name
        + "_n, "
        + test_name
        + "_G_nnz, "
        + test_name
        + "_G_x, "
        + test_name
        + "_G_p, "
        + test_name
        + "_G_i);\n"
    )
    f.write("    }\n")
    f.write("    else {\n")
    f.write("        G = nullptr;\n")
    f.write("    }\n")

    # Allocate settings struct.
    f.write(
        "    QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));\n"
    )
    f.write("    set_default_settings(settings);\n")
    f.write("    settings->verbose = 1;\n")

    # Manually specify ruiz iters. Needed to pass lcvx_badly scaled test.
    for key, value in kwargs.items():
        if key == "ruiz_iters" or key == "iter_ref_iters":
            f.write("    settings->%s = %d;\n" % (key, value))
    f.write("    QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));\n\n")
    f.write(
        "    QOCOInt exit = qoco_setup(solver, "
        + test_name
        + "_n, "
        + test_name
        + "_m, "
        + test_name
        + "_p, P, "
        + test_name
        + "_c, A, "
        + test_name
        + "_b, G, "
        + test_name
        + "_h, "
        + test_name
        + "_l, "
        + test_name
        + "_nsoc, "
        + test_name
        + "_q, settings);\n"
    )
    f.write("    ASSERT_EQ(exit, QOCO_NO_ERROR);\n\n")
    f.write("    exit = qoco_solve(solver);\n")
    f.write("    ASSERT_EQ(exit, QOCO_SOLVED);\n\n")
    f.write("    // Expect relative error of objective to be less than tolerance.\n")
    f.write(
        "    expect_rel_error(solver->sol->obj, "
        + test_name
        + "_objopt, "
        + str(tol)
        + ");\n\n"
    )

    # Cleanup memory allocations.
    f.write("    // Cleanup memory allocations. \n")
    f.write("    qoco_cleanup(solver);\n")
    f.write("    free(settings);\n")
    f.write("    free(P);\n")
    f.write("    free(A);\n")
    f.write("    free(G);\n")

    f.write("}\n")


def generate_data(
    n, m, p, P, c, A, b, G, h, l, nsoc, q, objopt, problem_name, test_name
):
    # Create data file.
    f = open(problem_name + "/" + test_name + "_" + "data.h", "w")

    # Print header.
    f.write(
        "// This file was autogenerated by the QOCO test suite on "
        + str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
        + "\n\n"
    )
    f.write('#include "qoco.h"\n\n')

    # Optimal objective.
    f.write("// Optimal objective.\n")
    write_float(f, objopt, test_name + "_objopt")
    f.write("\n")

    # Problem dimensions.
    f.write("// Problem dimensions.\n")
    write_int(f, n, test_name + "_n")
    write_int(f, m, test_name + "_m")
    write_int(f, p, test_name + "_p")
    write_int(f, l, test_name + "_l")
    write_int(f, nsoc, test_name + "_nsoc")
    write_vector_int(f, q, test_name + "_q")
    f.write("\n")

    # Cost data.
    f.write("// Cost data.\n")
    write_vector_float(f, c, test_name + "_c")
    f.write("\n")
    write_csc_matrix(f, P, test_name + "_P")

    # Constraint data.
    f.write("// Constraint data.\n")
    write_csc_matrix(f, A, test_name + "_A")
    write_vector_float(f, b, test_name + "_b")
    f.write("\n")
    write_csc_matrix(f, G, test_name + "_G")
    write_vector_float(f, h, test_name + "_h")

    f.close()
