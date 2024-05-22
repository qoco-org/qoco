import utils.codegen_functions as cgen
from scipy import sparse
import numpy as np
import cvxpy as cp


def generate_markowitz():

    # Seed rng.
    rng = np.random.default_rng(12345)

    # Number of factors.
    m = 100

    # Number of assets.
    n = 500

    # Factor loading matrix.
    F = sparse.random(m, n, density=0.25, random_state=rng)
    Pfull = F.T @ F + 0.1 * sparse.eye(n)
    P = sparse.triu(Pfull, format='csc')
    c = -rng.random(n)

    A = np.ones((1, n))
    A = sparse.csc_matrix(A)
    b = np.array([1.])

    G = -sparse.eye(n)
    G = sparse.csc_matrix(G)
    h = np.zeros(n)

    l = n
    ncones = 0
    q = None
    # breakpoint()

    xvar = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(xvar, Pfull) + c.T @ xvar),
                      [G @ xvar <= h,
                       A @ xvar == b])
    prob.solve(verbose=True)
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(xvar.value)

    cgen.generate_data(P, c, A, b, G, h, l, ncones, q,
                       prob.value, "portfolio", "markowitz")
    cgen.generate_test("portfolio", "markowitz")


generate_markowitz()