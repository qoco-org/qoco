import utils.codegen_functions as cgen
from matplotlib import pyplot as plt
from scipy import sparse
import numpy as np
import cvxpy as cp


def generate_pdg():

    # Number of timesteps.
    N = 15

    # Discretization interval.
    dt = 1

    # Number of states.
    nx = 6

    # Number of control inputs.
    nu = 3

    # Gravitational acceleration.
    g0 = 9.8

    # Initial condition.
    xi = np.array([10,10,100,0,0,0])

    # Terminal condition.
    xf = np.zeros(nx)

    # State cost matrix.
    Q = sparse.eye(nx)

    # Input cost matrix.
    R = 5.0 * sparse.eye(nu)

    n = nx * N + nu * (N - 1)

    # Parse cost function.
    Qfull = sparse.kron(sparse.eye(N), Q)
    Rfull = sparse.kron(sparse.eye(N - 1), R)
    P = sparse.block_diag((Qfull, Rfull))
    P = sparse.triu(P, format='csc')
    c = np.zeros(nx * N + nu * (N - 1))

    # Double integrator dynamics.
    Ad = np.block([[np.eye(nu), dt*np.eye(nu)],[np.zeros((nu,nu)), np.eye(nu)]])
    Bd = np.block([[0.5*dt**2*np.eye(nu)],[dt*np.eye(nu)]])
    g = np.array([0, 0, -0.5*g0*dt**2, 0, 0, -g0*dt])

    # Parse dynamics constraint.
    Ax = np.block([np.kron(np.eye(N-1), Ad), np.zeros((nx*(N - 1), nx))]) - np.block([np.zeros((nx*(N-1), nx)), np.eye(nx*(N - 1))])
    Au = np.kron(np.eye(N-1), Bd)
    Adyn = np.block([Ax, Au])
    bdyn = np.kron(np.ones(N - 1), -g)

    # Parse boundary conditions.
    Aic = np.block([np.eye(nx), np.zeros((nx, nx * (N - 1))), np.zeros((nx, nu * (N - 1)))])
    Atc = np.block([np.zeros((nx, nx * (N - 1))), np.eye(nx), np.zeros((nx, nu * (N - 1)))])
    Abc = np.block([[Aic],[Atc]])
    bbc = np.hstack((xi, xf))

    # Combine dynamics and boundary conditions into equality constraint matrix A, and vector b.
    A = np.block([[Adyn],[Abc]])
    A = sparse.csc_matrix(A)
    b = np.hstack((bdyn, bbc))

    p, _ = A.shape
    m = 0

    G = None
    h = None
    l = 0
    q = None
    nsoc = 0

    # Solve with cvxpy.
    xvar = cp.Variable((nx, N))
    uvar = cp.Variable((nu, N - 1))
    obj = 0
    con = [xvar[:, 0] == xi, xvar[:,N - 1] == xf]
    for i in range(N - 1):
        obj += (1/2)*(cp.quad_form(xvar[:,i], Q) + cp.quad_form(uvar[:,i], R))
        con += [xvar[:, i + 1] == Ad @ xvar[:, i] + Bd @ uvar[:, i] + g]
    obj += (1/2)*(cp.quad_form(xvar[:,N - 1], Q))
    prob = cp.Problem(cp.Minimize(obj), con)
    prob.solve(verbose=True)

    # data, chain, inverse_data = prob.get_problem_data(cp.SCS)
    # unorm = np.zeros(N - 1)
    # for i in range(N - 1):
    #     unorm[i] = np.linalg.norm(uvar.value[:, i])
    # fig = plt.figure()
    # # syntax for 3-D projection
    # ax = plt.axes(projection ='3d')
    # # plotting
    # ax.plot3D(xvar.value[0,:], xvar.value[1,:], xvar.value[2,:], 'green')
    # plt.figure()
    # plt.plot(unorm)
    # plt.title("Thrust magnitude")
    # plt.show()

    # Generate data file for unit test.
    cgen.generate_data(n, m, p, P, c, A, b, G, h, l, nsoc, q,
                       prob.value, "ocp", "pdg")
    cgen.generate_test("ocp", "pdg")


generate_pdg()
