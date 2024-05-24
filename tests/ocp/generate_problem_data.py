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
    zi = np.array([10,10,100,0,0,0])

    # Terminal condition.
    zf = np.zeros(nx)

    # State cost matrix.
    Q = sparse.eye(nx)

    # Input cost matrix.
    R = 5.0 * sparse.eye(nu)

    # Max inf norm on velocity.
    vmax = 10.0

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
    bbc = np.hstack((zi, zf))

    # Combine dynamics and boundary conditions into equality constraint matrix A, and vector b.
    A = np.block([[Adyn],[Abc]])
    A = sparse.csc_matrix(A)
    b = np.hstack((bdyn, bbc))
    p, _ = A.shape

    Gub = np.kron(np.eye(N), np.block([np.zeros((3,3)), np.eye(3)]))
    Glb = np.kron(np.eye(N), np.block([np.zeros((3,3)), -np.eye(3)]))
    G = np.block([[Gub],[Glb]])
    h = vmax * np.ones((6 * N))
    G = np.block([G, np.zeros((nx * N, nu*(N-1)))])
    G = sparse.csc_matrix(G)
    m, _ = G.shape
    l = m

    q = None
    nsoc = 0

    # Solve with cvxpy.
    zvar = cp.Variable((nx, N))
    uvar = cp.Variable((nu, N - 1))
    obj = 0
    con = [zvar[:, 0] == zi, zvar[:,N - 1] == zf]
    for i in range(N - 1):
        obj += (1/2)*(cp.quad_form(zvar[:,i], Q) + cp.quad_form(uvar[:,i], R))
        con += [zvar[:, i + 1] == Ad @ zvar[:, i] + Bd @ uvar[:, i] + g]
        con += [cp.norm_inf(zvar[4:6,:]) <= vmax]
    obj += (1/2)*(cp.quad_form(zvar[:,N - 1], Q))
    prob = cp.Problem(cp.Minimize(obj), con)
    prob.solve(verbose=True)

    # # data, chain, inverse_data = prob.get_problem_data(cp.SCS)
    # unorm = np.zeros(N - 1)
    # for i in range(N - 1):
    #     unorm[i] = np.linalg.norm(uvar.value[:, i])
    # fig = plt.figure()
    # # syntax for 3-D projection
    # ax = plt.axes(projection ='3d')
    # # plotting
    # ax.plot3D(zvar.value[0,:], zvar.value[1,:], zvar.value[2,:], 'green')
    # plt.figure()
    # plt.plot(unorm)
    # plt.title("Thrust magnitude")

    # plt.figure()
    # plt.plot(zvar.value[3,:])
    # plt.plot(zvar.value[4,:])
    # plt.plot(zvar.value[5,:])

    # plt.show()

    # Generate data file for unit test.
    cgen.generate_data(n, m, p, P, c, A, b, G, h, l, nsoc, q,
                       prob.value, "ocp", "pdg")
    cgen.generate_test("ocp", "pdg")


generate_pdg()
