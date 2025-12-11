
.. _lcvx_example:

Lossless Convexification
========================

We will solve losslessly convexified powered descent guidance problem (given below) with QOCO (called through CVXPY) and qoco_custom (generated with QOCOGEN and CVXPYgen)

.. math::
    \begin{split}
        \underset{x, z, u, \sigma}{\text{minimize}} 
        \quad & -z_T  \\
        \text{subject to} 
        \quad & x_{k+1} = Ax_k + Bu_k + g \quad \forall k \in [0, T-1] \\  
        \quad & z_{k+1} = z_k - \alpha \sigma_k \Delta t \quad \forall k \in [0, T-1] \\  
        \quad & \|u_k\|_2 \leq \sigma_k \quad \forall k \in [0, T-1] \\
        \quad & \log(m_{\mathrm{wet}} - \alpha\rho_2 k \Delta t) \leq z_k \leq \log(m_{\mathrm{wet}} - \alpha\rho_1 k \Delta t) \quad \forall k \in [0, T-1] \\
        \quad & \mu_{1,k}\left[1-[z_k-z_{0,k}] + \frac{[z_k-z_{0,k}]^2}{2}\right] \leq \sigma_k \leq \mu_{2,k}[1-(z_k-z_{0,k})] \quad \forall k \in [0, T-1] \\ 
        \quad & e_3^\top u_k \geq \sigma_k \cos(\theta_{\mathrm{max}}) \quad \forall k \in [0, T-1] \\
        \quad & x_0 = x_{\mathrm{init}}, \; z_0 = \log(m_{\mathrm{wet}}), \; z_T \geq \log(m_{\mathrm{dry}})
    \end{split}


CVXPY/CVXPYgen
--------------
.. code:: python

    import cvxpy as cp
    import numpy as np
    from cvxpygen import cpg
    import time, sys

    T = 15
    tspan = 20
    dt = tspan / (T - 1)
    x0 = cp.Parameter(6)
    g = 9.807
    tvc_max = np.deg2rad(45.0)
    rho1 = 100.0
    rho2 = 500.0
    m_dry = 25.0
    m_fuel = 10.0
    Isp = 100.0

    g0 = 9.807
    m0 = m_dry + m_fuel
    a = 1 / (Isp * g0)
    nx = 6
    nu = 3

    A = np.array(
        [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [0.5 * dt**2, 0.0, 0.0],
            [0.0, 0.5 * dt**2, 0.0],
            [0.0, 0.0, 0.5 * dt**2],
            [dt, 0.0, 0.0],
            [0.0, dt, 0.0],
            [0.0, 0.0, dt],
        ]
    )
    G = np.array([0.0, 0.0, -0.5 * g * dt**2, 0.0, 0.0, -g * dt])
    xT = np.zeros((nx))

    x = cp.Variable((nx, T + 1))
    z = cp.Variable(T + 1)
    u = cp.Variable((nu, T))
    s = cp.Variable(T)

    # Objective
    obj = -z[T]

    # IC and TC
    con = [x[:, 0] == x0]
    con += [x[:, T] == xT]
    con += [z[0] == np.log(m0)]
    con += [z[T] >= np.log(m_dry)]

    # Dynamics
    for k in range(T):
        con += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + G]
        con += [z[k + 1] == z[k] - a * s[k] * dt]

    # State and Input Constraints
    for k in range(T):
        z0 = np.log(m0 - (a * rho2 * k * dt))
        mu1 = rho1 * np.exp(-z0)
        mu2 = rho2 * np.exp(-z0)
        con += [cp.norm(u[:, k]) <= s[k]]
        con += [mu1 * (1.0 - (z[k] - z0) + 0.5 * (z[k] - z0) ** 2) <= s[k]]
        con += [s[k] <= mu2 * (1.0 - (z[k] - z0))]
        con += [np.log(m0 - a * rho2 * k * dt) <= z[k]]
        con += [z[k] <= np.log(m0 - a * rho1 * k * dt)]
        con += [u[2, k] >= s[k] * np.cos(tvc_max)]

    prob = cp.Problem(cp.Minimize(obj), con)

    # Set initial condition
    x0.value = np.array(
            [
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                np.random.uniform(200, 400),
                0.0,
                0.0,
                0.0,
            ]
        )

    # Generate code with CVXPYgen/QOCOGEN (Can replace 'QOCOGEN' with 'QOCO' below to generate CVXPYgen solver with QOCO)
    cpg.generate_code(prob, code_dir='lcvx_qocogen', solver='QOCOGEN')

    # Solve problem with CVXPY/QOCO
    t0 = time.time()
    val = prob.solve(solver='QOCO', verbose=True)
    t1 = time.time()
    sys.stdout.write('QOCO\nSolve time via CVXPY: %.3f ms\n' % (1000*(t1-t0)))
    sys.stdout.write('Objective function value: %.6f\n' % val)

    # Solve problem with CVXPYgen/QOCOGEN
    t0 = time.time()
    val = prob.solve(method='CPG', verbose=True)
    t1 = time.time()
    sys.stdout.write('QOCOGEN \nSolve time via CVXPYgen: %.3f ms\n' % (1000 * (t1 - t0)))
    sys.stdout.write('Objective function value: %.6f\n' % val)