.. _simple_example:

Simple Example
==============

We will solve the simple SOCP below

.. math::
  \begin{array}{ll}
    \mbox{minimize} & x_1^2+x_2^2+x_3^2+x_4 \\
    \mbox{subject to} & x_1+x_2=1 \\
    & x_2+x_3 = 1 \\
    & x_1 \geq 0 \\
    & \sqrt{x_3^2+x_4^2} \leq x_2
  \end{array}

This problem can be written as

.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T \begin{bmatrix}2 & 0 & 0 & 0\\ 0 & 2 & 0 & 0 \\  0 & 0 & 2 & 0 \\  0 & 0 & 0 & 2 \end{bmatrix} x + \begin{bmatrix}0 \\ 0 \\ 0 \\1\end{bmatrix}^T x \\ \\
    \mbox{subject to} & \begin{bmatrix} 1 & 1 & 0 & 0\\ 0 & 1 & 1 & 0\end{bmatrix} x =  \begin{bmatrix}1 \\ 1 \end{bmatrix} \\
    & \begin{bmatrix} -1 & 0 & 0 & 0\\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0\\ 0 & 0 & 0 & -1\\ \end{bmatrix} x \preceq_\mathcal{C}  \begin{bmatrix}0 \\ 0 \\ 0 \\ 0 \end{bmatrix} \\
  \end{array}

where :math:`\mathcal{C} = \mathbb{R} \times \mathcal{Q}^3`, so :code:`l = 1`, :code:`nsoc = 1`, and :code:`q = [3]`.


Python
-----------

.. code:: python

    import qoco
    import numpy as np
    from scipy import sparse

    # Define problem data
    P = sparse.diags([2, 2, 2, 0], 0).tocsc()

    c = np.array([0, 0, 0, 1])
    G = -sparse.identity(4).tocsc()
    h = np.zeros(4)
    A = sparse.csc_matrix([[1, 1, 0, 0], [0, 1, 1, 0]]).tocsc()
    b = np.array([1, 1])

    l = 1
    n = 4
    m = 4
    p = 2
    nsoc = 1
    q = np.array([3])

    # Create an QOCO object.
    prob = qoco.QOCO()

    # Setup workspace.
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q, verbose=True)

    # Solve problem.
    res = prob.solve()

C/C++
-------------------------
.. code:: c
  
  #include "qoco.h"

  int main()
  {
    QOCOInt p = 2;     // Number of affine equality constraints (rows of A).
    QOCOInt m = 4;     // Number of conic constraints (rows of G).
    QOCOInt n = 4;     // Number of optimization variables.
    QOCOInt l = 1;     // Dimension of non-negative orthant.
    QOCOInt nsoc = 1;  // Number of second-order cones.
    QOCOInt q[] = {3}; // Dimension of second-order cones.

    QOCOFloat Px[] = {2, 2, 2};     // Data for upper triangular part of P.
    QOCOInt Pnnz = 3;               // Number of nonzero elements.
    QOCOInt Pp[] = {0, 1, 2, 3, 3}; // Column pointers.
    QOCOInt Pi[] = {0, 1, 2};       // Row indices.

    QOCOFloat Ax[] = {1, 1, 1, 1};
    QOCOInt Annz = 4;
    QOCOInt Ap[] = {0, 1, 3, 4, 4};
    QOCOInt Ai[] = {0, 0, 1, 1};

    QOCOFloat Gx[] = {-1, -1, -1, -1};
    QOCOInt Gnnz = 4;
    QOCOInt Gp[] = {0, 1, 2, 3, 4};
    QOCOInt Gi[] = {0, 1, 2, 3};

    QOCOFloat c[] = {0, 0, 0, 1};
    QOCOFloat b[] = {1, 1};
    QOCOFloat h[] = {0, 0, 0, 0};

    // Allocate storage for data matrices.
    QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
    QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
    QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));

    // Set data matrices.
    qoco_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
    qoco_set_csc(A, p, n, Annz, Ax, Ap, Ai);
    qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

    // Allocate settings.
    QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));

    // Set default settings.
    set_default_settings(settings);
    settings->verbose = 1;

    // Allocate solver.
    QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));

    // Setup problem.
    QOCOInt exit =
        qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);

    // Solve problem.
    if (exit == QOCO_NO_ERROR) {
      exit = qoco_solve(solver);
    }

    // Free allocated memory.
    qoco_cleanup(solver);
    free(P);
    free(A);
    free(G);
  }