Generating Custom Solver
------------------------
.. _generate:

Below is code that will generate a custom solver for the problem family defined in :ref:`simple example <simple_example>`.

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

    # Generate custom solver in this directory with name qoco_custom.
    prob.generate_solver(".", "qoco_custom")

You should now find your custom solver in the :code:`qoco_custom` directory within the current directory along with a file titled :code:`runtest.c` which demonstrates how to call :code:`qoco_custom` to solve the problem instance specified when generating the solver.
