.. QOCO documentation master file, created by
   sphinx-quickstart on Thu Jun 13 20:02:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QOCO documentation
================================

QOCO (Quadratic objective Conic Optimization Solver) is an software package to solve second-order cone programs with quadratic objectives of the following form

.. math::
   \begin{split}
      \underset{x}{\text{minimize}} 
      \quad & \frac{1}{2}x^\top P x + c^\top x \\
      \text{subject to} 
      \quad & Gx \preceq_\mathcal{C} h \\
      \quad & Ax = b
  \end{split}

Our code is open-source and distrubuted under the `BSD 2-Clause license <https://opensource.org/license/bsd-2-clause>`_, and can be found on `GitHub <https://github.com/govindchari/qoco>`_.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Solver Documentation:

   overview/index
   install/index
   api/index
   parsers/index
   examples/index
   codegen/index
   contributing/index
   citing/index