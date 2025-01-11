.. QOCO documentation master file, created by
   sphinx-quickstart on Thu Jun 13 20:02:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QOCO documentation
================================

This website documents the Quadratic Objective Conic Optimizer (QOCO) and its code generator :ref:`QOCOGEN <qocogen>`.

Standard Form
--------------

QOCO (pronounced co-co) is an software package to solve second-order cone programs with quadratic objectives of the following form

.. math::
  \begin{split}
      \underset{x}{\text{minimize}} 
      \quad & \frac{1}{2}x^\top P x + c^\top x \\
      \text{subject to} 
      \quad & Gx \preceq_\mathcal{C} h \\
      \quad & Ax = b
  \end{split}


with optimization variable :math:`x \in \mathbb{R}^n` and problem data :math:`P = P^\top \succeq 0`, :math:`c \in \mathbb{R}^n`, :math:`G \in \mathbb{R}^{m \times n}`, :math:`h \in \mathbb{R}^m`, :math:`A \in \mathbb{R}^{p \times n}`, :math:`b \in \mathbb{R}^p`, and :math:`\preceq_\mathcal{C}` 
is an inequality with respect to cone :math:`\mathcal{C}`, i.e. :math:`h - Gx \in \mathcal{C}`. Cone :math:`\mathcal{C}` is the Cartesian product of the non-negative orthant and second-order cones, which can be expressed as

.. math::
    \mathcal{C} =  \mathbb{R}^l_+ \times \mathcal{Q}^{q_1}_1 \times \ldots \times \mathcal{Q}^{q_N}_N

where :math:`l` is the dimension of the non-negative orthant, and :math:`\mathcal{Q}^{q_i}_i` is the :math:`i^{th}` second-order cone with dimension :math:`q_i` defined by

.. math::
    \mathcal{Q}^{q_i}_i = \{(t,x)  \in \mathbb{R} \times \mathbb{R}^{q_i - 1} \; : \; \|x\|_2 \leq t \}

Features
--------------

License
--------------
Our code is open-source and distrubuted under the `BSD 3-Clause license <https://opensource.org/license/bsd-3-Clause>`_, and can be found on `GitHub <https://github.com/qoco-org/qoco>`_.

Citing
--------------
If you are using QOCO please

* :ref:`Cite the related paper <citing>`
* Star the repository on `GitHub <https://github.com/qoco-org/qoco>`_

.. Benchmarks
.. --------------
.. Benchmarks against other solvers can be found `here https://github.com/qoco-org/qoco_benchmarks>`_
 
.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Solver Documentation:

   install/index
   api/index
   parsers/index
   examples/index
   codegen/index
   contributing/index
   citing/index