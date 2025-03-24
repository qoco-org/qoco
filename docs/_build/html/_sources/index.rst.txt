.. QOCO documentation master file, created by
   sphinx-quickstart on Thu Jun 13 20:02:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/img/qoco-logo.JPEG
  :width: 400
  :alt: QOCO
  :align: center
  :target: https://github.com/qoco-org

This website documents the Quadratic Objective Conic Optimizer (QOCO) and the code generator :ref:`QOCOGEN <qocogen>`, developed by the `Autonomous Controls Laboratory <https://uwacl.com/>`_ at the University of Washington. The code for both is available on `GitHub <https://github.com/qoco-org>`_.

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
.. glossary::

* **Robust**: Given that QOCO implements a primal-dual interior point method, it is very robust to ill-conditioning in problem data.
* **Fast**: Faster and more robust than many commercial and open-source second-order cone solvers.
* **Easy to use**: Can be called from C, C++, Python and with parsers such as CVXPY making it easy to use.
* **Free and open source**: Distributed under the `BSD 3-Clause license <https://opensource.org/license/bsd-3-Clause>`_
* **Embeddable**: Written in C, so it can be easily run on any embedded system.
* **Library-free**: Does not require any external libraries.
* **Tuning-free**: Does not require any hyperparameter tuning to achieve good performance.
* **Code Generation**: :ref:`QOCOGEN <qocogen>` is a custom solver generator which generates extremely fast, library-free custom solvers for second-order cone programs.

Benchmarks
--------------
Benchmarks against other solvers can be found `here <https://github.com/qoco-org/qoco_benchmarks>`_.

Credits
--------
The main developer of QOCO is `Govind Chari <https://govindchari.com/>`_, who is advised by `Behçet Açikmeşe <https://www.aa.washington.edu/facultyfinder/behcet-acikmese>`_. Both are affiliated with the `Autonomous Controls Laboratory <https://uwacl.com/>`_ at the University of Washington.

QOCO is an adapted implementation of Lieven Vandenberghe's `coneqp <https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf>`_ solver with various numerical enhancements for stable computation of search directions and is built on the following open-source libraries:

* `qdldl <https://github.com/osqp/qdldl>`_: Responsible for solving the KKT system to compute search directions.
* `OSQP <https://osqp.org/>`_: The C, Matlab, and Python interfaces were inspired by OSQP.
* `pybind11 <https://github.com/pybind/pybind11>`_: Used to generate QOCO's python wrapper.

Thank you to `Srinidhi Chari <https://www.linkedin.com/in/srinidhi-chari>`_ for designing the QOCO logo.

Citing
--------------
If you find QOCO useful please star the repository on `GitHub <https://github.com/qoco-org/qoco>`_ and cite the `QOCO paper <https://arxiv.org/abs/2503.12658>`_ as follows

.. code:: latex

  @misc{chari2025qoco,
    title         = {Custom Solver Generation for Quadratic Objective Second-Order Cone Programs},
    author        = {Chari, Govind M and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
    year          = {2025},
    eprint        = {2503.12658},
    archiveprefix = {arXiv},
    primaryclass  = {math.OC},
    url           = {https://arxiv.org/abs/2503.12658}
  }
 
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