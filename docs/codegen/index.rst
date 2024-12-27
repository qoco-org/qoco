.. _qocogen:

QOCOGEN
========================

QOCOGEN is a custom code generator which takes in an SOCP problem family and generates a customized C solver (called qoco_custom) for this problem family which implements the same algorithm as QOCO. This customized solver is library-free, only uses static memory allocation, and can be a few times faster than QOCO.

All problems in the same problem family have identical sparsity patterns for matrices :code:`P`, :code:`A`, and :code:`G`, and have identical values for :code:`l`, :code:`m`, :code:`p`, :code:`nsoc`, and :code:`q`.

Standard Form
--------------
QOCOGEN generates custom solvers to solve SOCPs with the same standard form that QOCO solves.

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

Usage
------

.. toctree::
   :maxdepth: 2

   installation.rst
   generate.rst
   build.rst
   api.rst
   calling.rst