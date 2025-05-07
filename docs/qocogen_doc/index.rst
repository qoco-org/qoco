.. _qocogen_doc:

QOCOGEN
=========

QOCOGEN is a custom code generator which takes in an SOCP problem family and generates a customized C solver (called qoco_custom) for this problem family which implements the same algorithm as QOCO. This customized solver is library-free, only uses static memory allocation, and can be a few times faster than QOCO.

All problems in the same problem family have identical sparsity patterns for matrices :code:`P`, :code:`A`, and :code:`G`, and have identical values for :code:`l`, :code:`m`, :code:`p`, :code:`nsoc`, and :code:`q`.

The easiest way to use QOCOGEN is through CVXPYgen (see :ref:`Lossless Convexification <lcvx_example>`).


QOCOGEN generates custom solvers to solve SOCPs with the same standard form that QOCO solves.

.. toctree::
   :maxdepth: 1

   installation.rst
   generate
   build
   api
   calling
   cvxpygen