.. _settings:

Settings
--------

The settings are defined in the :code:`include/structs.h` file.

.. tabularcolumns:: |p{4.5cm}|p{8.5cm}|p{1.5cm}|p{6.5cm}|L|

+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| Name                           | Type                  | Description                                                | Allowed values      | Default value |
+================================+=======================+============================================================+=====================+===============+
| :code:`max_iters`              |  :code:`QOCOInt`      | Maximum number of iterations                               | :math:`(0, \infty)` | 200           |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`ruiz_iters`             |  :code:`QOCOInt`      | Number of Ruiz equilibration iterations performed          | :math:`(0, \infty)` | 0             |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`ruiz_scaling_min`       |  :code:`QOCOFloat`    | Minimum cumulative Ruiz scaling                            | :math:`(0, \infty)` | 1e-4          |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`ruiz_scaling_max`       |  :code:`QOCOFloat`    | Maximum cumulative Ruiz scaling                            | :math:`(0, \infty)` | 1e4           |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`max_ir_iters`           |  :code:`QOCOInt`      | Maximum number of iterative refinement iterations          | :math:`(0, \infty)` | 5             |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`ir_tol`                 |  :code:`QOCOFloat`    | Iterative refinement stopping tolerance: stop when         | :math:`(0, \infty)` | 1e-6          |
|                                |                       | :math:`\lVert Kx - b\rVert <` :code:`ir_tol`               |                     |               |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`kkt_static_reg_P`       |  :code:`QOCOFloat`    | Static regularization for the (1,1) P block of the KKT     | :math:`(0, \infty)` | 1e-13         |
|                                |                       | system. Added to the diagonal of P before factorization    |                     |               |
|                                |                       | to ensure the block remains positive definite.             |                     |               |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`kkt_static_reg_A`       |  :code:`QOCOFloat`    | Static regularization for the (2,2) A block of the KKT     | :math:`(0, \infty)` | 1e-8          |
|                                |                       | system. Subtracted from the diagonal of the equality       |                     |               |
|                                |                       | constraint block to give it a definite sign.               |                     |               |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`kkt_static_reg_G`       |  :code:`QOCOFloat`    | Static regularization for the (3,3) G block of the KKT     | :math:`(0, \infty)` | 1e-13         |
|                                |                       | system. Subtracted from the diagonal of the NT scaling     |                     |               |
|                                |                       | block to give it a definite sign.                          |                     |               |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`kkt_dynamic_reg`        |  :code:`QOCOFloat`    | Additional regularization applied dynamically during KKT   | :math:`(0, \infty)` | 1e-11         |
|                                |                       | factorization to any diagonal entry whose absolute value   |                     |               |
|                                |                       | is below this threshold. Improves robustness on            |                     |               |
|                                |                       | near-degenerate problems.                                  |                     |               |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`abstol`                 |  :code:`QOCOFloat`    | absolute tolerance                                         | :math:`(0, \infty)` | 1e-7          |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`reltol`                 |  :code:`QOCOFloat`    | relative tolerance                                         | :math:`(0, \infty)` | 1e-7          |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`abstol_inacc`           |  :code:`QOCOFloat`    | low accuracy absolute tolerance                            | :math:`(0, \infty)` | 1e-5          |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`reltol_inacc`           | :code:`QOCOFloat`     | low accuracy relative tolerance                            | :math:`(0, \infty)` | 1e-5          |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`verbose`                | :code:`unsigned_char` | should the solver print progress                           | 0 or 1              |  0            |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`dump_problem`           | :code:`str` or        | (Python only) Dump the problem data to a binary            | any valid filepath  | None          |
| (Python only)                  | :code:`bool`          | :code:`.bin` file. If a string, used as the filename.      | or :code:`True`     |               |
|                                |                       | If :code:`True`, defaults to :code:`"problem.bin"`.        |                     |               |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
