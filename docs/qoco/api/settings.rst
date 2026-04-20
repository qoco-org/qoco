.. _settings:

Settings
--------

The settings are defined in the :code:`include/structs.h` file.

.. tabularcolumns:: |p{4.5cm}|p{8.5cm}|p{1.5cm}|p{6.5cm}|L|

+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| Name                           | Type                  |Description                                                 | Allowed values      | Default value |
+================================+=======================+============================================================+=====================+===============+
| :code:`max_iters`              |  :code:`QOCOInt`      | Maximum number of iterations                               | :math:`(0, \infty)` | 500           |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`ruiz_iters`             |  :code:`QOCOInt`      | Number of Ruiz equilibration iterations performed          | :math:`(0, \infty)` | 0             |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`max_ir_iters`           |  :code:`QOCOInt`      | Maximum number of iterative refinement iterations          | :math:`(0, \infty)` | 5             |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`ir_tol`                 |  :code:`QOCOFloat`    | Iterative refinement stopping tolerance: stop when         | :math:`(0, \infty)` | 1e-7          |
|                                |                       | :math:`\|Kx - b\| <` :code:`ir_tol`                       |                     |               |
+--------------------------------+-----------------------+------------------------------------------------------------+---------------------+---------------+
| :code:`kkt_static_reg`         |  :code:`QOCOFloat`    | Positive constant added to the diagonal of the KKT system  | :math:`(0, \infty)` | 1e-12         |
|                                |                       | before every solve to ensure it remains nonsingular.       |                     |               |
|                                |                       | Increase if the solver reports numerical errors on         |                     |               |
|                                |                       | ill-conditioned problems.                                  |                     |               |
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
