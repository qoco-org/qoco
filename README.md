# Quadratic Objective Conic Optimizer (QOCO)
<p align="center">
  <img src="https://github.com/user-attachments/assets/7bd44fa7-d198-4739-bb79-a5c15e04a8de" alt="drawing" width="500"/>
</p>

<p align="center">
  <a href=https://github.com/qoco-org/qoco/actions/workflows/unit_tests.yml/badge.svg"><img src="https://github.com/qoco-org/qoco/actions/workflows/unit_tests.yml/badge.svg"/></a>
  <a href="https://img.shields.io/pypi/dm/qoco.svg?label=Pypi%20downloads"><img src="https://img.shields.io/pypi/dm/qoco.svg?label=Pypi%20downloads" alt="PyPI Downloads" /></a>
  <a href="https://qoco-org.github.io/qoco/"><img src="https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat" alt="Documentation" /></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-green.svg" alt="License" /></a>
</p>


QOCO is a C implementation of a primal-dual interior point method to solve second-order cone programs with quadratic objectives of the following form

$$
  \begin{split}
      \underset{x}{\text{minimize}}
      \quad & \frac{1}{2}x^\top P x + c^\top x \\
      \text{subject to}
      \quad & Gx \preceq_\mathcal{C} h \\
      \quad & Ax = b
  \end{split}
$$

with optimization variable $x \in \mathbb{R}^n$ and problem data $P = P^\top \succeq 0$, $c \in \mathbb{R}^n$, $G \in \mathbb{R}^{m \times n}$, $h \in \mathbb{R}^m$, $A \in \mathbb{R}^{p \times n}$, $b \in \mathbb{R}^p$, and $\preceq_\mathcal{C}$
is an inequality with respect to cone $\mathcal{C}$, i.e. $h - Gx \in \mathcal{C}$. Cone $\mathcal{C}$ is the Cartesian product of the non-negative orthant and second-order cones, which can be expressed as

$$\mathcal{C} =  \mathbb{R}^l_+ \times \mathcal{Q}^{q_1}_1 \times \ldots \times \mathcal{Q}^{q_N}_N$$

where $l$ is the dimension of the non-negative orthant, and $\mathcal{Q}^{q_i}_i$ is the $i^{th}$ second-order cone with dimension $q_i$ defined by

$$\mathcal{Q}^{q_i}_i = \\{(t,x)  \in \mathbb{R} \times \mathbb{R}^{q_i - 1} \\; | \\; norm(x) \leq t \\}$$

## Bug reports

File any issues or bug reports using the [issue tracker](https://github.com/qoco-org/qoco/issues).

## Citing
```
@misc{chari2025qoco,
  title         = {QOCO: A Quadratic Objective Conic Optimizer with Custom Solver Generation},
  author        = {Chari, Govind M and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  year          = {2025},
  eprint        = {2503.12658},
  archiveprefix = {arXiv},
  primaryclass  = {math.OC},
  url           = {https://arxiv.org/abs/2503.12658}
}
```

## License
QOCO is licensed under the BSD-3-Clause license.
