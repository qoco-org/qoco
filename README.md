# Quadratic Objective Conic Optimizer (QOCO)

QOCO implements a primal-dual interior point method to solve second-order cone programs with quadratic objectives of the following form

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

$$\mathcal{Q}^{q_i}_i = \\{(t,x)  \in \mathbb{R} \times \mathbb{R}^{q_i - 1} \\; | \\; ||x||_2 \leq t \\}$$

## Bug reports

File any issues or bug reports using the [issue tracker](https://github.com/qoco-org/qoco/issues).
