Yes — the AL-level convergence test should use the **bound-projected gradient of the Lagrangian**, not the raw gradient. Concretely:

**Stationarity (KKT dual feasibility):**

Let the augmented Lagrangian subproblem minimize (over the box $\ell \le x \le u$)

$$
L_A(x,\lambda,\mu) = f(x) + \sum_i \lambda_i c_i(x) + \tfrac{\mu}{2}\sum_i c_i(x)^2
$$

The inner solver returns an approximate stationary point of *that* box-constrained subproblem. At the AL (outer) level you want stationarity of the *original* problem's Lagrangian w.r.t. the box, so the residual is

$$
\Big\| P_{[\ell,u]}\!\big(x - \nabla_x L(x,\lambda)\big) - x \Big\|
$$

where $\nabla_x L = \nabla f(x) + \sum_i \lambda_i \nabla c_i(x)$ uses the **current (updated) multiplier estimates**, and $P_{[\ell,u]}$ is the projection onto the box. Equivalently, the componentwise projected gradient

$$
[\nabla_x L]_j^{\text{proj}} =
\begin{cases}
\min(\nabla_x L_j, 0) & x_j = \ell_j\\
\nabla_x L_j & \ell_j < x_j < u_j\\
\max(\nabla_x L_j, 0) & x_j = u_j
\end{cases}
$$

So yes: gradient of $f$ plus $\lambda^T \nabla c$, then bound-projected. Note this uses $\lambda$ (the multiplier), **not** $\lambda + \mu c$ — using the penalized quantity is what the *inner* solver drives to zero; at the outer level you substitute the updated $\lambda_{k+1} = \lambda_k + \mu_k c(x_k)$ and check the plain Lagrangian gradient.

**Primal feasibility:**

$$
\|c(x)\|_\infty \le \epsilon_{\text{feas}}
$$

(with equality constraints; for inequalities handled via slacks/projection, the corresponding measure). This is the second, independent test — the projected-gradient test alone is not enough.

**Complementarity** for the bound multipliers is handled implicitly by the projection (it's automatically satisfied at a projected-stationary box point), so you don't test it separately for the bounds — only feasibility + projected dual feasibility remain at the AL level.

This matches the standard scheme in Birgin & Martínez (Algencan) and Conn–Gould–Toint (LANCELOT): outer loop stops when $\|c(x_k)\|$ is below the feasibility tolerance **and** the projected Lagrangian gradient is below the optimality tolerance, with the inner tolerance $\eta_k$ tightened as $\mu_k$ grows / $\|c\|$ shrinks. You have `birgin-2014-constrained-opt.pdf` in the project — §4 there gives the exact stopping conditions Algencan uses if you want to match them precisely.

One practical caveat: keep the *inner* subproblem's own projected-gradient tolerance ($\eta_k$) distinct from the *outer* test above. They use different gradients ($\nabla L_A$ with penalty vs. $\nabla L$ with updated multipliers), and conflating them is a common source of premature or stalled outer convergence.