"""
asa.py — A minimal implementation of the Active Set Algorithm (ASA) from

    Hager, W. W. & Zhang, H. (2006).
    "A new active set algorithm for box constrained optimization."
    SIAM J. Optim., 17(2), 526-557.

Problem:   minimize  f(x)   subject to   l <= x <= u

Two-phase design:
  Phase 1 (NGPA): nonmonotone projected-gradient steps to identify the
                  active face of the box.
  Phase 2 (UA):   unconstrained optimization on the free variables only,
                  with the active variables fixed at their bound.
  A small set of rules (Section 3, Figure 3.1) decide when to branch
  between the two phases.

Simplifications vs. the paper (kept faithful to the *structure*, not every
implementation detail):
  - NGPA initial step uses the plain Barzilai-Borwein formula instead of
    the cyclic-BB scheme of the appendix.
  - Reference value f_r is the simple GLL max-of-last-M (eq. 2.3) instead
    of the elaborate R0-R4 update.
  - UA is unconstrained limited-memory BFGS on the free subspace, with a
    projected backtracking-Armijo line search that adds any free variable
    crossing a bound to the active set and exits (the paper uses
    CG_DESCENT; any method meeting U1-U3 from Section 3 works).  No nested
    bound-constrained solver -- the active-set management is done by the
    outer driver, which is the whole point of the algorithm.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import os
import numpy as np


# ---------------------------------------------------------------------------
# Projection onto the box  l <= x <= u
# ---------------------------------------------------------------------------
def project(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """P(x) = arg min_{y in B} ||y - x||  (eq. 2.2 specialised to a box)."""
    return np.minimum(np.maximum(x, lo), hi)


def d_alpha(x, g, alpha, lo, hi):
    """d_alpha(x) = P(x - alpha*g) - x  (Section 2, just below eq. 2.5)."""
    return project(x - alpha * g, lo, hi) - x


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ASAResult:
    x: np.ndarray
    fun: float
    grad: np.ndarray
    nit: int
    nfev: int
    ngev: int
    converged: bool
    kkt: float                              # ||P(x - g) - x||_inf
    phase_log: list = field(default_factory=list)   # 'N' or 'U' per outer step
    message: str = ""


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
class ASA:
    """
    Hager-Zhang Active Set Algorithm for   min f(x)  s.t.  l <= x <= u.

    Parameters
    ----------
    fun, jac : callables  f(x) -> float,  jac(x) -> grad array
    lo, hi   : array_like bounds (use +/- np.inf for unbounded sides)
    tol      : KKT tolerance,    stops when ||P(x - g) - x||_inf <= tol
    max_iter : outer iteration cap
    mu       : Section 3 — ||g_I|| < mu * ||d^1|| means "face solved enough"
    rho      : decay factor applied to mu when the test triggers
    n1, n2   : counters from Figure 3.1 controlling phase switches
    M        : memory for the nonmonotone reference f_max (eq. 2.3)
    delta    : Armijo descent constant in (0, 1)
    eta      : backtracking factor in (0, 1)
    alpha_min, alpha_max : safeguard interval for the BB step
    alpha_U_iter, alpha_U_grad : per-call UA work budget
    verbose  : print phase transitions
    """

    # Identification-set exponents (Section 3): alpha in (0,1), beta in (1,2)
    ID_ALPHA = 0.5
    ID_BETA = 1.5

    def __init__(self, fun, jac, lo, hi,
                 tol=1e-6, max_iter=500,
                 mu=0.1, rho=0.5, n1=2, n2=1,
                 M=8, delta=1e-4, eta=0.5,
                 alpha_min=1e-20, alpha_max=1e20,
                 ua_max_iter=50, ua_max_grad=50,
                 verbose=False):
        self.fun, self.jac = fun, jac
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.tol, self.max_iter = tol, max_iter
        self.mu0, self.rho, self.n1, self.n2 = mu, rho, n1, n2
        self.M, self.delta, self.eta = M, delta, eta
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
        self.ua_max_iter, self.ua_max_grad = ua_max_iter, ua_max_grad
        self.verbose = verbose

        # evaluation counters
        self._nfev = 0
        self._ngev = 0
        self._proj_grad0_l2 = None

    # ---- tracing ------------------------------------------------------------
    def _trace(self, msg=""):
        if self.verbose:
            print(msg)

    def _fmt_vec(self, x):
        return np.array2string(np.asarray(x), precision=8, suppress_small=False)

    def _projected_grad(self, x, g):
        return d_alpha(x, g, 1.0, self.lo, self.hi)

    def _print_status(self, k, phase, x, f, g, mu):
        projected = self._projected_grad(x, g)
        proj_inf = np.max(np.abs(projected))
        proj_l2 = float(np.linalg.norm(projected))
        rel = np.nan if not self._proj_grad0_l2 else proj_l2 / self._proj_grad0_l2

        self._trace(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> k = {k}")
        self._trace(f"phase: {phase}")
        self._trace(f"f: {f:1.8e}")
        self._trace(f"x: {self._fmt_vec(x)}")
        self._trace(f"grad_f: {self._fmt_vec(g)}")
        self._trace(f"grad_f_proj: {self._fmt_vec(projected)}")
        self._trace(f"||grad_f_proj||_inf = {proj_inf:1.4e}")
        self._trace(f"||grad_f_proj||_2 / ||grad_f_proj_0||_2 = {rel:1.4e}")
        self._trace(
            f"|A| = {int(self._active_mask(x).sum())} mu = {mu:1.4e} "
            f"nfev = {self._nfev} ngev = {self._ngev}"
        )

    # ---- evaluation wrappers ------------------------------------------------
    def _f(self, x):
        self._nfev += 1
        return float(self.fun(x))

    def _g(self, x):
        self._ngev += 1
        return np.asarray(self.jac(x), dtype=float)

    # ---- active-set bookkeeping ---------------------------------------------
    def _active_mask(self, x):
        """Boolean mask of indices touching a bound (Section 3, A(x))."""
        return (x <= self.lo) | (x >= self.hi)

    def _active_signature(self, x):
        """Stable hashable summary of WHICH bound each component is at;
        used to detect 'A(x_k) = A(x_{k-1}) = ...' from Figure 3.1."""
        at_lo = x <= self.lo
        at_hi = x >= self.hi
        sig = np.zeros(x.size, dtype=np.int8)
        sig[at_lo] = -1
        sig[at_hi] = 1
        return sig.tobytes()

    def _grad_I(self, x, g):
        """g_I(x): gradient on the inactive set, zero elsewhere
        (Section 3, just above the U(x) definition)."""
        out = g.copy()
        out[self._active_mask(x)] = 0.0
        return out

    def _undecided_empty(self, x, g, d1_norm):
        """U(x) from Section 3.  We test whether U is empty.

        i in U iff   |g_i(x)| >= ||d^1||^alpha  AND  dist(x_i, bound) >= ||d^1||^beta
        """
        if d1_norm == 0.0:
            return True
        thresh_g = d1_norm ** self.ID_ALPHA
        thresh_x = d1_norm ** self.ID_BETA
        # distance to the nearer bound
        dist = np.minimum(x - self.lo, self.hi - x)
        # treat +/-inf bounds as "very far"
        dist = np.where(np.isfinite(dist), dist, np.inf)
        in_U = (np.abs(g) >= thresh_g) & (dist >= thresh_x)
        return not np.any(in_U)

    # ---- KKT residual -------------------------------------------------------
    def _kkt(self, x, g):
        """||P(x - g) - x||_inf — natural stationarity measure for box
        problems; used both as stopping criterion and as ||d^1(x)||."""
        return np.max(np.abs(d_alpha(x, g, 1.0, self.lo, self.hi)))

    # ---- BB initial step ----------------------------------------------------
    def _bb_step(self, s, y, fallback):
        """alpha_BB = (s.s) / (s.y)  (eq. A.1), safeguarded."""
        sy = float(s @ y)
        if sy <= 0:
            self._trace(f"BB fallback: s.y = {sy:1.4e}")
            return fallback
        a = float(s @ s) / sy
        self._trace(f"a = {a:1.4e}")
        return float(np.clip(a, self.alpha_min, self.alpha_max))

    # ========================================================================
    # NGPA — one outer step  (Section 2)
    # ========================================================================
    def _ngpa_step(self, x, fx, g, alpha_bb, f_history):
        """One iteration of the nonmonotone gradient projection algorithm.

        Returns: (x_new, f_new, g_new, accepted_bool)
        """
        # Step 1: search direction (Figure 2.1)
        d = d_alpha(x, g, alpha_bb, self.lo, self.hi)
        if np.all(d == 0.0):
            self._trace("Projected grad is small")
            return x, fx, g, False

        # Reference value: simple GLL max-of-last-M  (eq. 2.3)
        f_ref = max(f_history) if f_history else fx

        # Step 3/4: nonmonotone Armijo on the segment x -> x + d
        gTd = float(g @ d)               # < 0 by Prop. 2.1 (P6)
        alpha = 1.0
        f_trial = self._f(x + d)
        i = 0
        self._trace("Running backtracking armijo")
        self._trace(
            f"NGPA trial: alpha_bb = {alpha_bb:1.4e} "
            f"f_ref = {f_ref:1.4e} grad_dot_d = {gTd:1.4e}"
        )
        # backtrack until  f(x + alpha*d) <= f_ref + delta * alpha * g.d
        while f_trial > f_ref + self.delta * alpha * gTd:
            alpha *= self.eta
            if alpha < 1e-30:            # safeguard against zero step
                break
            f_trial = self._f(x + alpha * d)
            i += 1
        self._trace(f"Found step i = {i} alpha = {alpha:1.4e} f_trial = {f_trial:1.4e}")

        x_new = x + alpha * d
        g_new = self._g(x_new)
        return x_new, f_trial, g_new, True

    # ========================================================================
    # UA — unconstrained L-BFGS on the current free face  (Section 3)
    # ========================================================================
    #
    # The UA is an UNCONSTRAINED method on the free variables.  Its only
    # interaction with the bounds is the projected line search U4:
    #
    #        x_{k+1} = P(x_k - alpha_k g_I(x_k)),
    #
    # i.e. step along the negative inactive gradient, and if a free variable
    # would cross a bound, snap it to the bound and ADD that index to the
    # active set.  As soon as that happens we exit so the driver can apply
    # the step-2b rules of Figure 3.1.  This is option (2) from the discussion
    # above: no nested bound-constrained solver.
    #
    # Inside one UA call we run a small L-BFGS loop (two-loop recursion,
    # memory `m`) on the free subspace.  The active variables are physically
    # frozen at their bound, so U2 ("the active set can only grow") holds by
    # construction.

    def _ua_run(self, x, fx, g, m=10):
        """One UA episode: limited-memory BFGS on the free subspace with
        a projected backtracking-Armijo line search.

        Returns (x_new, f_new, g_new, grew_active_set_bool).
        """
        free = ~self._active_mask(x)
        if not np.any(free):
            self._trace("UA skipped: no free variables")
            return x, fx, g, False

        free_idx = np.where(free)[0]
        n_free = free_idx.size
        self._trace(
            f"UA restricted problem: n_free = {n_free} "
            f"free_idx = {self._fmt_vec(free_idx)}"
        )

        # subspace coordinates
        z = x[free_idx].copy()
        lo_z = self.lo[free_idx]
        hi_z = self.hi[free_idx]

        # helpers that lift a subspace point into full-space
        def lift(z_):
            xx = x.copy()                 # active components stay frozen
            xx[free_idx] = z_
            return xx

        fz = fx
        gz = g[free_idx].copy()           # = g_I restricted to the free indices

        # L-BFGS history (two-loop recursion buffers)
        S, Y, RHO = [], [], []

        for inner in range(self.ua_max_iter):
            self._trace(
                f"UA inner {inner}: f = {fz:1.8e} "
                f"||grad_free||_inf = {np.max(np.abs(gz)):1.4e}"
            )
            # ---- search direction via two-loop recursion ------------------
            q = gz.copy()
            alphas = []
            for s_, y_, rho_ in zip(reversed(S), reversed(Y), reversed(RHO)):
                a = rho_ * float(s_ @ q)
                alphas.append(a)
                q -= a * y_
            if S:                          # H0 = (s.y)/(y.y) * I, standard
                gamma = float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])
            else:
                gamma = 1.0
            r = gamma * q
            for s_, y_, rho_, a in zip(S, Y, RHO, reversed(alphas)):
                b = rho_ * float(y_ @ r)
                r += (a - b) * s_
            d = -r                          # quasi-Newton step

            # safety: if curvature info gave a non-descent direction, reset
            if float(gz @ d) >= 0.0:
                self._trace("UA reset to steepest descent direction")
                d = -gz                    # plain steepest descent

            # ---- projected backtracking-Armijo line search ----------------
            # alpha_max_box: the largest alpha for which z + alpha*d stays
            # inside the subspace box.  Going beyond it activates a new bound.
            with np.errstate(divide="ignore", invalid="ignore"):
                pos = d > 0
                neg = d < 0
                steps_hi = np.where(pos, (hi_z - z) / d, np.inf)
                steps_lo = np.where(neg, (lo_z - z) / d, np.inf)
            alpha_box = float(min(steps_hi.min(), steps_lo.min(), np.inf))

            alpha = 1.0
            gTd = float(gz @ d)
            self._trace(f"UA line search: alpha_box = {alpha_box:1.4e} grad_dot_d = {gTd:1.4e}")
            # backtrack
            for ls_iter in range(60):
                z_trial = z + alpha * d
                # project (only matters when alpha > alpha_box)
                z_trial = np.minimum(np.maximum(z_trial, lo_z), hi_z)
                f_trial = self._f(lift(z_trial))
                # Armijo: monotone here (paper's U1 requires monotone UA)
                if f_trial <= fz + self.delta * alpha * gTd:
                    self._trace(
                        f"UA found step i = {ls_iter} alpha = {alpha:1.4e} "
                        f"f_trial = {f_trial:1.4e}"
                    )
                    break
                alpha *= self.eta
            else:
                # line search failed; bail out without updating
                self._trace("UA line search failed")
                break

            # Did this step actually hit a bound?  (alpha >= alpha_box means
            # the unprojected trial would have crossed at least one bound;
            # after projection one or more free vars are exactly at the bound.)
            hit_bound = alpha >= alpha_box - 1e-16
            if hit_bound:
                self._trace("UA step hit a bound")
            z = z_trial

            # update full-space iterate and recompute gradient
            x_full = lift(z)
            fz = f_trial
            g_full = self._g(x_full)
            gz_new = g_full[free_idx]

            # L-BFGS history update (curvature gate sk.yk > 0)
            s_k = alpha * d
            y_k = gz_new - gz
            sy = float(s_k @ y_k)
            if sy > 1e-12 * float(s_k @ s_k):
                S.append(s_k); Y.append(y_k); RHO.append(1.0 / sy)
                if len(S) > m:
                    S.pop(0); Y.pop(0); RHO.pop(0)
            else:
                self._trace(f"UA skipped L-BFGS update: s.y = {sy:1.4e}")

            gz = gz_new

            # If a free variable just became active, return so the outer
            # driver can re-check step 2b of the ASA.  This is the clean
            # analogue of the appendix instruction "any components which
            # reach the boundary are added to the current active set."
            if hit_bound:
                return x_full, fz, g_full, True

            # Inner stopping: ||g on free vars|| small enough.  We let the
            # driver decide globally; here we just stop wasting work.
            if np.max(np.abs(gz)) <= self.tol * 0.1:
                self._trace("UA inner gradient tolerance met")
                return x_full, fz, g_full, False

        # exhausted inner iteration budget
        x_full = lift(z)
        return x_full, fz, self._g(x_full), False

    # ========================================================================
    # Driver — orchestrates the NGPA <-> UA branching of Figure 3.1
    # ========================================================================
    def solve(self, x0):
        x = project(np.asarray(x0, dtype=float), self.lo, self.hi)
        f = self._f(x)
        g = self._g(x)
        self._proj_grad0_l2 = float(np.linalg.norm(self._projected_grad(x, g)))

        mu = self.mu0
        phase = "N"                          # start in NGPA
        active_sigs = [self._active_signature(x)]
        f_hist = [f]                         # rolling window for f_max
        log = []

        alpha_bb = 1.0                       # initial BB step
        x_prev, g_prev = None, None          # for BB update after step

        for k in range(self.max_iter):
            kkt = self._kkt(x, g)
            log.append(phase)
            if self.verbose:
                self._print_status(k, phase, x, f, g, mu)
                self._trace("Checking convergence")

            # Stopping test (Section 6: ||P(x - g) - x||_inf <= tol)
            if kkt <= self.tol:
                self._trace("Atol reached")
                return ASAResult(x=x, fun=f, grad=g, nit=k,
                                 nfev=self._nfev, ngev=self._ngev,
                                 converged=True, kkt=kkt,
                                 phase_log=log, message="KKT tolerance met.")

            # ----------------- Phase 1: NGPA --------------------------------
            if phase == "N":
                self._trace("Entering NGPA Phase")
                x_prev_inner, g_prev_inner = x.copy(), g.copy()
                x, f, g, ok = self._ngpa_step(x, f, g, alpha_bb, f_hist)
                if not ok:
                    # zero direction at a non-stationary point shouldn't happen
                    break

                # Update rolling reference window (memory M, eq. 2.3)
                f_hist.append(f)
                if len(f_hist) > self.M:
                    f_hist.pop(0)

                # Refresh BB step from this iteration's s, y
                s = x - x_prev_inner
                y = g - g_prev_inner
                alpha_bb = self._bb_step(s, y, fallback=alpha_bb)
                self._trace(f"alpha_bb = {alpha_bb:1.4e}")

                # Track recent active-set signatures for the n1 test
                active_sigs.append(self._active_signature(x))
                if len(active_sigs) > self.n1 + 2:
                    active_sigs.pop(0)

                # ---- Branching test 1a: undecided set empty? --------------
                d1n = self._kkt(x, g)
                gI_norm = np.linalg.norm(self._grad_I(x, g))
                undecided_empty = self._undecided_empty(x, g, d1n)
                self._trace(
                    f"NGPA branch tests: undecided_empty = {undecided_empty} "
                    f"||grad_inactive|| = {gI_norm:1.4e} "
                    f"mu * ||grad_proj|| = {mu * d1n:1.4e}"
                )
                if undecided_empty:
                    if gI_norm < mu * d1n:
                        self._trace("||grad_inactive|| < mu ||grad_proj||")
                        mu *= self.rho     # shrink mu, stay in NGPA
                        self._trace(f"mu = {mu:1.4e}")
                    else:
                        self._trace("Switching to UA")
                        phase = "U"        # face looks identified -> UA
                        continue

                # ---- Branching test 1b: A(x) stable for n1 iterations -----
                if len(active_sigs) >= self.n1 + 1:
                    recent = active_sigs[-(self.n1 + 1):]
                    if all(s_ == recent[0] for s_ in recent):
                        if gI_norm >= mu * d1n:
                            self._trace("Active set stable")
                            self._trace("Switching to UA")
                            phase = "U"
                            continue

            # ----------------- Phase 2: UA ----------------------------------
            else:  # phase == "U"
                self._trace("Entering UA Phase")
                size_A_before = int(self._active_mask(x).sum())
                x_new, f_new, g_new, grew = self._ua_run(x, f, g)

                # Subproblem-solved test (step 2a, Figure 3.1)
                d1n_new = self._kkt(x_new, g_new)
                gI_norm_new = np.linalg.norm(self._grad_I(x_new, g_new))
                subproblem_solved = gI_norm_new < mu * d1n_new
                self._trace(
                    f"UA branch tests: grew = {grew} "
                    f"||grad_inactive|| = {gI_norm_new:1.4e} "
                    f"mu * ||grad_proj|| = {mu * d1n_new:1.4e}"
                )

                x, f, g = x_new, f_new, g_new
                f_hist.append(f)
                if len(f_hist) > self.M:
                    f_hist.pop(0)
                active_sigs.append(self._active_signature(x))

                if subproblem_solved:
                    self._trace("||grad_inactive|| < mu ||grad_proj||")
                    self._trace("Switching to NGPA")
                    phase = "N"            # restart NGPA (step 2a)
                    continue

                # Step 2b: new constraints became active during UA
                size_A_after = int(self._active_mask(x).sum())
                if size_A_after > size_A_before:
                    if (not self._undecided_empty(x, g, d1n_new)
                            and size_A_after <= size_A_before + self.n2):
                        self._trace("No. of active bounds has increased")
                        self._trace("Switching to NGPA")
                        phase = "N"        # restart NGPA
                        continue
                    # else: restart UA at current x (just stay in U)
                self._trace("Sticking to UA")

        # exhausted max_iter
        kkt = self._kkt(x, g)
        return ASAResult(x=x, fun=f, grad=g, nit=self.max_iter,
                         nfev=self._nfev, ngev=self._ngev,
                         converged=kkt <= self.tol, kkt=kkt,
                         phase_log=log,
                         message="Iteration limit reached.")


# ---------------------------------------------------------------------------
# Convenience wrapper mirroring scipy.optimize.minimize's signature
# ---------------------------------------------------------------------------
def minimize_box(fun, x0, jac, bounds, **kwargs):
    """Minimize fun(x) subject to box bounds using the Hager-Zhang ASA.

    Parameters
    ----------
    fun, jac : callables
    x0       : starting point (will be projected into the box)
    bounds   : sequence of (lo_i, hi_i) pairs, use None / +-inf for unbounded
    **kwargs : forwarded to ASA(...)
    """
    lo = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds])
    hi = np.array([( np.inf if b[1] is None else b[1]) for b in bounds])
    solver = ASA(fun=fun, jac=jac, lo=lo, hi=hi, **kwargs)
    return solver.solve(np.asarray(x0, dtype=float))


# ---------------------------------------------------------------------------
# Self-test / demo:  three classic box-constrained problems
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    verbose = os.environ.get("ASA_VERBOSE", "").lower() in {"1", "true", "yes", "on"}

    def banner(title):
        print("\n" + "=" * 64)
        print(" " + title)
        print("=" * 64)

    def box_kkt(jac, x, bounds):
        lo = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds])
        hi = np.array([( np.inf if b[1] is None else b[1]) for b in bounds])
        g = jac(x)
        return np.max(np.abs(project(x - g, lo, hi) - x))

    # ---- 1. Simple quadratic with active bounds -----------------------------
    banner("Problem 1:  min sum (x_i - i)^2,  x in [0, 3]^5")
    # Unconstrained minimum is (0,1,2,3,4).  With upper bound 3, x_5 hits it.
    n = 5
    target = np.arange(n, dtype=float)

    def f1(x): return float(np.sum((x - target) ** 2))
    def g1(x): return 2.0 * (x - target)

    res = minimize_box(f1, x0=np.full(n, 1.5),
                       jac=g1, bounds=[(0.0, 3.0)] * n, verbose=verbose)
    print(f"  converged = {res.converged}")
    print(f"  x*        = {res.x}      (expected [0 1 2 3 3])")
    print(f"  f(x*)     = {res.fun:.6e} (expected {1.0:.6e})")
    print(f"  KKT       = {res.kkt:.2e}")
    print(f"  iters     = {res.nit},  nfev={res.nfev}, ngev={res.ngev}")
    print(f"  phases    = {''.join(res.phase_log)}")

    # ---- 2. Rosenbrock with box bounds --------------------------------------
    banner("Problem 2:  10-D Rosenbrock,  x in [-2, 2]^10  (unconstrained opt)")
    n = 10

    def f2(x):
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 +
                            (1.0 - x[:-1]) ** 2))

    def g2(x):
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    res = minimize_box(f2, x0=np.full(n, -1.2), jac=g2,
                       bounds=[(-2.0, 2.0)] * n, max_iter=2000,
                       verbose=verbose)
    print(f"  converged = {res.converged}")
    print(f"  x*        = {res.x}")
    print(f"  f(x*)     = {res.fun:.6e}   (expected ~0)")
    print(f"  KKT       = {res.kkt:.2e}")
    print(f"  iters     = {res.nit},  nfev={res.nfev}, ngev={res.ngev}")

    # ---- 2b. Rosenbrock with constrained minimizer outside the box ----------
    banner("Problem 2b:  10-D Rosenbrock,  x in [-2, 0.99]^10  (outside box)")
    bounds_2b = [(-2.0, 0.99)] * n
    expected_x = np.full(n, 0.99)
    expected_f = f2(expected_x)

    res = minimize_box(f2, x0=np.full(n, -1.2), jac=g2,
                       bounds=bounds_2b, max_iter=2000,
                       verbose=verbose)
    print(f"  converged = {res.converged}")
    print(f"  x*        = {res.x}      (Rust test expects all 0.99)")
    print(f"  max |x*-expected| = {np.max(np.abs(res.x - expected_x)):.6e}")
    print(f"  f(x*)     = {res.fun:.6e}")
    print(f"  f(0.99)   = {expected_f:.6e}   (the selected Rust test expects 0)")
    print(f"  KKT       = {res.kkt:.2e}      (for bounds [-2, 0.99])")
    print(f"  KKT[-2,2] = {box_kkt(g2, res.x, [(-2.0, 2.0)] * n):.2e}")
    print(f"  iters     = {res.nit},  nfev={res.nfev}, ngev={res.ngev}")
    print(f"  phases    = {''.join(res.phase_log)}")

    # ---- 3. Quadratic where bounds matter, degenerate case ------------------
    banner("Problem 3:  min 0.5 x^T A x - b^T x,  x >= 0  (NNLS-ish)")
    rng = np.random.default_rng(0)
    n = 20
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + 0.1 * np.eye(n)        # SPD
    # build b so that the unconstrained solution has many negative entries
    x_un = rng.standard_normal(n)
    x_un[::2] = -np.abs(x_un[::2])       # force some negatives
    b = A @ x_un

    def f3(x): return 0.5 * float(x @ A @ x) - float(b @ x)
    def g3(x): return A @ x - b

    res = minimize_box(f3, x0=np.ones(n), jac=g3,
                       bounds=[(0.0, np.inf)] * n, max_iter=500,
                       verbose=verbose)
    print(f"  converged = {res.converged}")
    print(f"  f(x*)     = {res.fun:.6e}")
    print(f"  KKT       = {res.kkt:.2e}")
    print(f"  iters     = {res.nit},  nfev={res.nfev}, ngev={res.ngev}")
    print(f"  active    = {int((res.x <= 0).sum())} / {n} components at bound")
    print(f"  phases    = {''.join(res.phase_log)}")

    # Cross-check against scipy's L-BFGS-B
    from scipy.optimize import minimize
    ref = minimize(f3, np.ones(n), jac=g3, method="L-BFGS-B",
                   bounds=[(0.0, None)] * n, options=dict(ftol=1e-14, gtol=1e-9))
    print(f"  L-BFGS-B reference f = {ref.fun:.6e}   diff = {res.fun - ref.fun:+.2e}")
