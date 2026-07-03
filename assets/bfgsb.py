"""
BFGS-B: Full-Hessian BFGS for bound-constrained optimization
=============================================================

A single-file pure-NumPy implementation of the L-BFGS-B framework of
Byrd, Lu, Nocedal & Zhu (1995) with the limited-memory Hessian
replaced by a dense n*n BFGS approximation.  This is suitable when n
is small (say n < 1000) and the cost of an n*n Cholesky per iteration
is dwarfed by the per-iteration function/gradient cost.

The three-phase iteration structure is unchanged from Byrd 1995:

  Phase 1 - GENERALIZED CAUCHY POINT.  Walk the projected steepest-
            descent path P(x_k - t g_k, l, u) against the quadratic
            model
                m_k(x) = f_k + g_k^T (x - x_k)
                       + (1/2)(x - x_k)^T B_k (x - x_k)
            and locate the first piecewise-linear minimizer x^c.
            Section 4 / Algorithm CP of Byrd et al. 1995.

  Phase 2 - SUBSPACE MINIMIZATION.  Approximately minimize m_k on the
            free face exposed by the Cauchy point by solving
                B_{FF} d_F = -r_F
            with a dense Cholesky on the principal submatrix
            B_{FF} := B_k[F, F], then truncating the move to the box.

  Phase 3 - LINE SEARCH.  Strong-Wolfe bracket-and-zoom along
            d = z - x_k, capped at the largest feasibility-preserving
            step length.

The only differences from this module's sibling `lbfgsb.py` are:

  * The compact-form state (S, Y, theta, U) is replaced by a single
    dense matrix self.B of shape (n, n), updated in place by the
    standard rank-2 BFGS formula
            B <- B + y y^T / (y^T s) - (B s)(B s)^T / (s^T B s).
  * Cauchy-point segment updates use B_d := B d and B_z := B z
    maintained in n-space, costing O(n) per segment.
  * Subspace minimization is a dense Cholesky solve on B_{FF},
    instead of the Sherman-Morrison-Woodbury / 2m x 2m K-system in
    L-BFGS-B.

Asymptotic cost per outer iteration:
    Cauchy point:        O(n^2)   (n breakpoints * O(n) B_d update)
    Subspace solve:      O(|F|^3) (Cholesky factorization)
    BFGS update:         O(n^2)

This implementation depends on NumPy only.

I am not aware of a canonical "BFGS-B" reference in the literature -
when dense quasi-Newton meets bound constraints in published work it
tends to be inside a trust-region framework (e.g. Lin & More's TRON,
Birgin & Martinez's GENCAN) rather than the line-search / projected-
gradient framework used here.  This file is best read as "the
limited-memory machinery of Byrd et al. 1995 instantiated with a
full Hessian approximation" rather than as a port of any specific
prior implementation.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Bound-type codes (same convention as the SciPy / Fortran L-BFGS-B interface)
# ----------------------------------------------------------------------------
_UNBOUNDED   = 0
_LOWER_ONLY  = 1
_BOTH_BOUNDS = 2
_UPPER_ONLY  = 3

_EPS = float(np.finfo(np.float64).eps)


# ============================================================================
# Bound handling utilities
# ============================================================================

def _make_nbd(l: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Build the nbd array of per-variable bound types from finite/infinite (l, u)."""
    has_lo = np.isfinite(l)
    has_hi = np.isfinite(u)
    nbd = np.zeros(l.shape[0], dtype=np.int64)
    nbd[ has_lo & ~has_hi] = _LOWER_ONLY
    nbd[ has_lo &  has_hi] = _BOTH_BOUNDS
    nbd[~has_lo &  has_hi] = _UPPER_ONLY
    return nbd


def _project(x: np.ndarray, l: np.ndarray, u: np.ndarray,
             nbd: np.ndarray) -> np.ndarray:
    """Componentwise projection onto {z : l <= z <= u}."""
    out = x.copy()
    has_lo = (nbd == _LOWER_ONLY) | (nbd == _BOTH_BOUNDS)
    has_hi = (nbd == _UPPER_ONLY) | (nbd == _BOTH_BOUNDS)
    if has_lo.any():
        out[has_lo] = np.maximum(out[has_lo], l[has_lo])
    if has_hi.any():
        out[has_hi] = np.minimum(out[has_hi], u[has_hi])
    return out


def _proj_grad_inf_norm(x: np.ndarray, g: np.ndarray, l: np.ndarray,
                        u: np.ndarray, nbd: np.ndarray) -> float:
    """||P(x - g, l, u) - x||_inf, the standard KKT residual for box problems."""
    return float(np.max(np.abs(_project(x - g, l, u, nbd) - x)))


def _max_feasible_step(x: np.ndarray, d: np.ndarray, l: np.ndarray,
                       u: np.ndarray, nbd: np.ndarray) -> float:
    """Largest alpha > 0 with x + alpha d feasible.  Returns +inf if d is
    feasibly unbounded."""
    alpha_max = np.inf
    has_lo = (nbd == _LOWER_ONLY) | (nbd == _BOTH_BOUNDS)
    has_hi = (nbd == _UPPER_ONLY) | (nbd == _BOTH_BOUNDS)

    hit_upper = has_hi & (d > 0)
    if hit_upper.any():
        alpha_max = min(alpha_max,
                        float(np.min((u[hit_upper] - x[hit_upper]) / d[hit_upper])))
    hit_lower = has_lo & (d < 0)
    if hit_lower.any():
        alpha_max = min(alpha_max,
                        float(np.min((l[hit_lower] - x[hit_lower]) / d[hit_lower])))
    return alpha_max


# ============================================================================
# Dense BFGS state
# ============================================================================

class _BFGSState:
    """Maintains a dense n*n positive-definite Hessian approximation B.

    On the first successful curvature pair, B is rescaled to
        B := (y^T y / y^T s) * I,
    the dense analog of the L-BFGS-B initial scaling theta_0 = y^T y / y^T s.
    Subsequent updates apply the standard rank-2 BFGS formula in place.
    """

    def __init__(self, n: int):
        self.n = n
        self.B = np.eye(n, dtype=np.float64)
        self.had_first_update = False

    def Bv(self, v: np.ndarray) -> np.ndarray:
        """Compute B v."""
        return self.B @ v

    def reset(self) -> None:
        """Discard curvature information and reinitialize B := I."""
        self.B[:] = np.eye(self.n)
        self.had_first_update = False

    def try_update(self, s: np.ndarray, y: np.ndarray) -> bool:
        """Apply the BFGS rank-2 update with the (s, y) pair.

        Returns False (without modifying state) if the curvature condition
        s^T y > eps * ||y||^2 is violated, or if the rank-2 update would
        produce a non-positive-definite Hessian (i.e. s^T B s <= 0, which
        cannot happen mathematically when B is PD but is guarded for safety).
        """
        sty = float(s @ y)
        yty = float(y @ y)
        if sty <= _EPS * max(yty, 1.0):
            return False  # curvature condition failed; skip

        if not self.had_first_update:
            # Self-scaling: B_0 <- (y^T y / y^T s) * I, then standard BFGS.
            scale = yty / sty
            self.B[:] = 0.0
            np.fill_diagonal(self.B, scale)
            self.had_first_update = True

        Bs  = self.B @ s
        sBs = float(s @ Bs)
        if sBs <= 0.0:
            return False  # defensive; shouldn't occur if B was PD and sty > 0

        # B += (y y^T) / (s^T y) - (B s)(B s)^T / (s^T B s)
        self.B += np.outer(y, y) / sty - np.outer(Bs, Bs) / sBs

        # Force symmetry to suppress roundoff drift (cheap, n^2).
        self.B[:] = 0.5 * (self.B + self.B.T)

        return True


# ============================================================================
# Phase 1 - Generalized Cauchy Point (dense version)
# ============================================================================

def _cauchy_point(x: np.ndarray, g: np.ndarray, l: np.ndarray, u: np.ndarray,
                  nbd: np.ndarray, state: _BFGSState
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generalized Cauchy point with a dense Hessian approximation.

    Returns
    -------
    xcp : ndarray, shape (n,)
        The generalized Cauchy point.
    Bz : ndarray, shape (n,)
        B (xcp - x_k), the dense analog of c = W^T (xcp - x_k) used in the
        compact-form L-BFGS-B Cauchy point.  Subspace minimization uses
        this to form the reduced gradient r = g + B(xcp - x).
    free_mask : ndarray of bool, shape (n,)
        True at indices whose xcp value is strictly interior.

    Follows Algorithm CP of Byrd et al. 1995, with all per-segment
    bookkeeping done in n-space rather than 2m-space.
    """
    n = x.shape[0]
    has_lo = (nbd == _LOWER_ONLY) | (nbd == _BOTH_BOUNDS)
    has_hi = (nbd == _UPPER_ONLY) | (nbd == _BOTH_BOUNDS)

    # --- Breakpoints t_i and initial path direction d_i (eqn. 4.1) ---------
    t = np.full(n, np.inf)
    mask_neg = (g < 0) & has_hi
    t[mask_neg] = (x[mask_neg] - u[mask_neg]) / g[mask_neg]
    mask_pos = (g > 0) & has_lo
    t[mask_pos] = (x[mask_pos] - l[mask_pos]) / g[mask_pos]

    d = -g.copy()
    d[t == 0.0] = 0.0  # variable already pinned at the relevant bound

    free = t > 0.0
    xcp  = x.copy()

    if not np.any(free):
        return xcp, np.zeros(n), free.copy()

    # --- Initial dense quantities ------------------------------------------
    # z := x(t) - x_k, accumulated along the path.
    # B_d := B d, B_z := B z, maintained incrementally.
    z   = np.zeros(n)
    B_d = state.Bv(d)
    B_z = np.zeros(n)

    f_prime  = float(g @ d)        # g^T d (= -||d||^2 on this initial segment)
    f_double = float(d @ B_d)      # d^T B d

    if f_double <= 0.0:
        f_double = _EPS  # safeguard against a concave/flat segment

    dt_min = -f_prime / f_double
    t_old  = 0.0

    free_idx = np.where(free)[0]
    order    = free_idx[np.argsort(t[free_idx])]
    pos      = 0

    b     = int(order[pos])
    t_cur = float(t[b])
    dt    = t_cur - t_old

    # --- Walk subsequent piecewise-linear segments -------------------------
    while dt_min >= dt and pos < len(order):
        # 1) Variable b just hit its bound: snap xcp[b].
        xcp[b] = u[b] if d[b] > 0 else l[b]

        # 2) Advance z and B_z to t_cur using the OLD d, B_d.
        z   = z   + dt * d
        B_z = B_z + dt * B_d

        g_b = float(g[b])

        # 3) Update f' (using new B_z, OLD f_double).
        #    Derivation: d := d_old + g_b e_b (so d_b transitions -g_b -> 0).
        #    f'_new = g^T d_new + d_new^T B z(t_cur)
        #           = f'_old + dt * f''_old           [advance within seg]
        #             + g_b^2                         [from g^T (g_b e_b)]
        #             + g_b * (B z(t_cur))_b          [from e_b^T B z]
        f_prime = (f_prime
                   + dt * f_double
                   + g_b * g_b
                   + g_b * float(B_z[b]))

        # 4) Update f'' (using OLD B_d and B[b,b]).
        #    f''_new = d_new^T B d_new
        #           = f''_old + 2 g_b * (B d_old)_b + g_b^2 * B_{b,b}
        f_double = (f_double
                    + 2.0 * g_b * float(B_d[b])
                    + g_b * g_b * float(state.B[b, b]))

        # 5) Zero out d[b].
        d[b] = 0.0

        # 6) Update B_d to reflect the change in d.  d changed by +g_b e_b,
        #    so B_d picks up + g_b * B[:, b].
        B_d = B_d + g_b * state.B[:, b]

        if f_double <= 0.0:
            f_double = _EPS

        dt_min = -f_prime / f_double
        t_old  = t_cur
        pos   += 1
        if pos >= len(order):
            break
        b     = int(order[pos])
        t_cur = float(t[b])
        dt    = t_cur - t_old

    # --- Finalize: advance the remaining free variables by dt_min ----------
    dt_min = max(dt_min, 0.0)
    t_total = t_old + dt_min
    still_free = order[pos:]
    if still_free.size > 0:
        xcp[still_free] = x[still_free] + t_total * d[still_free]
    # Bring B_z up to date as well; we will hand it back for use in
    # subspace minimization.
    B_z = B_z + dt_min * B_d

    free_mask = np.zeros(n, dtype=bool)
    free_mask[still_free] = True

    return xcp, B_z, free_mask


# ============================================================================
# Phase 2 - Subspace minimization (dense Cholesky)
# ============================================================================

def _subspace_minimize(x: np.ndarray, g: np.ndarray, xcp: np.ndarray,
                       Bz: np.ndarray, l: np.ndarray, u: np.ndarray,
                       nbd: np.ndarray, free_mask: np.ndarray,
                       state: _BFGSState) -> np.ndarray:
    """Solve B_{FF} d_F = -r_F by dense Cholesky on the free-face submatrix.

    The reduced gradient is r = g_k + B_k (xcp - x_k), so on the free face
        r_F = g[F] + B (xcp - x_k)[F] = g[F] + B_z[F].
    The unconstrained subspace step d_F is then truncated against the box.
    """
    free_idx = np.where(free_mask)[0]
    t_free   = free_idx.size
    if t_free == 0:
        return xcp.copy()

    r_F = g[free_idx] + Bz[free_idx]

    # Principal submatrix of B on the free indices.
    B_FF = state.B[np.ix_(free_idx, free_idx)]
    # Symmetrize against roundoff before Cholesky.
    B_FF = 0.5 * (B_FF + B_FF.T)

    try:
        L_chol = np.linalg.cholesky(B_FF)
    except np.linalg.LinAlgError:
        # B_FF lost PD-ness due to roundoff (very rare).  Fall back to xcp;
        # we still get the Cauchy decrease.
        return xcp.copy()

    # Solve L L^T d_F = -r_F.
    tmp = np.linalg.solve(L_chol,  -r_F)
    d_F = np.linalg.solve(L_chol.T, tmp)

    # Truncate to the box: largest alpha in [0, 1] with xcp_F + alpha d_F feasible.
    alpha = 1.0
    nbd_F = nbd[free_idx]
    has_lo_F = (nbd_F == _LOWER_ONLY) | (nbd_F == _BOTH_BOUNDS)
    has_hi_F = (nbd_F == _UPPER_ONLY) | (nbd_F == _BOTH_BOUNDS)

    up_hits = has_hi_F & (d_F > 0)
    if up_hits.any():
        alpha = min(alpha, float(np.min(
            (u[free_idx][up_hits] - xcp[free_idx][up_hits]) / d_F[up_hits])))
    lo_hits = has_lo_F & (d_F < 0)
    if lo_hits.any():
        alpha = min(alpha, float(np.min(
            (l[free_idx][lo_hits] - xcp[free_idx][lo_hits]) / d_F[lo_hits])))
    alpha = max(alpha, 0.0)

    z_out = xcp.copy()
    z_out[free_idx] = xcp[free_idx] + alpha * d_F
    z_out = _project(z_out, l, u, nbd)
    return z_out


# ============================================================================
# Phase 3 - Strong-Wolfe line search (bracket + zoom)
# ============================================================================

def _line_search_wolfe(fun_grad, x0, d, f0, g0, alpha_max,
                       c1=1e-4, c2=0.9, max_iter=25):
    """Strong-Wolfe line search along x0 + alpha d.

    Returns (alpha, f_new, g_new, nfev).  Returns (None, None, None, nfev)
    if the search fails (e.g. non-descent direction).

    Nocedal & Wright, "Numerical Optimization", Algorithms 3.5 and 3.6.
    """
    nfev = 0
    phi0  = float(f0)
    dphi0 = float(g0 @ d)
    if not (dphi0 < 0):
        return None, None, None, 0

    alpha_init = min(1.0, 0.99 * alpha_max) if alpha_max < np.inf else 1.0
    alpha_init = max(alpha_init, 1e-30)

    alpha_prev = 0.0
    phi_prev   = phi0
    dphi_prev  = dphi0
    alpha      = alpha_init

    f_new = None
    g_new = None

    for i in range(max_iter):
        x_new = x0 + alpha * d
        f_new, g_new = fun_grad(x_new)
        nfev += 1
        phi  = float(f_new)
        dphi = float(g_new @ d)

        if (phi > phi0 + c1 * alpha * dphi0) or (i > 0 and phi >= phi_prev):
            a, fv, gv, n2 = _zoom(alpha_prev, alpha, phi_prev, phi,
                                  dphi_prev, fun_grad, x0, d,
                                  phi0, dphi0, c1, c2,
                                  max_iter=max_iter)
            return a, fv, gv, nfev + n2

        if abs(dphi) <= -c2 * dphi0:
            return alpha, f_new, g_new, nfev

        if dphi >= 0:
            a, fv, gv, n2 = _zoom(alpha, alpha_prev, phi, phi_prev,
                                  dphi, fun_grad, x0, d,
                                  phi0, dphi0, c1, c2,
                                  max_iter=max_iter)
            return a, fv, gv, nfev + n2

        alpha_prev = alpha
        phi_prev   = phi
        dphi_prev  = dphi
        if alpha >= 0.99 * alpha_max:
            return alpha, f_new, g_new, nfev
        alpha = min(2.0 * alpha, alpha_max)

    return alpha, f_new, g_new, nfev


def _zoom(a_lo, a_hi, phi_lo, phi_hi, dphi_lo, fun_grad, x0, d,
          phi0, dphi0, c1, c2, max_iter=25):
    """The 'zoom' phase, safeguarded bisection."""
    nfev = 0
    f_new = None
    g_new = None
    alpha = 0.5 * (a_lo + a_hi)
    for _ in range(max_iter):
        alpha = 0.5 * (a_lo + a_hi)
        if abs(a_hi - a_lo) < 1e-16 * max(abs(a_lo), 1.0):
            x_new = x0 + alpha * d
            f_new, g_new = fun_grad(x_new)
            return alpha, f_new, g_new, nfev + 1

        x_new = x0 + alpha * d
        f_new, g_new = fun_grad(x_new)
        nfev += 1
        phi  = float(f_new)
        dphi = float(g_new @ d)

        if (phi > phi0 + c1 * alpha * dphi0) or (phi >= phi_lo):
            a_hi   = alpha
            phi_hi = phi
        else:
            if abs(dphi) <= -c2 * dphi0:
                return alpha, f_new, g_new, nfev
            if dphi * (a_hi - a_lo) >= 0:
                a_hi   = a_lo
                phi_hi = phi_lo
            a_lo    = alpha
            phi_lo  = phi
            dphi_lo = dphi

    return alpha, f_new, g_new, nfev


# ============================================================================
# Driver
# ============================================================================

class BFGSBResult(dict):
    """Result of a BFGS-B run.  Mirrors a tiny subset of scipy's
    OptimizeResult."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.__dict__ = self


def minimize_bfgsb(fun_and_grad, x0, bounds=None, factr=1e7, pgtol=1e-5,
                   maxiter=15000, max_ls=25, callback=None):
    """Minimize f(x) subject to l <= x <= u using full-Hessian BFGS-B.

    Parameters
    ----------
    fun_and_grad : callable
        f, g = fun_and_grad(x).  Must return a Python float and an
        n-vector NumPy ndarray.
    x0 : array_like
        Initial guess (will be projected into the feasible set).
    bounds : list of (lo, hi) or None
        Per-variable bounds; None or +-inf for unbounded.  None means
        unconstrained optimization.
    factr : float
        Function-value progress tolerance:
            stop when (f_old - f_new)/max(|f_old|,|f_new|,1) <= factr * eps.
    pgtol : float
        Projected-gradient infinity-norm tolerance.
    maxiter : int
        Maximum number of outer iterations.
    max_ls : int
        Maximum line search iterations per outer iteration.
    callback : callable or None
        Called as callback(x) after each accepted iterate.

    Returns
    -------
    BFGSBResult with attributes
        x, f, g, status, message, nit, nfev, pg_norm.

    Notes
    -----
    Storage is O(n^2) and each iteration costs O(n^2) for Cauchy plus
    O(|F|^3) for the Cholesky.  For n <= a few hundred this is
    completely uncompetitive with the limited-memory variant in terms
    of clock time, but the dense BFGS Hessian is sometimes preferable
    for ill-conditioned problems where m correction pairs do not
    capture the relevant curvature, or when one wants the full B at
    convergence for downstream use (e.g. covariance estimates).
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    n = x.shape[0]

    if bounds is None:
        l = np.full(n, -np.inf)
        u = np.full(n,  np.inf)
    else:
        if len(bounds) != n:
            raise ValueError(f"bounds has length {len(bounds)} but x0 has length {n}")
        l = np.array([(b[0] if (b[0] is not None) else -np.inf) for b in bounds], dtype=np.float64)
        u = np.array([(b[1] if (b[1] is not None) else  np.inf) for b in bounds], dtype=np.float64)
        if np.any(l > u):
            raise ValueError("Found a lower bound exceeding the corresponding upper bound.")

    nbd  = _make_nbd(l, u)
    ftol = factr * _EPS

    x = _project(x, l, u, nbd)

    f, g = fun_and_grad(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    nfev = 1

    state = _BFGSState(n)
    pg_norm = _proj_grad_inf_norm(x, g, l, u, nbd)

    if pg_norm <= pgtol:
        return BFGSBResult(x=x, f=f, g=g, status=0,
                           message="CONVERGENCE: projected gradient below pgtol",
                           nit=0, nfev=nfev, pg_norm=pg_norm)

    status  = -1
    message = "maxiter reached"
    k       = 0

    for k in range(maxiter):
        # --- Phase 1: Cauchy point ---
        xcp, Bz, free_mask = _cauchy_point(x, g, l, u, nbd, state)

        # --- Phase 2: Subspace minimization ---
        if free_mask.any():
            z = _subspace_minimize(x, g, xcp, Bz, l, u, nbd, free_mask, state)
        else:
            z = xcp

        # --- Phase 3: Line search along d = z - x ---
        d  = z - x
        gd = float(g @ d)
        descent_tol = _EPS * max(abs(f), 1.0)
        if gd >= descent_tol:
            # B is numerically corrupt: refresh and retry from scratch.
            state.reset()
            continue
        if gd >= -descent_tol:
            # Numerically a stationary point: no further descent possible
            # at machine precision.  Use pg_norm to label the exit.
            if pg_norm <= pgtol:
                status, message = 0, "CONVERGENCE: projected gradient below pgtol"
            else:
                status, message = 1, "CONVERGENCE: search direction below machine precision"
            break

        alpha_max = _max_feasible_step(x, d, l, u, nbd)
        if alpha_max <= 0.0:
            # Snapped to active bounds; no feasible move along d.  If pg_norm
            # is already at the user's tolerance, this is a constrained
            # stationary point, not an algorithmic failure.
            if pg_norm <= pgtol:
                status, message = 0, "CONVERGENCE: projected gradient below pgtol"
            else:
                status, message = 3, "abnormal: no feasible step along d"
            break

        alpha, f_new, g_new, ls_nfev = _line_search_wolfe(
            fun_and_grad, x, d, f, g, alpha_max,
            c1=1e-4, c2=0.9, max_iter=max_ls)
        nfev += ls_nfev

        if alpha is None or f_new is None:
            if not state.had_first_update:
                status, message = 3, "abnormal: line search failed with B = I"
                break
            state.reset()
            continue

        f_new = float(f_new)
        g_new = np.asarray(g_new, dtype=np.float64)

        x_new = _project(x + alpha * d, l, u, nbd)

        s = x_new - x
        y = g_new - g

        pg_norm = _proj_grad_inf_norm(x_new, g_new, l, u, nbd)
        rel_red = (f - f_new) / max(abs(f), abs(f_new), 1.0)

        x, f, g = x_new, f_new, g_new
        state.try_update(s, y)

        if callback is not None:
            callback(x)

        if pg_norm <= pgtol:
            status, message = 0, "CONVERGENCE: projected gradient below pgtol"
            break
        if rel_red <= ftol:
            status, message = 1, "CONVERGENCE: relative reduction of f below factr*eps"
            break

    if status == -1:
        status = 2

    return BFGSBResult(x=x, f=f, g=g, status=status, message=message,
                       nit=k + 1, nfev=nfev, pg_norm=pg_norm)


# ============================================================================
# Convenience: wrap a function returning only f, with finite-difference grad.
# ============================================================================

def _wrap_with_fd_gradient(func, eps=None):
    """Wrap f(x) -> (f(x), grad(x)) using forward differences."""
    eps = eps if eps is not None else np.sqrt(_EPS)
    def fun_and_grad(x):
        f0 = float(func(x))
        g  = np.empty_like(x)
        for i in range(x.shape[0]):
            xp = x.copy()
            xp[i] += eps
            g[i] = (float(func(xp)) - f0) / eps
        return f0, g
    return fun_and_grad


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    import sys

    np.random.seed(0)

    _failed = []

    def _check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            _failed.append(name)

    def _close(a, b, tol=1e-5):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) <= tol

    # --- Test problems -----------------------------------------------------
    def quadratic(A, b):
        def fg(x):
            Ax = A @ x
            return 0.5 * float(x @ Ax) - float(b @ x), Ax - b
        return fg

    def rosenbrock(x):
        x = np.asarray(x, dtype=np.float64)
        f = float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1 - x[:-1])
        g[1:]  +=  200.0 * (x[1:] - x[:-1] ** 2)
        return f, g

    def gradient_check(fg, x, eps=1e-6, tol=1e-5):
        f0, g_an = fg(x)
        g_fd = np.empty_like(x)
        for i in range(x.shape[0]):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            g_fd[i] = (fg(xp)[0] - fg(xm)[0]) / (2 * eps)
        return float(np.max(np.abs(g_an - g_fd))) <= tol

    # ----------------------------------------------------------------------
    # Test 1: 2D unconstrained quadratic
    # ----------------------------------------------------------------------
    print("\n=== Test 1: 2D unconstrained quadratic ===")
    A = np.array([[3.0, 0.5], [0.5, 2.0]])
    b = np.array([1.0, 2.0])
    x_star = np.linalg.solve(A, b)
    fg = quadratic(A, b)
    res = minimize_bfgsb(fg, np.zeros(2))
    _check("converged", res.status in (0, 1), f"status={res.status}, msg='{res.message}'")
    _check("x matches analytic minimum", _close(res.x, x_star, tol=1e-5),
           f"||x - x*|| = {np.linalg.norm(res.x - x_star):.2e}")
    _check("pg_norm small", res.pg_norm < 1e-4, f"pg_norm = {res.pg_norm:.2e}")
    # On a 2-D quadratic with self-scaling and Wolfe line search, BFGS-B
    # typically takes around 4-8 iterations (a few iters to learn B, then
    # near-quadratic convergence).  The L-BFGS-B sibling on the same problem
    # takes ~7 iters; use the same generous bound.
    _check("converges in <=10 iters", res.nit <= 10, f"nit = {res.nit}")

    # ----------------------------------------------------------------------
    # Test 2: bounds present but inactive
    # ----------------------------------------------------------------------
    print("\n=== Test 2: bounds present but not active at minimizer ===")
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    res = minimize_bfgsb(fg, np.array([4.0, -4.0]), bounds=bounds)
    _check("converged", res.status in (0, 1))
    _check("x matches unconstrained minimum",
           _close(res.x, x_star, tol=1e-5),
           f"||x - x*|| = {np.linalg.norm(res.x - x_star):.2e}")

    # ----------------------------------------------------------------------
    # Test 3: bounds active at minimizer
    # ----------------------------------------------------------------------
    print("\n=== Test 3: bounds active at minimizer ===")
    bounds = [(1.0, None), (None, None)]
    res = minimize_bfgsb(fg, np.array([5.0, 5.0]), bounds=bounds)
    _check("converged", res.status in (0, 1))
    _check("x[0] pinned at lower bound", abs(res.x[0] - 1.0) < 1e-6,
           f"x[0] = {res.x[0]:.6f}")
    x1_star = (b[1] - A[1, 0] * 1.0) / A[1, 1]
    _check("x[1] satisfies stationarity",
           abs(res.x[1] - x1_star) < 1e-5,
           f"x[1] = {res.x[1]:.6f} (expected {x1_star:.6f})")

    # ----------------------------------------------------------------------
    # Test 4: corner solution
    # ----------------------------------------------------------------------
    print("\n=== Test 4: corner solution ===")
    n = 3
    c_vec = np.array([-1.0, -2.0, -3.0])
    def linear_fg(x):
        return float(c_vec @ x), c_vec.copy()
    bounds = [(0.0, 1.0)] * n
    res = minimize_bfgsb(linear_fg, np.array([0.5, 0.5, 0.5]), bounds=bounds)
    _check("converged", res.status in (0, 1))
    _check("x at corner (1,1,1)", _close(res.x, np.ones(n), tol=1e-6),
           f"x = {res.x}")

    # ----------------------------------------------------------------------
    # Test 5: Rosenbrock (unconstrained)
    # ----------------------------------------------------------------------
    print("\n=== Test 5: Rosenbrock, n=5, unconstrained ===")
    n = 5
    _check("Rosenbrock gradient matches FD",
           gradient_check(rosenbrock, np.array([-1.2, 1.0, 0.5, 0.0, 2.0])))
    res = minimize_bfgsb(rosenbrock, -np.ones(n) * 1.2, pgtol=1e-8, maxiter=2000)
    _check("converged", res.status in (0, 1), f"status={res.status}")
    _check("at the global minimum (1,...,1)",
           _close(res.x, np.ones(n), tol=1e-3),
           f"||x-1|| = {np.linalg.norm(res.x - 1):.2e}, f = {res.f:.2e}")

    # ----------------------------------------------------------------------
    # Test 6: Rosenbrock with active bounds
    # ----------------------------------------------------------------------
    print("\n=== Test 6: Rosenbrock with bounds [0, 0.5]^n ===")
    n = 4
    bounds = [(0.0, 0.5)] * n
    res = minimize_bfgsb(rosenbrock, np.full(n, 0.3), bounds=bounds,
                         pgtol=1e-10, factr=1.0, maxiter=3000)
    _check("converged", res.status in (0, 1))
    _check("all components feasible",
           bool(np.all(res.x >= -1e-9) and np.all(res.x <= 0.5 + 1e-9)),
           f"x = {res.x}")
    try:
        from scipy.optimize import minimize as _sp_minimize
        _sp = _sp_minimize(lambda x: rosenbrock(x)[0], np.full(n, 0.3),
                           jac=lambda x: rosenbrock(x)[1],
                           method="L-BFGS-B", bounds=bounds,
                           options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 3000})
        _check("x agrees with scipy L-BFGS-B on bounded Rosenbrock",
               _close(res.x, _sp.x, tol=1e-6),
               f"||x_mine - x_sp|| = {np.linalg.norm(res.x - _sp.x):.2e}")
    except ImportError:
        pass

    # ----------------------------------------------------------------------
    # Test 7: 50-dim quadratic with mixed bounds
    # ----------------------------------------------------------------------
    print("\n=== Test 7: 50-dim quadratic with mixed bounds ===")
    n = 50
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + n * np.eye(n)
    b_vec = rng.standard_normal(n)
    fg7 = quadratic(A, b_vec)
    x_unc = np.linalg.solve(A, b_vec)
    bounds = [(-np.inf, 0.0) if i % 3 == 0 else (None, None) for i in range(n)]
    res = minimize_bfgsb(fg7, np.zeros(n), bounds=bounds,
                         pgtol=1e-6, factr=10.0, maxiter=2000)
    _check("converged", res.status in (0, 1), f"status={res.status}")
    _check("all feasible",
           bool(np.all(res.x[::3] <= 1e-8)),
           f"max violation = {float(np.max(res.x[::3])):.2e}")
    _check("pg_norm small", res.pg_norm < 1e-5,
           f"pg_norm = {res.pg_norm:.2e}")
    x_clamped = x_unc.copy()
    x_clamped[::3] = np.minimum(x_clamped[::3], 0.0)
    f_clamped, _ = fg7(x_clamped)
    _check("objective beats naive clamp",
           res.f <= f_clamped + 1e-8,
           f"f = {res.f:.6e}, f(clamp) = {f_clamped:.6e}")

    # ----------------------------------------------------------------------
    # Test 8: cross-check against L-BFGS-B (sibling module) and scipy
    # ----------------------------------------------------------------------
    print("\n=== Test 8: cross-check against L-BFGS-B and scipy ===")
    try:
        import lbfgsb as _lbfgsb_mod
        have_lbfgsb = True
    except ImportError:
        have_lbfgsb = False

    try:
        from scipy.optimize import minimize as sp_minimize
        have_scipy = True
    except ImportError:
        have_scipy = False

    A2 = np.array([[4.0, 1.0], [1.0, 3.0]])
    b2 = np.array([1.0, 2.0])
    fg8 = quadratic(A2, b2)
    bounds = [(0.5, 2.0), (-1.0, 2.0)]
    x0 = np.array([0.5, 0.5])

    res_mine = minimize_bfgsb(fg8, x0, bounds=bounds, pgtol=1e-9, factr=1.0)
    if have_scipy:
        res_sp = sp_minimize(lambda x: fg8(x)[0], x0, jac=lambda x: fg8(x)[1],
                             method="L-BFGS-B", bounds=bounds,
                             options={"gtol": 1e-9, "ftol": 1e-15})
        _check("BFGS-B agrees with scipy on bounded quadratic",
               _close(res_mine.x, res_sp.x, tol=1e-6),
               f"||diff|| = {np.linalg.norm(res_mine.x - res_sp.x):.2e}")
        _check("BFGS-B f agrees with scipy on bounded quadratic",
               abs(res_mine.f - res_sp.fun) < 1e-9,
               f"|df| = {abs(res_mine.f - res_sp.fun):.2e}")

    if have_lbfgsb:
        res_lb = _lbfgsb_mod.minimize_lbfgsb(fg8, x0, bounds=bounds,
                                             pgtol=1e-9, factr=1.0)
        _check("BFGS-B agrees with sibling L-BFGS-B on bounded quadratic",
               _close(res_mine.x, res_lb.x, tol=1e-6),
               f"||diff|| = {np.linalg.norm(res_mine.x - res_lb.x):.2e}")

    # Bounded Rosenbrock
    nr = 6
    bounds_r = [(-1.5, 1.5)] * nr
    x0_r = -np.ones(nr) * 1.2
    res_mine_r = minimize_bfgsb(rosenbrock, x0_r, bounds=bounds_r,
                                pgtol=1e-8, factr=1.0, maxiter=2000)
    if have_scipy:
        res_sp_r = sp_minimize(lambda x: rosenbrock(x)[0], x0_r,
                               jac=lambda x: rosenbrock(x)[1],
                               method="L-BFGS-B", bounds=bounds_r,
                               options={"gtol": 1e-8, "ftol": 1e-15, "maxiter": 2000})
        _check("BFGS-B agrees with scipy on bounded Rosenbrock",
               _close(res_mine_r.x, res_sp_r.x, tol=1e-4),
               f"||diff|| = {np.linalg.norm(res_mine_r.x - res_sp_r.x):.2e}")
    if have_lbfgsb:
        res_lb_r = _lbfgsb_mod.minimize_lbfgsb(rosenbrock, x0_r, bounds=bounds_r,
                                               pgtol=1e-8, factr=1.0, maxiter=2000)
        _check("BFGS-B agrees with L-BFGS-B on bounded Rosenbrock",
               _close(res_mine_r.x, res_lb_r.x, tol=1e-4),
               f"||diff|| = {np.linalg.norm(res_mine_r.x - res_lb_r.x):.2e}")

    # ----------------------------------------------------------------------
    # Test 9: FD-wrapped Rosenbrock
    # ----------------------------------------------------------------------
    print("\n=== Test 9: FD-gradient wrapper on Rosenbrock 3D ===")
    rosen_f = lambda x: rosenbrock(x)[0]
    fg_fd = _wrap_with_fd_gradient(rosen_f, eps=1e-7)
    res = minimize_bfgsb(fg_fd, np.array([-1.2, 1.0, 0.5]), pgtol=1e-5, maxiter=500)
    _check("FD-wrapped Rosenbrock converges near (1,1,1)",
           _close(res.x, np.ones(3), tol=1e-2),
           f"x = {res.x}, f = {res.f:.2e}")

    # ----------------------------------------------------------------------
    # Test 10: KKT residual sanity
    # ----------------------------------------------------------------------
    print("\n=== Test 10: KKT residual sanity ===")
    for label, fg_, x0, bnds in [
        ("quadratic-2d",      fg,         np.array([0.0, 0.0]),  None),
        ("rosenbrock-5d",     rosenbrock, -np.ones(5) * 1.2,     None),
        ("rosenbrock-4d-box", rosenbrock, np.full(4, 0.3),       [(0.0, 0.5)] * 4),
    ]:
        res = minimize_bfgsb(fg_, x0, bounds=bnds, pgtol=1e-8,
                             factr=1.0, maxiter=5000)
        ok = res.pg_norm <= 1e-5
        _check(f"  KKT ok for {label}", ok,
               f"pg_norm={res.pg_norm:.2e}, status={res.status}")

    # ----------------------------------------------------------------------
    # Test 11: BFGS secant equation B s = y holds after each update
    # ----------------------------------------------------------------------
    print("\n=== Test 11: BFGS secant equation B s = y after update ===")
    rng = np.random.default_rng(11)
    n = 6
    state = _BFGSState(n)
    # Force the first update to happen by manually calling try_update with
    # an arbitrary curvature pair.
    s0 = rng.standard_normal(n)
    y0 = rng.standard_normal(n)
    while float(s0 @ y0) <= 1e-3:           # ensure healthy curvature
        s0 = rng.standard_normal(n)
        y0 = rng.standard_normal(n)
    state.try_update(s0, y0)
    err0 = float(np.max(np.abs(state.B @ s0 - y0)))
    _check("secant equation holds after first update", err0 < 1e-12,
           f"max|B s - y| = {err0:.2e}")
    # Apply several more random updates and re-check on each.
    worst = err0
    for _ in range(20):
        s_k = rng.standard_normal(n)
        y_k = state.B @ s_k + 0.1 * rng.standard_normal(n)  # ensure sty > 0 likely
        if float(s_k @ y_k) <= 1e-6:
            continue
        if state.try_update(s_k, y_k):
            worst = max(worst, float(np.max(np.abs(state.B @ s_k - y_k))))
    _check("secant equation holds across repeated updates", worst < 1e-10,
           f"worst |B s - y| = {worst:.2e}")
    # B should remain symmetric positive definite.
    sym_err = float(np.max(np.abs(state.B - state.B.T)))
    eigvals = np.linalg.eigvalsh(state.B)
    _check("B stays symmetric", sym_err < 1e-12, f"max|B - B^T| = {sym_err:.2e}")
    _check("B stays positive definite", float(np.min(eigvals)) > 0,
           f"min eigenvalue = {float(np.min(eigvals)):.2e}")

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    if _failed:
        print(f"FAILED: {len(_failed)} test(s)")
        for name in _failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)