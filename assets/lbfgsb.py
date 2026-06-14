"""
L-BFGS-B: Limited-memory BFGS for bound-constrained optimization
=================================================================

A single-file pure-NumPy implementation following:

  R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu (1995).
  "A limited memory algorithm for bound constrained optimization."
  SIAM Journal on Scientific Computing, 16(5), 1190-1208.

The algorithmic structure mirrors the reference Fortran/C implementation
distributed with SciPy (originally Zhu, Byrd, Lu, Nocedal 1997, with
revisions by Morales & Nocedal 2011):

    scipy/optimize/_lbfgsb_py.py              (Python driver)
    scipy/optimize/__lbfgsb/src/lbfgsb.c      (translated Fortran core)

We solve

        minimize    f(x)
        subject to  l <= x <= u,

with l, u in (R u {+/- inf})^n, by iterating three phases per step:

  Phase 1 - GENERALIZED CAUCHY POINT.  Walk the projected steepest-
            descent path P(x_k - t g_k, l, u) against the quadratic
            model
                m_k(x) = f_k + g_k^T (x - x_k)
                       + (1/2)(x - x_k)^T B_k (x - x_k)
            and locate the first piecewise-linear minimizer x^c,
            exposing a candidate active set A(x^c).
            (Section 4 and Algorithm CP of Byrd et al. 1995.)

  Phase 2 - SUBSPACE MINIMIZATION.  Approximately minimize m_k over
            the free variables (the complement of A(x^c)), holding
            the active variables fixed at their values at x^c.  We
            use the direct primal method (Section 5.1) with Sherman-
            Morrison-Woodbury inversion of (Z^T B_k Z), then truncate
            the candidate move so the new point lies in the box.

  Phase 3 - LINE SEARCH.  A strong-Wolfe bracket-and-zoom line search
            (Nocedal & Wright Alg. 3.5/3.6) along d = z - x_k, capped
            at the largest feasibility-preserving step length.

The L-BFGS approximation is maintained in the compact form of
Byrd, Nocedal, and Schnabel (1994):

        B_k = theta * I - W M W^T,
        W   = [ Y,   theta * S ],
        M^{-1} = [[ -D,            L^T          ],
                 [  L,   theta * S^T S         ]],

with S = [s_{k-m}, ..., s_{k-1}], Y = [y_{k-m}, ..., y_{k-1}],
D = diag(s_i^T y_i), and L the strictly lower triangular part of S^T Y.
Products with B_k then cost O(m n) per iteration, where m is the
number of stored correction pairs.

This implementation depends on NumPy only.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Bound-type codes (same convention as the reference Fortran/C interface).
# ----------------------------------------------------------------------------
_UNBOUNDED   = 0
_LOWER_ONLY  = 1
_BOTH_BOUNDS = 2
_UPPER_ONLY  = 3

# Numerical tolerances used throughout.
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
# Compact-form L-BFGS state
# ============================================================================

class _LBFGSState:
    """Stores S, Y and the derived compact-form matrices for B_k.

    Storage is the simple non-circular layout: when col reaches m, the
    oldest correction is dropped by shifting columns left.  This is
    slightly less cache-friendly than the circular buffer in lbfgsb.c
    but easier to read and equivalent algorithmically.
    """

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.theta = 1.0
        self.col = 0
        self.S  = np.zeros((n, m))
        self.Y  = np.zeros((n, m))
        self.SY = np.zeros((m, m))     # full S^T Y (only [:col, :col] is meaningful)
        self.SS = np.zeros((m, m))     # full S^T S
        self.U  = np.zeros((0, 0))     # upper Cholesky factor of T

    # ---- compact-form products --------------------------------------------

    def Wt_v(self, v: np.ndarray) -> np.ndarray:
        """Compute W^T v where W = [Y, theta * S]."""
        if self.col == 0:
            return np.zeros(0)
        out = np.empty(2 * self.col)
        out[:self.col] = self.Y[:, :self.col].T @ v
        out[self.col:] = self.theta * (self.S[:, :self.col].T @ v)
        return out

    def W_v(self, v: np.ndarray) -> np.ndarray:
        """Compute W v where v has length 2*col."""
        if self.col == 0:
            return np.zeros(self.n)
        return (self.Y[:, :self.col] @ v[:self.col]
                + self.theta * (self.S[:, :self.col] @ v[self.col:]))

    def W_row(self, i: int) -> np.ndarray:
        """Return W[i, :], the i-th row of W as a length-2*col vector."""
        if self.col == 0:
            return np.zeros(0)
        out = np.empty(2 * self.col)
        out[:self.col] = self.Y[i, :self.col]
        out[self.col:] = self.theta * self.S[i, :self.col]
        return out

    def Mv(self, v: np.ndarray) -> np.ndarray:
        """Compute p = M v by solving M^{-1} p = v.

        Uses the block-LDL factorization induced by the Schur complement
            T = theta * S^T S + L D^{-1} L^T,
        with T = U^T U the Cholesky factor stored in self.U.  See the
        Schur-complement derivation in the C subroutine ``bmv``.
        """
        col = self.col
        if col == 0:
            return np.zeros_like(v)

        v1 = v[:col]
        v2 = v[col:]
        d  = np.diag(self.SY)[:col]
        L  = np.tril(self.SY[:col, :col], -1)

        # Step 1: forward sweep, rhs_2 := v_2 + L D^{-1} v_1
        rhs2 = v2 + L @ (v1 / d)

        # Step 2: solve T p_2 = rhs_2, i.e. U^T U p_2 = rhs_2
        tmp = _solve_upper_triangular(self.U, rhs2, transpose=True)
        p2  = _solve_upper_triangular(self.U, tmp,  transpose=False)

        # Step 3: back sweep, p_1 := -D^{-1} v_1 + D^{-1} L^T p_2
        p1 = -v1 / d + (L.T @ p2) / d

        return np.concatenate([p1, p2])

    def Bv(self, v: np.ndarray) -> np.ndarray:
        """Compute B v = theta * v - W M W^T v."""
        if self.col == 0:
            return self.theta * v
        return self.theta * v - self.W_v(self.Mv(self.Wt_v(v)))

    # ---- maintenance ------------------------------------------------------

    def reset(self) -> None:
        """Discard all stored correction pairs (used after numerical trouble)."""
        self.col   = 0
        self.theta = 1.0
        self.U     = np.zeros((0, 0))

    def try_update(self, s: np.ndarray, y: np.ndarray) -> bool:
        """Attempt to add the (s, y) correction pair.  Returns False (and
        leaves state unchanged) if the curvature condition s^T y > eps ||y||^2
        is violated or if the new T loses positive definiteness."""
        sty = float(s @ y)
        yty = float(y @ y)
        if sty <= _EPS * max(yty, 1.0):
            return False  # skip update

        # New scaling theta_{k+1} = y^T y / s^T y (Byrd et al. eq. 3.10).
        new_theta = yty / sty

        m = self.m
        if self.col < m:
            k = self.col
            self.S[:, k] = s
            self.Y[:, k] = y
            # New row and column of S^T Y and S^T S.
            for j in range(k):
                self.SY[k, j] = float(s @ self.Y[:, j])
                self.SY[j, k] = float(self.S[:, j] @ y)
                self.SS[k, j] = float(s @ self.S[:, j])
                self.SS[j, k] = self.SS[k, j]
            self.SY[k, k] = sty
            self.SS[k, k] = float(s @ s)
            new_col = k + 1
        else:
            # Shift columns left to drop the oldest pair, then append.
            self.S[:, :-1] = self.S[:, 1:]
            self.Y[:, :-1] = self.Y[:, 1:]
            self.S[:, -1]  = s
            self.Y[:, -1]  = y
            self.SY[:-1, :-1] = self.SY[1:, 1:]
            self.SS[:-1, :-1] = self.SS[1:, 1:]
            for j in range(m - 1):
                self.SY[-1, j] = float(s @ self.Y[:, j])
                self.SY[j, -1] = float(self.S[:, j] @ y)
                self.SS[-1, j] = float(s @ self.S[:, j])
                self.SS[j, -1] = self.SS[-1, j]
            self.SY[-1, -1] = sty
            self.SS[-1, -1] = float(s @ s)
            new_col = m

        # Tentatively factor T with the new pair; back out on failure.
        new_U = _form_T_cholesky(new_theta, self.SY, self.SS, new_col)
        if new_U is None:
            # Roll back to the pre-update column count and theta.  We don't
            # need to roll back S/Y/SY/SS storage since they are overwritten
            # on the next successful update.
            return False

        self.theta = new_theta
        self.col   = new_col
        self.U     = new_U
        return True


def _form_T_cholesky(theta: float, SY: np.ndarray, SS: np.ndarray,
                     col: int) -> np.ndarray | None:
    """Cholesky factor U of T = theta * S^T S + L D^{-1} L^T, with T = U^T U.

    Returns the upper triangular U (shape (col, col)), or None if T is
    not numerically positive definite.
    """
    if col == 0:
        return np.zeros((0, 0))
    d = np.diag(SY)[:col].copy()
    if np.any(d <= 0):
        return None
    L = np.tril(SY[:col, :col], -1)
    T = theta * SS[:col, :col] + L @ (L.T / d[:, None])
    T = 0.5 * (T + T.T)  # symmetrize against roundoff
    try:
        # np.linalg.cholesky returns the lower-triangular factor L_c with
        # T = L_c L_c^T, so U = L_c^T satisfies T = U^T U.
        return np.linalg.cholesky(T).T
    except np.linalg.LinAlgError:
        return None


def _solve_upper_triangular(U: np.ndarray, b: np.ndarray,
                            transpose: bool = False) -> np.ndarray:
    """Solve U x = b (transpose=False) or U^T x = b (transpose=True),
    where U is upper triangular.  Pure NumPy; no SciPy dependency.
    """
    n = U.shape[0]
    x = np.empty_like(b, dtype=np.float64)
    if not transpose:
        # back-substitution
        for i in range(n - 1, -1, -1):
            s = b[i] - U[i, i+1:] @ x[i+1:]
            x[i] = s / U[i, i]
    else:
        # forward-substitution against U^T (which is lower triangular)
        for i in range(n):
            s = b[i] - U[:i, i] @ x[:i]
            x[i] = s / U[i, i]
    return x


# ============================================================================
# Phase 1 — Generalized Cauchy Point
# ============================================================================

def _cauchy_point(x: np.ndarray, g: np.ndarray, l: np.ndarray, u: np.ndarray,
                  nbd: np.ndarray, state: _LBFGSState
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the generalized Cauchy point x^c.

    Returns
    -------
    xcp : ndarray, shape (n,)
        The generalized Cauchy point.
    c   : ndarray, shape (2*col,)
        c = W^T (xcp - x), required to initialize subspace minimization.
    free_mask : ndarray of bool, shape (n,)
        True at indices whose xcp value is strictly interior; these are
        the free variables for the subspace minimization step.

    This follows "Algorithm CP" in Section 4 of Byrd et al. (1995).
    """
    n   = x.shape[0]
    col = state.col

    has_lo = (nbd == _LOWER_ONLY) | (nbd == _BOTH_BOUNDS)
    has_hi = (nbd == _UPPER_ONLY) | (nbd == _BOTH_BOUNDS)

    # --- Breakpoints t_i and initial descent direction d_i (eqn. 4.1) ------
    t = np.full(n, np.inf)
    # g_i < 0 means x_i moves toward u_i along x_i - t g_i.
    mask_neg = (g < 0) & has_hi
    t[mask_neg] = (x[mask_neg] - u[mask_neg]) / g[mask_neg]
    # g_i > 0 means x_i moves toward l_i.
    mask_pos = (g > 0) & has_lo
    t[mask_pos] = (x[mask_pos] - l[mask_pos]) / g[mask_pos]

    d = -g.copy()
    d[t == 0.0] = 0.0           # already pinned at the relevant bound

    free = t > 0.0
    xcp  = x.copy()

    # If no variable can move (all breakpoints are 0), xcp = x.
    if not np.any(free):
        return xcp, np.zeros(2 * col), free.copy()

    # --- Initialize p = W^T d, c = 0, and the segment-derivative scalars ---
    if col > 0:
        p = state.Wt_v(d)
        c = np.zeros(2 * col)
        Mp = state.Mv(p)
        f_prime  = float(-d @ d)                          # g^T d
        f_double = float(-state.theta * f_prime - p @ Mp) # d^T B d
    else:
        p = np.zeros(0)
        c = np.zeros(0)
        f_prime  = float(-d @ d)
        f_double = float(state.theta * (d @ d))

    if f_double <= 0.0:
        f_double = _EPS  # safeguard a concave/flat segment

    dt_min = -f_prime / f_double
    t_old  = 0.0

    # Pre-sort the breakpoints in F by ascending t.  Walking the path is then
    # an O(nbreak) sweep through this order.
    free_idx = np.where(free)[0]
    order    = free_idx[np.argsort(t[free_idx])]
    pos      = 0

    b      = int(order[pos])
    t_cur  = float(t[b])
    dt     = t_cur - t_old

    # --- "Examination of subsequent segments" loop -------------------------
    while dt_min >= dt and pos < len(order):
        # 1) Fix variable b at the bound it just hit.
        xcp[b] = u[b] if d[b] > 0 else l[b]
        z_b    = xcp[b] - x[b]
        g_b    = float(g[b])

        if col > 0:
            # c is updated first using the OLD p.
            c = c + dt * p

            w_b   = state.W_row(b)
            Mc    = state.Mv(c)
            Mwb   = state.Mv(w_b)
            wbtMc = float(w_b @ Mc)
            wbtMp = float(w_b @ Mp)
            wbtMw = float(w_b @ Mwb)
        else:
            wbtMc = wbtMp = wbtMw = 0.0

        # 2) Update f' (uses OLD f_double) and f''.
        f_prime  = (f_prime + dt * f_double
                    + g_b * g_b
                    + state.theta * g_b * z_b
                    - g_b * wbtMc)
        f_double = (f_double
                    - state.theta * g_b * g_b
                    - 2.0 * g_b * wbtMp
                    - g_b * g_b * wbtMw)

        # 3) Update p, then Mp for the next iteration.
        if col > 0:
            p  = p + g_b * w_b
            Mp = state.Mv(p)

        d[b] = 0.0

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
    t_old += dt_min
    still_free = order[pos:]
    if still_free.size > 0:
        xcp[still_free] = x[still_free] + t_old * d[still_free]
    if col > 0:
        c = c + dt_min * p

    # Variables in order[:pos] were fixed during the walk; those in
    # still_free are interior at xcp.  Variables outside `free` (the original
    # F) had t_i == 0 and didn't move, so they're also active at xcp.
    free_mask = np.zeros(n, dtype=bool)
    free_mask[still_free] = True

    # Variables permanently unbounded (nbd == _UNBOUNDED) with g_i == 0 are
    # also free even though their breakpoint is +inf.  These are already
    # captured by still_free because t_i == +inf put them at the end of
    # `order`, and they only get processed if the walk reaches them (it
    # generally doesn't, because dt to +inf is larger than any finite dt_min).

    return xcp, c, free_mask


# ============================================================================
# Phase 2 — Subspace minimization
# ============================================================================

def _subspace_minimize(x: np.ndarray, g: np.ndarray, xcp: np.ndarray,
                      c_vec: np.ndarray, l: np.ndarray, u: np.ndarray,
                      nbd: np.ndarray, free_mask: np.ndarray,
                      state: _LBFGSState) -> np.ndarray:
    """Direct primal subspace minimization of m_k.

    Solves (approximately)
        min  Q(x) = r^T (x - xcp) + 1/2 (x - xcp)^T B (x - xcp)
        s.t. l <= x <= u,  x_i = xcp_i for i not in free_mask,
    where r = g_k + B (xcp - x_k).

    We use the Sherman-Morrison-Woodbury identity on B = theta I - W M W^T
    restricted to the free coordinates; see Section 5.1 of Byrd et al.
    The unconstrained subspace step is then truncated against the box.
    """
    col = state.col
    n   = x.shape[0]
    free_idx = np.where(free_mask)[0]
    t_free   = free_idx.size

    if t_free == 0:
        return xcp.copy()

    # Reduced gradient r_hat = Z^T (g + B (xcp - x_k)).
    #   B (xcp - x) = theta (xcp - x) - W M W^T (xcp - x) = theta (xcp - x) - W M c
    # so r_hat = g_F + theta (xcp_F - x_F) - W_F M c.
    if col > 0:
        Mc = state.Mv(c_vec)
        W_F = np.hstack([state.Y[free_idx, :col],
                         state.theta * state.S[free_idx, :col]])
        r_hat = (g[free_idx]
                 + state.theta * (xcp[free_idx] - x[free_idx])
                 - W_F @ Mc)
    else:
        W_F = None
        r_hat = g[free_idx] + state.theta * (xcp[free_idx] - x[free_idx])

    # ---- Unconstrained subspace step --------------------------------------
    if col == 0:
        # B = theta I; closed form.
        dhat = -r_hat / state.theta
    else:
        # SMW: (Z^T B Z)^{-1} = (1/theta) I + (1/theta^2) W_F K^{-1} W_F^T,
        # where K = M^{-1} - W_F^T W_F / theta is a 2*col x 2*col matrix.
        d_vec = np.diag(state.SY)[:col]
        L_mat = np.tril(state.SY[:col, :col], -1)
        M_inv = np.block([
            [-np.diag(d_vec),                      L_mat.T                       ],
            [ L_mat,                                state.theta * state.SS[:col, :col]],
        ])
        K = M_inv - (W_F.T @ W_F) / state.theta
        try:
            sol = np.linalg.solve(K, W_F.T @ r_hat)
        except np.linalg.LinAlgError:
            # K singular: fall back to xcp (we still get the Cauchy decrease).
            return xcp.copy()
        dhat = -r_hat / state.theta - (W_F @ sol) / (state.theta ** 2)

    # ---- Truncate to the box ---------------------------------------------
    # Largest alpha in [0, 1] with xcp_F + alpha dhat feasible.
    alpha = 1.0
    nbd_F = nbd[free_idx]
    has_lo_F = (nbd_F == _LOWER_ONLY) | (nbd_F == _BOTH_BOUNDS)
    has_hi_F = (nbd_F == _UPPER_ONLY) | (nbd_F == _BOTH_BOUNDS)

    up_hits = has_hi_F & (dhat > 0)
    if up_hits.any():
        alpha = min(alpha, float(np.min(
            (u[free_idx][up_hits] - xcp[free_idx][up_hits]) / dhat[up_hits])))
    lo_hits = has_lo_F & (dhat < 0)
    if lo_hits.any():
        alpha = min(alpha, float(np.min(
            (l[free_idx][lo_hits] - xcp[free_idx][lo_hits]) / dhat[lo_hits])))
    alpha = max(alpha, 0.0)

    z = xcp.copy()
    z[free_idx] = xcp[free_idx] + alpha * dhat
    # Snap to the bound that triggered alpha < 1, to avoid roundoff drift.
    z = _project(z, l, u, nbd)
    return z


# ============================================================================
# Phase 3 — Strong-Wolfe line search (bracket + zoom)
# ============================================================================

def _line_search_wolfe(fun_grad, x0, d, f0, g0, alpha_max,
                       c1=1e-4, c2=0.9, max_iter=25):
    """Strong-Wolfe line search along x0 + alpha d, alpha in (0, alpha_max].

    Returns (alpha, f_new, g_new, nfev).  Returns (None, None, None, nfev)
    if the search fails outright (e.g. non-descent direction).

    Follows Algorithms 3.5 and 3.6 of Nocedal & Wright, "Numerical
    Optimization", 2nd ed.
    """
    nfev = 0
    phi0  = float(f0)
    dphi0 = float(g0 @ d)
    if not (dphi0 < 0):                  # not a descent direction
        return None, None, None, 0

    # If alpha_max is huge, cap the initial alpha at 1 (the natural choice
    # for a quasi-Newton step), but never exceed alpha_max.
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

        # Armijo violation, or non-monotone bracket
        if (phi > phi0 + c1 * alpha * dphi0) or (i > 0 and phi >= phi_prev):
            a, fv, gv, n2 = _zoom(alpha_prev, alpha, phi_prev, phi,
                                  dphi_prev, fun_grad, x0, d,
                                  phi0, dphi0, c1, c2,
                                  max_iter=max_iter)
            return a, fv, gv, nfev + n2

        if abs(dphi) <= -c2 * dphi0:     # strong Wolfe satisfied
            return alpha, f_new, g_new, nfev

        if dphi >= 0:                    # gradient flipped: bracket between alpha and alpha_prev
            a, fv, gv, n2 = _zoom(alpha, alpha_prev, phi, phi_prev,
                                  dphi, fun_grad, x0, d,
                                  phi0, dphi0, c1, c2,
                                  max_iter=max_iter)
            return a, fv, gv, nfev + n2

        # Expand the step.
        alpha_prev = alpha
        phi_prev   = phi
        dphi_prev  = dphi
        if alpha >= 0.99 * alpha_max:
            return alpha, f_new, g_new, nfev
        alpha = min(2.0 * alpha, alpha_max)

    # max_iter without explicit termination: accept what we have.
    return alpha, f_new, g_new, nfev


def _zoom(a_lo, a_hi, phi_lo, phi_hi, dphi_lo, fun_grad, x0, d,
          phi0, dphi0, c1, c2, max_iter=25):
    """The 'zoom' phase, Algorithm 3.6 of Nocedal & Wright."""
    nfev = 0
    f_new = None
    g_new = None
    for _ in range(max_iter):
        # Safeguarded bisection (could be cubic interpolation; bisection
        # is sufficient and bulletproof for this code path).
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

class LBFGSBResult(dict):
    """Result of an L-BFGS-B run.  Mirrors a tiny subset of scipy's
    OptimizeResult."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.__dict__ = self


def minimize_lbfgsb(fun_and_grad, x0, bounds=None, m=10, factr=1e7,
                    pgtol=1e-5, maxiter=15000, max_ls=25, callback=None):
    """Minimize f(x) subject to l <= x <= u using L-BFGS-B.

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
    m : int
        Maximum number of (s, y) correction pairs to keep.
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
    LBFGSBResult with attributes
        x, f, g, status, message, nit, nfev, pg_norm.
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

    # Project x0 into the feasible set.
    x = _project(x, l, u, nbd)

    f, g = fun_and_grad(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    nfev = 1

    state = _LBFGSState(n, m)
    f_prev = f
    pg_norm = _proj_grad_inf_norm(x, g, l, u, nbd)

    if pg_norm <= pgtol:
        return LBFGSBResult(x=x, f=f, g=g, status=0,
                            message="CONVERGENCE: projected gradient below pgtol",
                            nit=0, nfev=nfev, pg_norm=pg_norm)

    status  = -1
    message = "maxiter reached"

    for k in range(maxiter):
        # --- Phase 1: Cauchy point ---
        xcp, c_vec, free_mask = _cauchy_point(x, g, l, u, nbd, state)

        # --- Phase 2: Subspace minimization ---
        if free_mask.any():
            z = _subspace_minimize(x, g, xcp, c_vec, l, u, nbd, free_mask, state)
        else:
            z = xcp

        # --- Phase 3: Line search along d = z - x ---
        d  = z - x
        gd = float(g @ d)
        if gd >= -_EPS * max(abs(f), 1.0):
            # Not a descent direction: refresh the L-BFGS memory and try
            # a plain projected-gradient step (d = -g, capped at the box).
            state.reset()
            d = -g
            gd = float(g @ d)
            if gd >= 0:
                status, message = 3, "abnormal: gradient is zero but pg_norm > pgtol"
                break

        alpha_max = _max_feasible_step(x, d, l, u, nbd)
        if alpha_max <= 0.0:
            status, message = 3, "abnormal: no feasible step along d"
            break

        # Make sure alpha=1 is within the feasibility cap when z is feasible.
        # If z is feasible (it should be after _subspace_minimize), alpha_max>=1.
        alpha, f_new, g_new, ls_nfev = _line_search_wolfe(
            fun_and_grad, x, d, f, g, alpha_max,
            c1=1e-4, c2=0.9, max_iter=max_ls)
        nfev += ls_nfev

        if alpha is None or f_new is None:
            # Line search failed entirely: refresh and try again next outer iteration.
            if state.col == 0:
                status, message = 3, "abnormal: line search failed with empty L-BFGS memory"
                break
            state.reset()
            continue

        f_new = float(f_new)
        g_new = np.asarray(g_new, dtype=np.float64)

        # Quasi-Newton update
        x_new = x + alpha * d
        # Re-project as safeguard against tiny boundary overshoots.
        x_new = _project(x_new, l, u, nbd)

        s = x_new - x
        y = g_new - g

        # Convergence tests (after the step).
        pg_norm = _proj_grad_inf_norm(x_new, g_new, l, u, nbd)
        rel_red = (f - f_new) / max(abs(f), abs(f_new), 1.0)

        # Commit the step.
        x, f, g = x_new, f_new, g_new

        # Try to add the new (s, y).  Silently skipped if curvature
        # condition or Cholesky guard fails (state remains valid).
        state.try_update(s, y)

        if callback is not None:
            callback(x)

        if pg_norm <= pgtol:
            status, message = 0, "CONVERGENCE: projected gradient below pgtol"
            break
        if rel_red <= ftol:
            status, message = 1, "CONVERGENCE: relative reduction of f below factr*eps"
            break

        f_prev = f

    if status == -1:
        status = 2

    return LBFGSBResult(x=x, f=f, g=g, status=status, message=message,
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
    import time

    np.random.seed(0)

    # ------------------------------------------------------------------
    # Test infrastructure
    # ------------------------------------------------------------------
    _failed = []

    def _check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            _failed.append(name)

    def _close(a, b, tol=1e-5):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) <= tol

    # ------------------------------------------------------------------
    # Test problems
    # ------------------------------------------------------------------
    def quadratic(A, b):
        """f(x) = 0.5 x^T A x - b^T x;  grad = A x - b."""
        def fg(x):
            Ax = A @ x
            f = 0.5 * float(x @ Ax) - float(b @ x)
            g = Ax - b
            return f, g
        return fg

    def rosenbrock(x):
        """Standard Rosenbrock (n-d)."""
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

    # ------------------------------------------------------------------
    # Test 1: 2D unconstrained quadratic
    # ------------------------------------------------------------------
    print("\n=== Test 1: 2D unconstrained quadratic ===")
    A = np.array([[3.0, 0.5], [0.5, 2.0]])
    b = np.array([1.0, 2.0])
    x_star = np.linalg.solve(A, b)
    fg = quadratic(A, b)
    res = minimize_lbfgsb(fg, np.zeros(2))
    _check("converged", res.status in (0, 1), f"status={res.status}, msg='{res.message}'")
    # Tolerance is loose because the algorithm stops at the pgtol level,
    # which corresponds to roughly sqrt(pgtol) precision in x for a quadratic.
    _check("x matches analytic minimum", _close(res.x, x_star, tol=1e-5),
           f"||x - x*|| = {np.linalg.norm(res.x - x_star):.2e}")
    _check("pg_norm small", res.pg_norm < 1e-4, f"pg_norm = {res.pg_norm:.2e}")
    _check("nfev modest", res.nfev < 100, f"nfev = {res.nfev}")

    # ------------------------------------------------------------------
    # Test 2: 2D quadratic with inactive bounds
    # ------------------------------------------------------------------
    print("\n=== Test 2: bounds present but not active at minimizer ===")
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    res = minimize_lbfgsb(fg, np.array([4.0, -4.0]), bounds=bounds)
    _check("converged", res.status in (0, 1))
    _check("x matches unconstrained minimum",
           _close(res.x, x_star, tol=1e-5),
           f"||x - x*|| = {np.linalg.norm(res.x - x_star):.2e}")

    # ------------------------------------------------------------------
    # Test 3: bounds active at minimizer
    # ------------------------------------------------------------------
    print("\n=== Test 3: bounds active at minimizer ===")
    # A is positive definite, unconstrained minimum at (0.286, 0.929).
    # Force x0 lower bound = 1.0 so x_0* hits its lower bound.
    bounds = [(1.0, None), (None, None)]
    res = minimize_lbfgsb(fg, np.array([5.0, 5.0]), bounds=bounds)
    _check("converged", res.status in (0, 1))
    _check("x[0] pinned at lower bound", abs(res.x[0] - 1.0) < 1e-6,
           f"x[0] = {res.x[0]:.6f}")
    # At the optimum, x[1] satisfies A[1,0]*1.0 + A[1,1]*x[1] - b[1] = 0.
    x1_star = (b[1] - A[1, 0] * 1.0) / A[1, 1]
    _check("x[1] satisfies stationarity",
           abs(res.x[1] - x1_star) < 1e-5,
           f"x[1] = {res.x[1]:.6f} (expected {x1_star:.6f})")

    # ------------------------------------------------------------------
    # Test 4: pure box constraint (corner solution)
    # ------------------------------------------------------------------
    print("\n=== Test 4: corner solution ===")
    # Linear objective f(x) = c^T x with box [0, 1]^n has its minimum at a
    # corner.  Use c = (-1, -2, -3) so x* = (1, 1, 1).
    n = 3
    c = np.array([-1.0, -2.0, -3.0])
    def linear_fg(x):
        return float(c @ x), c.copy()
    bounds = [(0.0, 1.0)] * n
    res = minimize_lbfgsb(linear_fg, np.array([0.5, 0.5, 0.5]), bounds=bounds)
    _check("converged", res.status in (0, 1))
    _check("x at corner (1,1,1)", _close(res.x, np.ones(n), tol=1e-6),
           f"x = {res.x}")

    # ------------------------------------------------------------------
    # Test 5: Rosenbrock (unconstrained)
    # ------------------------------------------------------------------
    print("\n=== Test 5: Rosenbrock, n=5, unconstrained ===")
    n = 5
    _check("Rosenbrock gradient matches FD",
           gradient_check(rosenbrock, np.array([-1.2, 1.0, 0.5, 0.0, 2.0])))
    res = minimize_lbfgsb(rosenbrock, -np.ones(n) * 1.2, pgtol=1e-8, maxiter=2000)
    _check("converged", res.status in (0, 1), f"status={res.status}")
    _check("at the global minimum (1,...,1)",
           _close(res.x, np.ones(n), tol=1e-3),
           f"||x-1|| = {np.linalg.norm(res.x - 1):.2e}, f = {res.f:.2e}")

    # ------------------------------------------------------------------
    # Test 6: Rosenbrock with active bounds
    # ------------------------------------------------------------------
    print("\n=== Test 6: Rosenbrock with bounds [0, 0.5]^n ===")
    n = 4
    bounds = [(0.0, 0.5)] * n
    res = minimize_lbfgsb(rosenbrock, np.full(n, 0.3), bounds=bounds,
                          pgtol=1e-10, factr=1.0, maxiter=3000)
    _check("converged", res.status in (0, 1))
    _check("all components feasible",
           bool(np.all(res.x >= -1e-9) and np.all(res.x <= 0.5 + 1e-9)),
           f"x = {res.x}")
    # The constrained minimizer isn't a corner; the chain of squared terms
    # makes the active set just {x_0 = 0.5}, with x_1..x_{n-1} interior.
    # Cross-check against scipy if available.
    try:
        from scipy.optimize import minimize as _sp_minimize
        _sp = _sp_minimize(lambda x: rosenbrock(x)[0], np.full(n, 0.3),
                           jac=lambda x: rosenbrock(x)[1],
                           method="L-BFGS-B", bounds=bounds,
                           options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 3000})
        _check("x agrees with scipy on bounded Rosenbrock",
               _close(res.x, _sp.x, tol=1e-6),
               f"||x_mine - x_sp|| = {np.linalg.norm(res.x - _sp.x):.2e}")
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # Test 7: 50-dim quadratic with mixed bounds
    # ------------------------------------------------------------------
    print("\n=== Test 7: 50-dim quadratic with mixed bounds ===")
    n = 50
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + n * np.eye(n)                   # SPD
    b_vec = rng.standard_normal(n)
    fg7 = quadratic(A, b_vec)
    x_unc = np.linalg.solve(A, b_vec)
    # Make every third variable bounded above by zero (some will be active).
    bounds = [(-np.inf, 0.0) if i % 3 == 0 else (None, None) for i in range(n)]
    res = minimize_lbfgsb(fg7, np.zeros(n), bounds=bounds, m=15,
                          pgtol=1e-6, factr=10.0, maxiter=2000)
    _check("converged", res.status in (0, 1), f"status={res.status}, msg='{res.message}'")
    _check("all feasible",
           bool(np.all(res.x[::3] <= 1e-8)),
           f"max violation = {float(np.max(res.x[::3])):.2e}")
    _check("pg_norm small", res.pg_norm < 1e-5,
           f"pg_norm = {res.pg_norm:.2e}")
    # Sanity: where the unconstrained solution is feasible, we should match it.
    # (Hard to check tightly; just verify objective is at least as good as the
    # one obtained by clamping x_unc to the box.)
    x_clamped = x_unc.copy()
    x_clamped[::3] = np.minimum(x_clamped[::3], 0.0)
    f_clamped, _ = fg7(x_clamped)
    _check("objective beats naive clamp",
           res.f <= f_clamped + 1e-8,
           f"f = {res.f:.6e}, f(clamp) = {f_clamped:.6e}")

    # ------------------------------------------------------------------
    # Test 8: cross-check against scipy.optimize if available
    # ------------------------------------------------------------------
    print("\n=== Test 8: cross-check against scipy.optimize.minimize ===")
    try:
        from scipy.optimize import minimize as sp_minimize
        # Quadratic
        A2 = np.array([[4.0, 1.0], [1.0, 3.0]])
        b2 = np.array([1.0, 2.0])
        fg8 = quadratic(A2, b2)
        bounds = [(0.5, 2.0), (-1.0, 2.0)]
        x0 = np.array([0.5, 0.5])
        res_mine = minimize_lbfgsb(fg8, x0, bounds=bounds, pgtol=1e-9)
        res_sp = sp_minimize(lambda x: fg8(x)[0], x0, jac=lambda x: fg8(x)[1],
                             method="L-BFGS-B", bounds=bounds,
                             options={"gtol": 1e-9, "ftol": 1e-12})
        _check("our x agrees with scipy on bounded quadratic",
               _close(res_mine.x, res_sp.x, tol=1e-5),
               f"||diff|| = {np.linalg.norm(res_mine.x - res_sp.x):.2e}")
        _check("our f agrees with scipy on bounded quadratic",
               abs(res_mine.f - res_sp.fun) < 1e-8,
               f"|df| = {abs(res_mine.f - res_sp.fun):.2e}")

        # Rosenbrock with bounds
        n = 6
        bounds = [(-1.5, 1.5)] * n
        x0 = -np.ones(n) * 1.2
        res_mine = minimize_lbfgsb(rosenbrock, x0, bounds=bounds,
                                   pgtol=1e-8, maxiter=2000)
        res_sp = sp_minimize(lambda x: rosenbrock(x)[0], x0,
                             jac=lambda x: rosenbrock(x)[1],
                             method="L-BFGS-B", bounds=bounds,
                             options={"gtol": 1e-8, "ftol": 1e-12})
        _check("our x agrees with scipy on bounded Rosenbrock",
               _close(res_mine.x, res_sp.x, tol=1e-3),
               f"||diff|| = {np.linalg.norm(res_mine.x - res_sp.x):.2e}, "
               f"f_mine={res_mine.f:.4e}, f_sp={res_sp.fun:.4e}")

    except ImportError:
        print("  (scipy not available; skipping comparison)")

    # ------------------------------------------------------------------
    # Test 9: finite-difference gradient wrapper
    # ------------------------------------------------------------------
    print("\n=== Test 9: FD-gradient wrapper on Rosenbrock 3D ===")
    rosen_f = lambda x: rosenbrock(x)[0]
    fg_fd = _wrap_with_fd_gradient(rosen_f, eps=1e-7)
    res = minimize_lbfgsb(fg_fd, np.array([-1.2, 1.0, 0.5]), pgtol=1e-5, maxiter=500)
    _check("FD-wrapped Rosenbrock converges near (1,1,1)",
           _close(res.x, np.ones(3), tol=1e-2),
           f"x = {res.x}, f = {res.f:.2e}")

    # ------------------------------------------------------------------
    # Test 10: KKT residual at convergence for several problems
    # ------------------------------------------------------------------
    print("\n=== Test 10: KKT residual sanity ===")
    for label, fg_, x0, bnds in [
        ("quadratic-2d",      fg,         np.array([0.0, 0.0]),  None),
        ("rosenbrock-5d",     rosenbrock, -np.ones(5) * 1.2,     None),
        ("rosenbrock-4d-box", rosenbrock, np.full(4, 0.3),       [(0.0, 0.5)] * 4),
    ]:
        res = minimize_lbfgsb(fg_, x0, bounds=bnds, pgtol=1e-8,
                              factr=1.0, maxiter=5000)
        # Algorithm satisfies pgtol when factr doesn't trigger first.
        ok = res.pg_norm <= 1e-5
        _check(f"  KKT ok for {label}", ok,
               f"pg_norm={res.pg_norm:.2e}, status={res.status}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if _failed:
        print(f"FAILED: {len(_failed)} test(s)")
        for name in _failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)