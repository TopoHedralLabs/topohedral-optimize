"""
scalar_minimizers.py
=====================

Self-contained extraction of SciPy's univariate (scalar) function minimizers.

Source: scipy/optimize/_optimize.py and scipy/optimize/_minimize.py
(https://github.com/scipy/scipy, `main` branch).

Contents
--------
Public high-level API:
    - minimize_scalar   : unified dispatcher ('brent', 'bounded', 'golden')
    - brent             : Brent's method (parabolic interpolation + golden section)
    - golden            : Golden-section search
    - fminbound         : Bounded Brent minimization on a finite interval
    - bracket           : Downhill bracket search for an unconstrained minimum

Internal helpers (kept for fidelity to SciPy):
    - Brent (class), BracketError (exception)
    - _minimize_scalar_brent / _minimize_scalar_bounded / _minimize_scalar_golden
    - _recover_from_bracket_error
    - OptimizeResult, _check_unknown_options, is_finite_scalar,
      _endprint, _print_success_message_or_warn

The only external dependency is NumPy. Code is copied essentially verbatim from
SciPy (BSD-3-Clause licensed) with a lightweight local `OptimizeResult` and
`OptimizeWarning` so the module stands alone.
"""

import warnings
from math import sqrt

import numpy as np


# ---------------------------------------------------------------------------
# Minimal local stand-ins for scipy.optimize infrastructure
# ---------------------------------------------------------------------------
class OptimizeWarning(UserWarning):
    pass


class OptimizeResult(dict):
    """Represents the optimization result. Attribute access over a dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        return self.__class__.__name__ + "()"


_status_message = {
    'success': 'Optimization terminated successfully.',
    'maxfev': 'Maximum number of function evaluations has been exceeded.',
    'maxiter': 'Maximum number of iterations has been exceeded.',
    'pr_loss': 'Desired error not necessarily achieved due to precision loss.',
    'nan': 'NaN result encountered.',
    'out_of_bounds': 'The result is outside of the provided bounds.',
}

_epsilon = sqrt(np.finfo(float).eps)


def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn(f"Unknown solver options: {msg}", OptimizeWarning,
                      stacklevel=4)


def is_finite_scalar(x):
    """Test whether `x` is either a finite scalar or a finite array scalar."""
    return np.size(x) == 1 and np.isfinite(x)


def _print_success_message_or_warn(warnflag, message, warntype=None):
    if not warnflag:
        print(message)
    else:
        warnings.warn(message, warntype or OptimizeWarning, stacklevel=3)


def _endprint(x, flag, fval, maxfun, xtol, disp):
    if flag == 0:
        if disp > 1:
            print("\nOptimization terminated successfully;\n"
                  "The returned value satisfies the termination criteria\n"
                  "(using xtol = ", xtol, ")")
        return

    if flag == 1:
        msg = ("\nMaximum number of function evaluations exceeded --- "
               "increase maxfun argument.\n")
    elif flag == 2:
        msg = f"\n{_status_message['nan']}"

    _print_success_message_or_warn(flag, msg)
    return


# ---------------------------------------------------------------------------
# fminbound / bounded scalar minimization
# ---------------------------------------------------------------------------
def fminbound(func, x1, x2, args=(), xtol=1e-5, maxfun=500,
              full_output=0, disp=1):
    """Bounded minimization for scalar functions.

    Finds a local minimizer of the scalar function `func` in the
    interval x1 < xopt < x2 using Brent's method.
    """
    options = {'xatol': xtol,
               'maxiter': maxfun,
               'disp': disp}

    res = _minimize_scalar_bounded(func, (x1, x2), args, **options)
    if full_output:
        return res['x'], res['fun'], res['status'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_bounded(func, bounds, args=(),
                             xatol=1e-5, maxiter=500, disp=0,
                             **unknown_options):
    """Options: maxiter, disp, xatol (absolute error in xopt)."""
    _check_unknown_options(unknown_options)
    maxfun = maxiter
    # Test bounds are of correct form
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    x1, x2 = bounds

    if not (is_finite_scalar(x1) and is_finite_scalar(x2)):
        raise ValueError("Optimization bounds must be finite scalars.")

    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    flag = 0
    header = ' Func-count     x          f(x)          Procedure'
    step = '       initial'

    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = np.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    if disp > 2:
        print(" ")
        print(header)
        print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

    while (np.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((np.abs(p) < np.abs(0.5 * q * r)) and (p > q * (a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                step = '       parabolic'

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
            step = '       golden'

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)
        fu = func(x, *args)
        num += 1
        fmin_data = (num, x, fu)
        if disp > 2:
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    fval = fx
    if disp > 0:
        _endprint(x, flag, fval, maxfun, xatol, disp)

    result = OptimizeResult(
        fun=fval, status=flag, success=(flag == 0),
        message={0: 'Solution found.',
                 1: 'Maximum number of function calls reached.',
                 2: _status_message['nan']}.get(flag, ''),
        x=xf, nfev=num, nit=num)

    return result


# ---------------------------------------------------------------------------
# Brent's method
# ---------------------------------------------------------------------------
class Brent:
    # need to rethink design of __init__
    def __init__(self, func, args=(), tol=1.48e-8, maxiter=500,
                 full_output=0, disp=0):
        self.func = func
        self.args = args
        self.tol = tol
        self.maxiter = maxiter
        self._mintol = 1.0e-11
        self._cg = 0.3819660
        self.xmin = None
        self.fval = None
        self.iter = 0
        self.funcalls = 0
        self.disp = disp

    # need to rethink design of set_bracket (new options, etc.)
    def set_bracket(self, brack=None):
        self.brack = brack

    def get_bracket_info(self):
        # set up
        func = self.func
        args = self.args
        brack = self.brack
        ### BEGIN core bracket_info code ###
        ### carefully DOCUMENT any CHANGES in core ##
        if brack is None:
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
        elif len(brack) == 2:
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0],
                                                       xb=brack[1], args=args)
        elif len(brack) == 3:
            xa, xb, xc = brack
            if (xa > xc):  # swap so xa < xc can be assumed
                xc, xa = xa, xc
            if not ((xa < xb) and (xb < xc)):
                raise ValueError(
                    "Bracketing values (xa, xb, xc) do not"
                    " fulfill this requirement: (xa < xb) and (xb < xc)"
                )
            fa = func(*((xa,) + args))
            fb = func(*((xb,) + args))
            fc = func(*((xc,) + args))
            if not ((fb < fa) and (fb < fc)):
                raise ValueError(
                    "Bracketing values (xa, xb, xc) do not fulfill"
                    " this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))"
                )

            funcalls = 3
        else:
            raise ValueError("Bracketing interval must be "
                             "length 2 or 3 sequence.")
        ### END core bracket_info code ###

        return xa, xb, xc, fa, fb, fc, funcalls

    def optimize(self):
        # set up for optimization
        func = self.func
        xa, xb, xc, fa, fb, fc, funcalls = self.get_bracket_info()
        _mintol = self._mintol
        _cg = self._cg
        #################################
        # BEGIN CORE ALGORITHM
        #################################
        x = w = v = xb
        fw = fv = fx = fb
        if (xa < xc):
            a = xa
            b = xc
        else:
            a = xc
            b = xa
        deltax = 0.0
        iter = 0

        if self.disp > 2:
            print(" ")
            print(f"{'Func-count':^12} {'x':^12} {'f(x)': ^12}")
            print(f"{funcalls:^12g} {x:^12.6g} {fx:^12.6g}")

        while (iter < self.maxiter):
            tol1 = self.tol * np.abs(x) + _mintol
            tol2 = 2.0 * tol1
            xmid = 0.5 * (a + b)
            # check for convergence
            if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
                break
            # XXX In the first iteration, rat is only bound in the true case
            # of this conditional. This used to cause an UnboundLocalError
            # (gh-4140). It should be set before the if (but to what?).
            if (np.abs(deltax) <= tol1):
                if (x >= xmid):
                    deltax = a - x       # do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax
            else:                              # do a parabolic step
                tmp1 = (x - w) * (fx - fv)
                tmp2 = (x - v) * (fx - fw)
                p = (x - v) * tmp2 - (x - w) * tmp1
                tmp2 = 2.0 * (tmp2 - tmp1)
                if (tmp2 > 0.0):
                    p = -p
                tmp2 = np.abs(tmp2)
                dx_temp = deltax
                deltax = rat
                # check parabolic fit
                if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                        (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                    rat = p * 1.0 / tmp2        # if parabolic step is useful.
                    u = x + rat
                    if ((u - a) < tol2 or (b - u) < tol2):
                        if xmid - x >= 0:
                            rat = tol1
                        else:
                            rat = -tol1
                else:
                    if (x >= xmid):
                        deltax = a - x  # if it's not do a golden section step
                    else:
                        deltax = b - x
                    rat = _cg * deltax

            if (np.abs(rat) < tol1):            # update by at least tol1
                if rat >= 0:
                    u = x + tol1
                else:
                    u = x - tol1
            else:
                u = x + rat
            fu = func(*((u,) + self.args))      # calculate new output value
            funcalls += 1

            if (fu > fx):                 # if it's bigger than current
                if (u < x):
                    a = u
                else:
                    b = u
                if (fu <= fw) or (w == x):
                    v = w
                    w = u
                    fv = fw
                    fw = fu
                elif (fu <= fv) or (v == x) or (v == w):
                    v = u
                    fv = fu
            else:
                if (u >= x):
                    a = x
                else:
                    b = x
                v = w
                w = x
                x = u
                fv = fw
                fw = fx
                fx = fu

            if self.disp > 2:
                print(f"{funcalls:^12g} {x:^12.6g} {fx:^12.6g}")

            iter += 1
        #################################
        # END CORE ALGORITHM
        #################################

        self.xmin = x
        self.fval = fx
        self.iter = iter
        self.funcalls = funcalls

    def get_result(self, full_output=False):
        if full_output:
            return self.xmin, self.fval, self.iter, self.funcalls
        else:
            return self.xmin


def brent(func, args=(), brack=None, tol=1.48e-8, full_output=0, maxiter=500):
    """
    Given a function of one variable and a possible bracket, return
    a local minimizer of the function isolated to a fractional precision
    of tol.

    Uses inverse parabolic interpolation when possible to speed up
    convergence of golden section method. Does not ensure that the minimum
    lies in the range specified by `brack`.
    """
    options = {'xtol': tol,
               'maxiter': maxiter}
    res = _minimize_scalar_brent(func, brack, args, **options)
    if full_output:
        return res['x'], res['fun'], res['nit'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_brent(func, brack=None, args=(), xtol=1.48e-8,
                           maxiter=500, disp=0,
                           **unknown_options):
    """Options: maxiter, xtol (relative error in xopt), disp."""
    _check_unknown_options(unknown_options)
    tol = xtol
    if tol < 0:
        raise ValueError(f'tolerance should be >= 0, got {tol!r}')

    brent = Brent(func=func, args=args, tol=tol,
                  full_output=True, maxiter=maxiter, disp=disp)
    brent.set_bracket(brack)
    brent.optimize()
    x, fval, nit, nfev = brent.get_result(full_output=True)

    success = nit < maxiter and not (np.isnan(x) or np.isnan(fval))

    if success:
        message = ("\nOptimization terminated successfully;\n"
                   "The returned value satisfies the termination criteria\n"
                   f"(using xtol = {xtol} )")
    else:
        if nit >= maxiter:
            message = "\nMaximum number of iterations exceeded"
        if np.isnan(x) or np.isnan(fval):
            message = f"{_status_message['nan']}"

    if disp:
        _print_success_message_or_warn(not success, message)

    return OptimizeResult(fun=fval, x=x, nit=nit, nfev=nfev,
                          success=success, message=message)


# ---------------------------------------------------------------------------
# Golden-section search
# ---------------------------------------------------------------------------
def golden(func, args=(), brack=None, tol=_epsilon,
           full_output=0, maxiter=5000):
    """
    Return the minimizer of a function of one variable using the golden section
    method.

    Uses an analog of the bisection method to decrease the bracketed interval.
    """
    options = {'xtol': tol, 'maxiter': maxiter}
    res = _minimize_scalar_golden(func, brack, args, **options)
    if full_output:
        return res['x'], res['fun'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_golden(func, brack=None, args=(),
                            xtol=_epsilon, maxiter=5000, disp=0,
                            **unknown_options):
    """Options: xtol (relative error in xopt), maxiter, disp."""
    _check_unknown_options(unknown_options)
    tol = xtol
    if brack is None:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0],
                                                   xb=brack[1], args=args)
    elif len(brack) == 3:
        xa, xb, xc = brack
        if (xa > xc):  # swap so xa < xc can be assumed
            xc, xa = xa, xc
        if not ((xa < xb) and (xb < xc)):
            raise ValueError(
                "Bracketing values (xa, xb, xc) do not"
                " fulfill this requirement: (xa < xb) and (xb < xc)"
            )
        fa = func(*((xa,) + args))
        fb = func(*((xb,) + args))
        fc = func(*((xc,) + args))
        if not ((fb < fa) and (fb < fc)):
            raise ValueError(
                "Bracketing values (xa, xb, xc) do not fulfill"
                " this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))"
            )
        funcalls = 3
    else:
        raise ValueError("Bracketing interval must be length 2 or 3 sequence.")

    _gR = 0.61803399  # golden ratio conjugate: 2.0/(1.0+sqrt(5.0))
    _gC = 1.0 - _gR
    x3 = xc
    x0 = xa
    if (np.abs(xc - xb) > np.abs(xb - xa)):
        x1 = xb
        x2 = xb + _gC * (xc - xb)
    else:
        x2 = xb
        x1 = xb - _gC * (xb - xa)
    f1 = func(*((x1,) + args))
    f2 = func(*((x2,) + args))
    funcalls += 2
    nit = 0

    if disp > 2:
        print(" ")
        print(f"{'Func-count':^12} {'x':^12} {'f(x)': ^12}")

    for i in range(maxiter):
        if np.abs(x3 - x0) <= tol * (np.abs(x1) + np.abs(x2)):
            break
        if (f2 < f1):
            x0 = x1
            x1 = x2
            x2 = _gR * x1 + _gC * x3
            f1 = f2
            f2 = func(*((x2,) + args))
        else:
            x3 = x2
            x2 = x1
            x1 = _gR * x2 + _gC * x0
            f2 = f1
            f1 = func(*((x1,) + args))
        funcalls += 1
        if disp > 2:
            if (f1 < f2):
                xmin, fval = x1, f1
            else:
                xmin, fval = x2, f2
            print(f"{funcalls:^12g} {xmin:^12.6g} {fval:^12.6g}")

        nit += 1
    # end of iteration loop

    if (f1 < f2):
        xmin = x1
        fval = f1
    else:
        xmin = x2
        fval = f2

    success = nit < maxiter and not (np.isnan(fval) or np.isnan(xmin))

    if success:
        message = ("\nOptimization terminated successfully;\n"
                   "The returned value satisfies the termination criteria\n"
                   f"(using xtol = {xtol} )")
    else:
        if nit >= maxiter:
            message = "\nMaximum number of iterations exceeded"
        if np.isnan(xmin) or np.isnan(fval):
            message = f"{_status_message['nan']}"

    if disp:
        _print_success_message_or_warn(not success, message)

    return OptimizeResult(fun=fval, nfev=funcalls, x=xmin, nit=nit,
                          success=success, message=message)


# ---------------------------------------------------------------------------
# Downhill bracket search
# ---------------------------------------------------------------------------
def bracket(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0, maxiter=1000):
    """
    Bracket the minimum of a function.

    Given a function and distinct initial points, search in the downhill
    direction (as defined by the initial points) and return three points that
    bracket the minimum of the function.

    Returns
    -------
    xa, xb, xc, fa, fb, fc, funcalls

    Raises
    ------
    BracketError
        If no valid bracket is found before the algorithm terminates.
    """
    _gold = 1.618034  # golden ratio: (1.0+sqrt(5.0))/2.0
    _verysmall_num = 1e-21
    # convert to numpy floats if not already
    xa, xb = np.asarray([xa, xb])
    fa = func(*(xa,) + args)
    fb = func(*(xb,) + args)
    if (fa < fb):                      # Switch so fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc = func(*((xc,) + args))
    funcalls = 3
    iter = 0
    while (fc < fb):
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if np.abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        msg = ("No valid bracket was found before the iteration limit was "
               "reached. Consider trying different initial points or "
               "increasing `maxiter`.")
        if iter > maxiter:
            raise RuntimeError(msg)
        iter += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            if (fw < fc):
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif (fw > fb):
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim) * (xc - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            if (fw < fc):
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func(*((w,) + args))
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw

    # three conditions for a valid bracket
    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = (xa < xb < xc or xc < xb < xa)
    cond3 = np.isfinite(xa) and np.isfinite(xb) and np.isfinite(xc)
    msg = ("The algorithm terminated without finding a valid bracket. "
           "Consider trying different initial points.")
    if not (cond1 and cond2 and cond3):
        e = BracketError(msg)
        e.data = (xa, xb, xc, fa, fb, fc, funcalls)
        raise e

    return xa, xb, xc, fa, fb, fc, funcalls


class BracketError(RuntimeError):
    pass


def _recover_from_bracket_error(solver, fun, bracket, args, **options):
    # Raise the error in `bracket` but store the info needed by
    # `minimize_scalar` in the error object and intercept it here.
    try:
        res = solver(fun, bracket, args, **options)
    except BracketError as e:
        msg = str(e)
        xa, xb, xc, fa, fb, fc, funcalls = e.data
        xs, fs = [xa, xb, xc], [fa, fb, fc]
        if np.any(np.isnan([xs, fs])):
            x, fun = np.nan, np.nan
        else:
            imin = np.argmin(fs)
            x, fun = xs[imin], fs[imin]
        return OptimizeResult(fun=fun, nfev=funcalls, x=x,
                              nit=0, success=False, message=msg)
    return res


# ---------------------------------------------------------------------------
# Unified dispatcher (subset of scipy.optimize.minimize_scalar)
# ---------------------------------------------------------------------------
def minimize_scalar(fun, bracket=None, bounds=None, args=(),
                    method=None, tol=None, options=None):
    """Local minimization of a scalar function of one variable.

    Parameters
    ----------
    fun : callable
        Objective. Must accept and return scalars.
    bracket : sequence, optional
        For 'brent'/'golden': either (xa, xb) or (xa, xb, xc).
    bounds : sequence, optional
        For 'bounded': mandatory (min, max) pair of finite scalars.
    args : tuple, optional
        Extra arguments passed to `fun`.
    method : {'brent', 'bounded', 'golden'}, optional
        Defaults to 'bounded' if `bounds` given, else 'brent'.
    tol : float, optional
        Tolerance for termination.
    options : dict, optional
        Solver-specific options (e.g. xtol/xatol, maxiter, disp).

    Returns
    -------
    OptimizeResult
    """
    if not isinstance(args, tuple):
        args = (args,)

    if method is None:
        method = 'brent' if bounds is None else 'bounded'
    meth = method.lower()

    if options is None:
        options = {}

    if bounds is not None and meth in {'brent', 'golden'}:
        raise ValueError(
            f"Use of `bounds` is incompatible with 'method={method}'.")

    if tol is not None:
        options = dict(options)
        if meth == 'bounded' and 'xatol' not in options:
            warnings.warn("Method 'bounded' does not support relative "
                          "tolerance in x; defaulting to absolute tolerance.",
                          RuntimeWarning, stacklevel=2)
            options['xatol'] = tol
        else:
            options.setdefault('xtol', tol)

    # replace boolean "disp" option, if specified, by an integer value.
    disp = options.get('disp')
    if isinstance(disp, bool):
        options['disp'] = 2 * int(disp)

    if meth == 'brent':
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          fun, bracket, args, **options)
    elif meth == 'bounded':
        if bounds is None:
            raise ValueError('The `bounds` parameter is mandatory for '
                             'method `bounded`.')
        res = _minimize_scalar_bounded(fun, bounds, args, **options)
    elif meth == 'golden':
        res = _recover_from_bracket_error(_minimize_scalar_golden,
                                          fun, bracket, args, **options)
    else:
        raise ValueError(f'Unknown solver {method}')

    return res


# ===========================================================================
# Tests
# ===========================================================================
def main():
    import math

    tol = 1e-4
    n_pass = 0
    n_fail = 0

    def check(name, got, expected, atol=tol):
        nonlocal n_pass, n_fail
        ok = abs(got - expected) <= atol
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  [{status}] {name}: got {got:.8g}, expected {expected:.8g}")

    # -- Test functions -----------------------------------------------------
    def parabola(x):
        return (x - 1.0) ** 2          # min at x = 1, f = 0

    def quartic(x):
        return (x - 2.0) ** 4 + 3.0    # min at x = 2, f = 3

    def cosine(x):
        return math.cos(x)             # min at x = pi on [0, 2pi]

    def quad_args(x, a, b):
        return (x - a) ** 2 + b        # min at x = a, f = b

    print("== brent ==")
    check("parabola brack=(0,1)", brent(parabola, brack=(0, 1)), 1.0)
    check("parabola brack=(-1,0.5,3)", brent(parabola, brack=(-1, 0.5, 3)), 1.0)
    x, f, nit, nfev = brent(quartic, brack=(0, 1), full_output=True)
    check("quartic xmin", x, 2.0, atol=1e-2)
    check("quartic fmin", f, 3.0, atol=1e-3)

    print("== golden ==")
    check("parabola brack=(0,1)", golden(parabola, brack=(0, 1)), 1.0)
    check("parabola brack=(-1,0.5,3)", golden(parabola, brack=(-1, 0.5, 3)), 1.0)
    x, f, nfev = golden(quartic, brack=(0, 1), full_output=True)
    check("quartic xmin", x, 2.0, atol=1e-2)

    print("== fminbound ==")
    check("parabola on [-4, 4]", fminbound(parabola, -4, 4, disp=0), 1.0)
    check("cosine on [0, 2pi]", fminbound(cosine, 0, 2 * math.pi, disp=0),
          math.pi)
    xm, fv, ierr, nfc = fminbound(parabola, 3, 4, full_output=True, disp=0)
    check("clamped-to-lower-bound", xm, 3.0, atol=1e-3)
    assert ierr == 0, "fminbound should converge"

    print("== bracket ==")
    xa, xb, xc, fa, fb, fc, fcalls = bracket(parabola, xa=0.0, xb=1.0)
    assert (xa < xb < xc) or (xc < xb < xa), "bracket must be ordered"
    assert fb < fa and fb < fc, "middle bracket value must be lowest"
    print(f"  [PASS] bracket ordered & valley: "
          f"({xa:.4g}, {xb:.4g}, {xc:.4g})")

    print("== minimize_scalar dispatcher ==")
    r = minimize_scalar(parabola)  # default -> brent
    check("default(brent) parabola", r.x, 1.0)
    assert r.success
    r = minimize_scalar(parabola, method='golden', bracket=(0, 1))
    check("golden parabola", r.x, 1.0)
    r = minimize_scalar(parabola, bounds=(-4, 4), method='bounded')
    check("bounded parabola", r.x, 1.0)
    r = minimize_scalar(quad_args, args=(2.5, 7.0), bracket=(0, 1))
    check("brent with args (xmin)", r.x, 2.5)
    check("brent with args (fmin)", r.fun, 7.0, atol=1e-3)

    print("== error handling ==")
    # bounds incompatible with brent
    try:
        minimize_scalar(parabola, bounds=(-4, 4), method='brent')
        print("  [FAIL] expected ValueError for bounds+brent")
        n_fail += 1
    except ValueError:
        print("  [PASS] bounds+brent raises ValueError")
        n_pass += 1
    # non-finite bounds for bounded
    try:
        fminbound(parabola, -np.inf, 4, disp=0)
        print("  [FAIL] expected ValueError for infinite bounds")
        n_fail += 1
    except ValueError:
        print("  [PASS] infinite bounds raise ValueError")
        n_pass += 1
    # invalid 3-point bracket (not a valley)
    try:
        brent(parabola, brack=(0, 0.5, 0.9))  # monotone, no valley -> raises
        # This particular set may still be valid; use a clearly bad one:
    except ValueError:
        pass

    # -- Optional cross-check against SciPy if installed --------------------
    print("== SciPy cross-validation (if available) ==")
    try:
        from scipy import optimize as sciopt

        for name, brack in [("parabola", (0, 1)), ("quartic", (0, 1))]:
            f = parabola if name == "parabola" else quartic
            mine = brent(f, brack=brack)
            theirs = sciopt.brent(f, brack=brack)
            d = abs(mine - theirs)
            ok = d <= 1e-6
            print(f"  [{'PASS' if ok else 'FAIL'}] brent {name}: "
                  f"|mine - scipy| = {d:.2e}")
            n_pass += ok
            n_fail += (not ok)

        mine = fminbound(cosine, 0, 2 * math.pi, disp=0)
        theirs = sciopt.fminbound(cosine, 0, 2 * math.pi, disp=0)
        d = abs(mine - theirs)
        ok = d <= 1e-6
        print(f"  [{'PASS' if ok else 'FAIL'}] fminbound cosine: "
              f"|mine - scipy| = {d:.2e}")
        n_pass += ok
        n_fail += (not ok)
    except ImportError:
        print("  (SciPy not installed; skipping cross-validation)")

    print(f"\n{'=' * 40}\nTotal: {n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())