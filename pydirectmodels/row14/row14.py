"""
row14 — Rowshandel (2013) NGA-West2 rupture-directivity model
=============================================================

Implements Rowshandel's directivity model, Chapter 3 of PEER Report 2013/09
(Spudich, ed.), "Rowshandel's NGA-West2 Directivity Model", Badie Rowshandel.
Named ``row14`` per user request.

Directivity term (Eq. 3.14):

    fD = C1(T) * (xi - xi_c) * DT(R, T) * WP(M, T)

with (Eqs. 3.1-3.12):

* xi   = xi' * LD                         de-normalized directivity parameter
* xi'  = a*xi_s' + (1-a)*xi_p'  (Eq 3.5)  rupture/slip directivity predictor
    xi_p' = mean(p . q)         (Eq 3.2)  rupture-direction predictor
    xi_s' = mean(s . q)         (Eq 3.4)  slip-direction predictor
  where p = unit vector hypocentre -> sub-fault, q = unit vector sub-fault ->
  site, s = slip unit vector; sums are over N sub-faults (1 km x 1 km).
  Model-I sums only positive contributions (range 0..1); Model-II sums all
  (range -1..1) then maps xi' -> 0.5*(xi'+1).
* LD   = ln(Lrup)/ln(Lrup_max),  Lrup = sqrt(Ls^2 + Wrup^2),  Lrup_max=400  (3.7-3.8)
* DT   period-dependent distance taper (Eq. 3.9)
* WP   = exp(-[log10(T/Tp)]^2 / (2 Sig^2)),  Tp = exp(1.27 M - 7.28), Sig=0.6  (3.11-3.12)
* xi_c de-normalized centering term (Fig. 3.9): 0.21 SS, 0.15 RV/NM, 0.18 all
* C1(T) GMPE-specific, Model-I, Table 3.2

NOTES / interpretations (documented honestly; see README of the port):
* Eq. 3.3b as printed reads cos(theta) in the first term; for consistency with
  Eq. 3.3a and the Aki & Richards slip vector this must be sin(theta). The
  corrected form is used. For pure strike-slip xi_s' = xi_p' (per the report),
  so the slip-vector details do not affect strike-slip results.
* Ls (Eq. 3.8) is taken as the forward horizontal projection of the rupture
  onto the epicentre->site direction (the report's Figure 3.2 construction).
* Coordinates are a local planar frame: x=East, y=North, z=Depth (down), km.
"""

from dataclasses import dataclass, field

import numpy as np

from ..base import DirectivityModel, DirectivityResult, parse_imt

__all__ = ["Rowshandel2014Model", "Row14Rupture", "C1_TABLE",
           "XI_C", "LRUP_MAX", "SIG"]

# --- constants -------------------------------------------------------------
LRUP_MAX = 400.0          # effective rupture length for ~Mmax 8.5 SS (Eq 3.7)
SIG = 0.6                 # narrow-band bandwidth (recommended), Eq 3.12
TP_A, TP_B = 1.27, -7.28  # Tp = exp(1.27 M - 7.28), Eq 3.11

# de-normalized centering term xi_c (Fig. 3.9 averages, Model-I)
XI_C = {"SS": 0.21, "RV": 0.15, "NM": 0.15, "ALL": 0.18}

# C1(T) period-dependent directivity coefficient, Model-I, Table 3.2,
# one column per host GMPE. T = [1,2,3,4,5,7.5,10] s. (C&Y has no T=10 entry.)
_C1_PERIODS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])
C1_TABLE = {
    "A&S": np.array([0.375, 0.561, 0.833, 0.873, 1.019, 1.387, 1.551]),
    "B&A": np.array([0.257, 0.699, 0.824, 0.834, 0.986, 1.378, 1.588]),
    "C&B": np.array([0.149, 0.377, 0.570, 0.596, 0.772, 1.152, 1.369]),
    "C&Y": np.array([0.079, 0.321, 0.629, 0.693, 0.898, 1.258, np.nan]),
    "Id":  np.array([0.325, 0.803, 1.544, 1.576, 1.823, 2.089, 2.250]),
}


def get_c1(period, gmpe="B&A"):
    """Interpolate C1(T) (linear in log period) for the chosen host GMPE."""
    vals = C1_TABLE[gmpe]
    ok = np.isfinite(vals)
    per, v = _C1_PERIODS[ok], vals[ok]
    if period <= per[0]:
        return float(v[0])
    if period >= per[-1]:
        return float(v[-1])
    return float(np.interp(np.log10(period), np.log10(per), v))


def get_mechanism(rake):
    """Style of faulting from rake (report Section 3.4 convention)."""
    if 45.0 < rake < 135.0:
        return "RV"
    if -135.0 < rake < -45.0:
        return "NM"
    return "SS"


def _wp(period, mag):
    """Narrow-band multiplier WP (Eqs. 3.11-3.12)."""
    tp = np.exp(TP_A * mag + TP_B)
    return float(np.exp(-(np.log10(period / tp) ** 2) / (2.0 * SIG * SIG)))


def _dt(rcls, period):
    """Period-dependent distance taper DT (Eq. 3.9)."""
    base = 20.0 + 10.0 * np.log(period) if period >= 1.0 else 20.0
    return np.clip(2.0 - np.asarray(rcls, float) / base, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Rupture geometry
# ---------------------------------------------------------------------------
@dataclass
class Row14Rupture:
    """Planar rupture for Row14, in a local km frame (x=E, y=N, z=Depth down).

    corner: (x, y, z) of the top corner A (z = depth to top of rupture, km).
    length, width: along-strike and down-dip extents (km).
    strike, dip, rake: degrees (strike clockwise from North).
    hypo_along: along-strike distance of the hypocentre from corner A (km).
    hypo_down:  down-dip distance of the hypocentre from the top edge (km).
    ds: sub-fault size (km) for the directivity integral (report uses 1 km).
    """
    corner: tuple
    length: float
    width: float
    strike: float
    dip: float
    rake: float
    hypo_along: float
    hypo_down: float
    ds: float = 1.0

    # -- unit vectors (x=E, y=N, z=Depth down) --
    def _u_strike(self):
        t = np.radians(self.strike)
        return np.array([np.sin(t), np.cos(t), 0.0])

    def _u_dip(self):
        t, d = np.radians(self.strike), np.radians(self.dip)
        return np.array([np.cos(d) * np.cos(t), -np.cos(d) * np.sin(t), np.sin(d)])

    def hypocentre(self):
        A = np.asarray(self.corner, float)
        return A + self.hypo_along * self._u_strike() + self.hypo_down * self._u_dip()

    def subfaults(self):
        """Centres of the 1km x 1km (ds) sub-faults: array (N, 3)."""
        A = np.asarray(self.corner, float)
        us, ud = self._u_strike(), self._u_dip()
        nl = max(int(round(self.length / self.ds)), 1)
        nw = max(int(round(self.width / self.ds)), 1)
        ls = (np.arange(nl) + 0.5) * (self.length / nl)
        ws = (np.arange(nw) + 0.5) * (self.width / nw)
        L, W = np.meshgrid(ls, ws)
        L, W = L.ravel(), W.ravel()
        return A[None, :] + L[:, None] * us[None, :] + W[:, None] * ud[None, :], ls

    def trace_endpoints(self):
        A = np.asarray(self.corner, float)
        us = self._u_strike()
        return A[:2], (A + self.length * us)[:2]


def _slip_vector(rupture, along):
    """Per-sub-fault slip unit vectors s (Eq. 3.3, corrected; see module note).

    Strike-parallel part aligned with the local rupture direction (sign of
    along-strike offset from the hypocentre) so that, for pure strike-slip,
    xi_s' = xi_p'. Dip-parallel part points up-dip, weighted by sin|rake|.
    """
    g = np.radians(rupture.rake)
    us, ud = rupture._u_strike(), rupture._u_dip()
    sgn = np.sign(along - rupture.hypo_along)
    sgn[sgn == 0] = 1.0
    # cos|rake| along strike (rupture sense), sin|rake| up-dip (-u_dip)
    s = (np.cos(np.abs(g)) * sgn[:, None] * us[None, :]
         + np.sin(np.abs(g)) * (-ud)[None, :])
    return s / np.linalg.norm(s, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Rowshandel2014Model(DirectivityModel):
    """Rowshandel (2013) NGA-West2 directivity model (named row14)."""

    @property
    def name(self) -> str:
        return "Rowshandel (2013) NGA-West2 directivity [row14]"

    @property
    def period_limits(self):
        return (1.0, 10.0)         # C1 is tabulated 1-10 s (Table 3.2)

    @property
    def input_params(self) -> tuple:
        return ("mag", "site_x", "site_y", "rupture", "imt")

    def evaluate(self, mag, site_x, site_y, rupture, imt,
                 model=1, a=0.5, gmpe="B&A", mech=None) -> DirectivityResult:
        """Directivity adjustment ``fd`` (ln units) at the sites.

        Shared-convention signature: ``mag`` first, geometry in the middle,
        ``imt`` last. ``site_x``/``site_y`` are local km coordinates (E, N);
        ``rupture`` is a :class:`Row14Rupture`; ``imt`` a numeric SA period or
        "SA(T)". ``model`` 1 or 2, ``a`` the slip/rupture weight (Eq 3.5),
        ``gmpe`` selects the C1 column, ``mech`` overrides the style of faulting.
        """
        kind, period = parse_imt(imt)
        sx = np.atleast_1d(np.asarray(site_x, float)).ravel()
        sy = np.atleast_1d(np.asarray(site_y, float)).ravel()
        if kind != "SA":
            return DirectivityResult(fd=np.zeros_like(sx), model=self.name,
                                     extra={"imt": imt, "note": f"{kind}: no row14 directivity"})

        subs, ls = rupture.subfaults()                 # (N,3), along-strike centres
        N = subs.shape[0]
        nw = max(int(round(rupture.width / rupture.ds)), 1)
        along = np.tile(ls, nw)                         # matches subfault ravel order
        hypo = rupture.hypocentre()

        # rupture unit vectors p (hypocentre -> sub-fault), fixed across sites
        p = subs - hypo[None, :]
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        s = _slip_vector(rupture, along)               # slip unit vectors (N,3)

        e1, e2 = rupture.trace_endpoints()
        E = rupture.hypocentre()[:2]                   # epicentre (horizontal)
        Wrup = max(rupture.hypo_down, 0.5)

        c1 = get_c1(period, gmpe)
        wp = _wp(period, mag)
        xic = XI_C[mech if mech else get_mechanism(rupture.rake)]

        fd = np.zeros(sx.shape)
        xi_store = np.zeros(sx.shape)
        for k in range(sx.size):
            site = np.array([sx[k], sy[k], 0.0])
            q = site[None, :] - subs
            qn = q / np.linalg.norm(q, axis=1, keepdims=True)
            pq = np.einsum("ij,ij->i", p, qn)
            sq = np.einsum("ij,ij->i", s, qn)
            if model == 1:                              # positive contributions
                xip = np.mean(np.clip(pq, 0.0, None))
                xis = np.mean(np.clip(sq, 0.0, None))
                xipr = a * xis + (1.0 - a) * xip
            else:                                       # Model-II: all, then remap
                xip = np.mean(pq); xis = np.mean(sq)
                xipr = 0.5 * ((a * xis + (1.0 - a) * xip) + 1.0)

            # Ls: forward horizontal projection of rupture onto epicentre->site
            d = site[:2] - E
            nd = np.hypot(*d)
            ehat = d / nd if nd > 1e-9 else np.array([1.0, 0.0])
            Ls = max(np.dot(e1 - E, ehat), np.dot(e2 - E, ehat), 0.5)
            Lrup = np.hypot(Ls, Wrup)
            LD = np.log(max(Lrup, 1.0001)) / np.log(LRUP_MAX)

            rcls = np.min(np.linalg.norm(site[None, :] - subs, axis=1))
            dt = float(_dt(rcls, period))

            xi = xipr * LD
            xi_store[k] = xi
            fd[k] = c1 * (xi - xic) * dt * wp

        return DirectivityResult(
            fd=fd, model=self.name,
            extra={"imt": imt, "period": period, "xi": xi_store,
                   "C1": c1, "WP": wp, "xi_c": xic, "gmpe": gmpe,
                   "model": model, "a": a, "N_subfaults": N})
