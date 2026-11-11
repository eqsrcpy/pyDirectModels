"""
bea25 — Bayless et al. (2025) directivity adjustment model
==========================================================

Concrete implementation of the abstract DirectivityModel interface defined in
base.py. The model is described in USGS External Grants Report G22AP00199.

Original MATLAB by Jeff Bayless (jeff.bayless@aecom.com), Feb 2024.
Copyright (c) 2024, Jeff Bayless, covered by the GNU General Public License.

A thin functional wrapper ``bea25(...)`` is provided at the bottom for
backward compatibility with the original procedural API.
"""

import numpy as np

from ..base import DirectivityModel, DirectivityResult, parse_imt


# ---------------------------------------------------------------------------
# Concrete implementation: Bayless et al. (2025)
# ---------------------------------------------------------------------------
class Bea25Model(DirectivityModel):
    """Bayless et al. (2025) directivity adjustment model (bea25).

    Parameters
    ----------
    version : int, default 2
        1 -> simulation-based coefficients.
        2 -> NGA-W2 data-based coefficients.
    """

    # fixed model constants
    D_MIN = 3.0                 # km, minimum source-site travel-distance term
    FS_CAP_LENGTH = 465.0       # km, cap on S2 (~ L for an M8, ~6.14 ln units)
    N_PERIODS = 1000            # number of log-spaced output periods

    def __init__(self, version: int = 2):
        if version == 1:        # simulation-based
            self.Amax, self.k, self.SigG = 0.54, 1.58, 0.38
            self.PhiPer = np.array(
                [0.01, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
            self.e1 = np.array(
                [0.00, 0.000, 0.0003, 0.011, 0.038, 0.072, 0.107,
                 0.143, 0.172, 0.189, 0.195, 0.206, 0.200])
        elif version == 2:      # NGA-W2 event-based
            self.Amax, self.k, self.SigG = 0.34, 1.58, 0.26
            self.PhiPer = np.array(
                [0.01, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
            self.e1 = np.array(
                [0.00, 0.000, 0.0024, 0.0074, 0.024, 0.041, 0.064,
                 0.076, 0.091, 0.110, 0.124, 0.145, 0.157])
        else:
            raise ValueError(
                "version must be 1 (simulation-based) or 2 (NGA-W2 data-based).")
        self.version = version

    # -- metadata -----------------------------------------------------------
    @property
    def name(self) -> str:
        kind = "simulation-based" if self.version == 1 else "NGA-W2 data-based"
        return f"Bayless et al. (2025) [{kind}]"

    @property
    def m_limits(self) -> tuple:
        return (6.0, 8.6)

    @property
    def period_limits(self) -> tuple:
        return (0.01, 10.0)

    @property
    def input_params(self) -> tuple:
        return ("mag", "gc2u", "gc2t", "smax1", "smax2", "ztor", "rake", "imt")

    # -- core API -----------------------------------------------------------
    def evaluate(self, mag, gc2u, gc2t, smax1, smax2, ztor, rake,
                 imt) -> DirectivityResult:
        """Directivity adjustment (ln units) for Bea25.

        Shared-convention signature: ``mag`` first, ``imt`` last, lowercase
        names; ``gc2u``/``gc2t`` are the GC2 coordinates (km), ``smax1``/
        ``smax2`` the GC2-U at the rupture ends, ``ztor`` the depth to top of
        rupture, ``rake`` the rake. ``imt`` is a numeric SA period [s] or
        "SA(T)"; "PGA"/"PGV" yield no adjustment (Bea25 is period-based).
        """
        # alias to the symbol names used by the ported equations
        M, U, T, Smax1, Smax2, Ztor, Rake = (
            mag, gc2u, gc2t, smax1, smax2, ztor, rake)
        kind, Period = parse_imt(imt)
        # present setting M limit to 8.6

        self._check_magnitude(M)                       # limits on M; Table 4-3
        
        U = np.atleast_1d(np.asarray(U, dtype=float)).ravel()
        T = np.atleast_1d(np.asarray(T, dtype=float)).ravel()

        if kind != "SA":
            # PGA/PGV: Bea25 is period-based and defines no adjustment here
            return DirectivityResult(
                fd=np.zeros_like(U), model=self.name,
                extra={"imt": imt, "note": f"{kind}: no Bea25 directivity"})

        # (1) narrow-band amplitude and phi-reduction terms
        Per, Tpeak, A, e1interp = self._narrowband_terms(M)

        # (2) period-independent geometric predictors and tapers
        P = self._geometric_predictors(M, U, T, Smax1, Smax2, Ztor, Rake)

        # (3) directivity adjustment fD  [Eq. 2]
        logistic = 2.0 / (1.0 + np.exp(-self.k * P['fGprime'])) - 1.0
        fD = A[None, :] * logistic[:, None]            # (n, 1000)
        ti = int(np.argmin(np.abs(Per - Period)))      # requested-period index
        fDi = fD[:, ti]

        # (4) phi reduction, only inside the directivity footprint
        PhiRed = np.tile(e1interp, (fD.shape[0], 1))
        PhiRed[~P['Footprint'], :] = 0.0
        PhiRedi = PhiRed[:, ti]

        # (5) assemble outputs
        predic_funcs = {
            'fG': P['fG'],
            'fGprime': P['fGprime'],
            'fGbar': P['fGbar'],
            'fdist': P['fdist'],
            'fztor': P['fztor'],
            'ftheta': P['ftheta'],
            'fs2': P['fs2'],
            'A': A,
        }
        other = {
            'Per': Per,
            'Rmax': P['Rmax'],
            'Footprint': P['Footprint'],
            'Tpeak': Tpeak,
            'k': self.k,
            'Amax': self.Amax,
            'Rst': P['R'],
            'Ry0': P['Ry'],
            'S2': P['S2'],
            'theta': P['theta'],
        }
        return DirectivityResult(
            fd=fDi,                       # primary output: adjustment at Period
            model=self.name,
            extra={
                'fD': fD, 'fDi': fDi,
                'PhiRed': PhiRed, 'PhiRedi': PhiRedi,
                'predic_funcs': predic_funcs, 'other': other,
            },
        )

    # -- step (1): narrow-band terms ----------------------------------------
    def _narrowband_terms(self, M):
        Per = np.logspace(-2, 1, self.N_PERIODS)        # period array
        Tpeak = 10.0 ** (-2.15 + 0.404 * M)             # [Eq. 4a]
        x = np.log10(Per / Tpeak)                       # narrow band @ Tpeak
        A = self.Amax * np.exp(-x ** 2 / (2 * self.SigG ** 2))  # [Eq. 4]
        e1interp = np.interp(np.log(Per), np.log(self.PhiPer), self.e1)
        return Per, Tpeak, A, e1interp

    # -- step (2): geometric predictors and tapers --------------------------
    def _geometric_predictors(self, M, U, T, Smax1, Smax2, Ztor, Rake):
        D = self.D_MIN

        # (2a) convert U to S (positive toward strike, negative opposite)
        S = np.zeros_like(U)
        uneg = U < 0
        upos = U >= 0
        S[uneg] = -np.minimum(np.abs(U[uneg]), np.abs(Smax1))
        S[upos] = np.minimum(np.abs(U[upos]), np.abs(Smax2))

        # (2b) convert U to Ry0
        Ry = np.zeros_like(U)
        Ry[upos] = U[upos] - Smax2
        Ry[uneg] = np.abs(U[uneg]) - np.abs(Smax1)
        utween = (U <= Smax2) & (U >= Smax1)
        Ry[utween] = 0.0

        # (2c) S2
        Srake = S * np.cos(np.deg2rad(Rake))            # sign can flip with rake
        S2 = np.sqrt(D ** 2 + Srake ** 2)               # [Eq. 3b]

        # (2d) predictor variable fs2 (with cap)
        fs2 = np.log(S2)
        fsCap = np.log(self.FS_CAP_LENGTH)
        fs2[fs2 > fsCap] = fsCap

        # (2e) angular predictor variables
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.abs(np.arctan(T / U))            # [Eq. 3c]
        theta[np.isnan(theta)] = 0.0                     # undefined on the trace
        ftheta = np.abs(np.cos(2.0 * theta))

        # (2f) distance taper
        R = np.sqrt(T ** 2 + Ry ** 2 + Ztor ** 2)
        Rmax = -60.0 + 20.0 * M                          # [Eq. 3f]
        if M > 7:
            Rmax = 80.0
        Footprint = R <= Rmax
        fdist = np.zeros_like(R)
        with np.errstate(divide='ignore', invalid='ignore'):
            fdist[Footprint] = 1.0 - np.exp(-4.0 * Rmax / R[Footprint] + 4.0)  # [Eq. 3e]

        # (2g) Ztor taper [Eq. 3d]
        if abs(Ztor) < 20:
            fztor = np.full_like(R, 1.0 - abs(Ztor) / 20.0)
        else:
            fztor = np.zeros_like(R)

        # (2h) fG, fGprime
        fG = fs2 * ftheta                               # [Eq. 3a]
        fGbar = np.zeros(len(R))
        for ii in range(len(R)):
            fGbar[ii] = self._centering_term(Smax2, abs(Smax1), R[ii], D, Rake)
        fGprime = (fG - fGbar) * fdist * fztor          # [Eq. 3]

        return {
            'S': S, 'Ry': Ry, 'S2': S2, 'fs2': fs2, 'theta': theta,
            'ftheta': ftheta, 'R': R, 'Rmax': Rmax, 'Footprint': Footprint,
            'fdist': fdist, 'fztor': fztor, 'fG': fG, 'fGbar': fGbar,
            'fGprime': fGprime,
        }

    # -- centering term (the original centerfunc) ---------------------------
    def _centering_term(self, L1, L2, R, D, Rake):
        """Centering term fGbar for a given rupture dimension, hypocenter
        location, rake angle, and distance."""
        if R < 0.1:
            R = 0.1

        dx = 0.1
        cr = np.cos(np.deg2rad(Rake))

        with np.errstate(divide='ignore', invalid='ignore'):
            # I1 — between the ends of the fault, strike direction
            xx = self._mcolon(0.0, dx, L1)
            s = np.sqrt((xx * cr) ** 2 + D ** 2)
            y1 = np.log(s) * np.abs(np.cos(2.0 * np.arctan(R / xx)))

            # I2 — between the ends of the fault, anti-strike direction
            xx = self._mcolon(0.0, dx, L2)
            s = np.sqrt((xx * cr) ** 2 + D ** 2)
            y2 = np.log(s) * np.abs(np.cos(2.0 * np.arctan(R / xx)))

            # I3 — off the end of the fault, strike direction
            ll = L1
            s = np.sqrt((ll * cr) ** 2 + D ** 2)
            xx = self._mcolon(ll + dx, dx, ll + R)
            r = np.sqrt(np.maximum(R ** 2 - (xx - ll) ** 2, 0.0))
            y3 = np.log(s) * np.abs(np.cos(2.0 * np.arctan(r / xx)))

            # I4 — off the end, anti-strike direction
            ll = L2
            s = np.sqrt((ll * cr) ** 2 + D ** 2)
            xx = self._mcolon(ll + dx, dx, ll + R)
            r = np.sqrt(np.maximum(R ** 2 - (xx - ll) ** 2, 0.0))
            y4 = np.log(s) * np.abs(np.cos(2.0 * np.arctan(r / xx)))

        return float(np.mean(np.concatenate([y1, y2, y3, y4])))

    @staticmethod
    def _mcolon(a, d, b):
        """Replicate MATLAB's ``a:d:b`` colon operator (inclusive of b)."""
        if d == 0:
            return np.array([a])
        n_int = int(np.floor((b - a) / d + 1e-10))
        if n_int < 0:
            return np.array([])
        return a + d * np.arange(n_int + 1)


# ---------------------------------------------------------------------------
# Backward-compatible functional API
# ---------------------------------------------------------------------------
def bea25(M, U, T, Smax1, Smax2, Ztor, Rake, Period, Version):
    """Procedural wrapper mirroring the original ``bea25.m`` signature.

    Returns the 6-tuple ``(fD, fDi, PhiRed, PhiRedi, predic_funcs, other)``.
    """
    r = Bea25Model(version=Version).evaluate(
        M, U, T, Smax1, Smax2, Ztor, Rake, Period)
    return (r['fD'], r['fDi'], r['PhiRed'], r['PhiRedi'],
            r['predic_funcs'], r['other'])
