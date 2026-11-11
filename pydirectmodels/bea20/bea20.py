"""
bea20 — Bayless, Somerville & Skarlatoudis (2020) rupture-directivity model
===========================================================================

Port of OpenQuake's ``BaylessEtAl2020`` directivity model (Bea20):
Bayless, Somerville & Skarlatoudis (2020), "A Rupture Directivity Adjustment
Model Applicable to the NGA-West2 Ground Motion Models and Complex Fault
Geometries", USGS Final Report G18AP00092. Bea20 is the predecessor of Bea25
(this package's bea25 model); both are GC2-based.

The model equations, constants and the e1 coefficient table are taken VERBATIM
from OpenQuake's implementation (openquake.hazardlib.gsim.directivity, GEM
Foundation, GNU AGPL v3). Only the interface is adapted to pyDirectModels:

* the OpenQuake ``DirectivityModel`` base -> this package's
  :class:`pydirectmodels.base.DirectivityModel`, returning a
  :class:`DirectivityResult`;
* the OpenQuake ``RuptureContext`` recarray -> the lightweight
  :class:`Bea20Context` (any duck-typed object with the same attributes also
  works, including an OpenQuake context);
* the OpenQuake ``CoeffsTable`` for e1 -> :func:`get_e1` (log-period interp);
* ``imt`` objects -> a numeric SA period or "PGA"/"SA(T)" string.

The median directivity adjustment (ln units) is
``fD = (a + b * (fs2 * ftheta * fphi)) * fdist``
and, for SA with T > 0.2 s, the within-event sigma is reduced per Eqs. 9a/9b.
"""

from dataclasses import dataclass, field

import numpy as np

from ..base import DirectivityModel, DirectivityResult, parse_imt

__all__ = ["Bayless2020Model", "Bea20Context", "get_e1",
           "get_median_adjustment_factor", "get_within_event_stddev",
           "CONSTANTS"]


# Model constants (Bea20; verbatim from OpenQuake)
CONSTANTS = {
    "b1": 0.5469,
    "b2": -0.0336,
    "c1": -1.2090,
    "c2": 0.2858,
    "d1_sof1": -4.8300,
    "d1_sof2": -1.5415,
    "d2_sof1": 0.9928,
    "d2_sof2": 0.3946,
    "sigma_g": 0.4653,
}

# Within-event directivity sigma-reduction coefficient e1 (Table 4.4)
_E1_PERIOD = np.array([0.01, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50,
                       2.00, 3.00, 4.00, 5.00, 7.50, 10.0])
_E1_VALUE = np.array([0.000, 0.000, 0.008, 0.020, 0.035, 0.051, 0.067, 0.080,
                      0.084, 0.093, 0.110, 0.139, 0.166, 0.188, 0.199])


def get_e1(period):
    """Interpolate the e1 sigma-reduction coefficient (linear in log period)."""
    if period <= _E1_PERIOD[0]:
        return float(_E1_VALUE[0])
    if period >= _E1_PERIOD[-1]:
        return float(_E1_VALUE[-1])
    return float(np.interp(np.log10(period),
                           np.log10(_E1_PERIOD), _E1_VALUE))


def _imt_flags(imt):
    """Return (is_pga, period) using the package-shared parse_imt."""
    kind, period = parse_imt(imt)
    return kind == "PGA", period


# ---------------------------------------------------------------------------
# Model functions (ported verbatim from OpenQuake's bayless_2020.py)
# ---------------------------------------------------------------------------
def _get_sof(rake, n):
    """Style-of-faulting index (1 strike-slip, 2 dip-slip)."""
    abs_rake = np.fabs(rake) + np.zeros(n)
    return np.where(np.logical_or(abs_rake < 30.0, abs_rake > 150.0), 1, 2)


def _get_sval(gc2uprime, gc_length, gc2u_updip_hypo):
    """S-value: GC2-U within the fault, else the U of the nearest end point."""
    return np.clip(gc2uprime, -gc2u_updip_hypo, gc_length - gc2u_updip_hypo)


def _get_a_b(mag, period, sof):
    """Magnitude/period scaling factors a and b (Eqs. 2b, 2c, 3a-3c)."""
    d1 = np.where(sof == 1, CONSTANTS["d1_sof1"], CONSTANTS["d1_sof2"])
    d2 = np.where(sof == 1, CONSTANTS["d2_sof1"], CONSTANTS["d2_sof2"])
    fg0 = d1 + d2 * mag                                         # Eq. 3c
    bmax = CONSTANTS["b1"] + CONSTANTS["b2"] * mag              # Eq. 3a
    tpeak = 10.0 ** (CONSTANTS["c1"] + CONSTANTS["c2"] * mag)   # Eq. 3b
    bmt = bmax * np.exp((np.log10(period / tpeak) ** 2) /
                        (-2.0 * (CONSTANTS["sigma_g"] ** 2.)))
    amt = -bmt * fg0
    return amt, bmt


def _get_rmax(sof, mag):
    """Maximum distance of the distance taper (Eqs. 3e, 3f)."""
    rsof = np.where(sof == 1, 0.0, 20.0)
    return np.where(mag > 7.0, 80.0 - rsof, 20.0 * mag - 60.0 - rsof)


def _get_fdist(ry0, gc2t, rmax):
    """Directivity distance taper (Eqs. 2d, 3d)."""
    rtaper = np.sqrt(gc2t ** 2.0 + ry0 ** 2.0)                 # Eq. 3d
    fdist = np.zeros(ry0.shape)
    idx = rtaper == 0.0
    if np.any(idx):
        fdist[idx] = 1.0
    idx = np.logical_and(rtaper > 0.0, rtaper <= rmax)
    if np.any(idx):
        fdist[idx] = 1.0 - np.exp(((-4.0 * rmax[idx]) / rtaper[idx]) + 4.0)
    return fdist


def _get_ftheta(sof, rake, ry0, gc2t, gc2u):
    """Azimuthal predictor ftheta (Eqs. 5a, 5b)."""
    ftheta = np.zeros(gc2t.shape)
    idx = sof == 1
    if np.any(idx):
        theta = np.fabs(np.arctan(gc2t[idx] / gc2u[idx]))
        theta[np.isnan(theta)] = 0.0
        ftheta[idx] = np.fabs(np.cos(2.0 * theta))
    idx = np.logical_not(idx)
    if np.any(idx):
        rake_sof2 = (rake + np.zeros(gc2t.shape))[idx]
        gc2t_sof2 = gc2t[idx]
        ry0_sof2 = ry0[idx]
        tmin = 10.0 * (np.fabs(np.cos(np.radians(rake_sof2))) + 1.0)
        theta = np.clip(np.fabs(gc2t_sof2), tmin, np.inf)
        idx2 = gc2t_sof2 > 0.0
        theta[idx2] = tmin[idx2]
        ftheta_sel = np.ones(theta.shape)
        idx2 = ry0_sof2 > 0.0
        ftheta_sel[idx2] = np.sin(np.arctan(np.fabs(theta[idx2] /
                                                    ry0_sof2[idx2])))
        ftheta[idx] = ftheta_sel
    return ftheta


def _get_fphi(sof, dip, gc2t, tbot, dbot):
    """Azimuthal predictor fphi (Eqs. 6a, 6b; Appendix B)."""
    nshp = gc2t.shape
    fphi = np.ones(nshp)
    idx0 = sof != 1
    if not np.any(idx0):
        return fphi                          # strike-slip: fphi = cos(0) = 1
    dip_sof2 = (dip + np.zeros(nshp))[idx0]
    tbot_sof2 = (tbot + np.zeros(nshp))[idx0]
    dbot_sof2 = (dbot + np.zeros(nshp))[idx0]
    gc2t_sof2 = gc2t[idx0]
    phi = np.zeros(gc2t_sof2.shape)
    idx1 = gc2t_sof2 < 0.0
    if np.any(idx1):
        phi[idx1] = np.radians(dip_sof2[idx1] - 90.0) + \
            np.arctan((np.fabs(gc2t_sof2[idx1]) + tbot_sof2[idx1]) /
                      dbot_sof2[idx1])
    idx1 = np.logical_and(gc2t_sof2 >= 0.0, gc2t_sof2 <= tbot_sof2)
    if np.any(idx1):
        phi[idx1] = np.radians(90.0 - dip_sof2[idx1]) - \
            np.arctan((tbot_sof2[idx1] - gc2t_sof2[idx1]) / dbot_sof2[idx1])
    idx1 = gc2t_sof2 > tbot_sof2
    if np.any(idx1):
        phi[idx1] = np.radians(90.0 - dip_sof2[idx1]) + \
            np.arctan((gc2t_sof2[idx1] - tbot_sof2[idx1]) / dbot_sof2[idx1])
    phi[phi > (np.pi / 4.0)] = np.pi / 4.0   # capped at dip 45 deg (authors)
    fphi[idx0] = np.cos(2.0 * phi)           # Eq. 6b
    return fphi


def _get_fs2(sof, rake, dval, sval):
    """Rupture travel-distance term fs2 (Eqs. 4a, 4b)."""
    s_cos_rake = sval * np.cos(np.radians(rake))
    fs2 = np.zeros(sval.shape)
    idx = np.logical_and(sof == 2, s_cos_rake < 0.0)
    if np.any(idx):
        fs2[idx] = np.clip(np.log(dval[idx]), -np.inf, np.log(465.0))
    idx = np.logical_not(idx)
    if np.any(idx):
        s_2 = np.sqrt(dval[idx] ** 2.0 + s_cos_rake[idx] ** 2.0)
        fs2[idx] = np.clip(np.log(s_2), -np.inf, np.log(188.0))
    return fs2


def get_median_adjustment_factor(ctx, period, sof, rmax):
    """Median directivity adjustment fD (Eq. 1; uses Eqs. 2a-6b)."""
    gc2t = np.asarray(ctx.gc2t, dtype=float)
    shape = gc2t.shape
    aval, bval = _get_a_b(ctx.mag, period, sof)
    dval = np.clip((ctx.hypo_depth - ctx.ztor) /
                   np.sin(np.radians(ctx.dip)), 3.0, np.inf) + np.zeros(shape)
    t_offset = np.where(
        ctx.ztor > 0.0,
        ctx.ztor / np.tan(np.radians(ctx.dip)),   # buried: recentre on trace
        0.0,                                       # surface-breaking
    )
    tbot = (ctx.zbor / np.tan(np.radians(ctx.dip))) + t_offset
    gc2tprime = gc2t + t_offset
    gc2uprime = np.asarray(ctx.gc2u, dtype=float) - ctx.gc2u_updip_hypo
    sval = _get_sval(gc2uprime, ctx.gc_length, ctx.gc2u_updip_hypo)
    ry0 = np.asarray(ctx.ry0, dtype=float)
    fdist = _get_fdist(ry0, gc2tprime, rmax)
    fs2 = _get_fs2(sof, ctx.rake, dval, sval)
    ftheta = _get_ftheta(sof, ctx.rake, ry0, gc2tprime, gc2uprime)
    fphi = _get_fphi(sof, ctx.dip, gc2tprime, tbot, ctx.zbor)
    return (aval + bval * (fs2 * ftheta * fphi)) * fdist


def get_within_event_stddev(e1, period, is_pga, phi, rmax, rrup):
    """Reduced within-event sigma (Eqs. 9a, 9b). No reduction for T <= 0.2 s."""
    phi = np.asarray(phi, dtype=float)
    if is_pga or period <= 0.2:
        return phi
    rrup = np.asarray(rrup, dtype=float)
    phi_red = np.zeros(rrup.shape)
    phi_red[rrup < rmax] = e1
    return np.sqrt(phi ** 2.0 - phi_red ** 2.0)


# ---------------------------------------------------------------------------
# Input context + model class
# ---------------------------------------------------------------------------
@dataclass
class Bea20Context:
    """Inputs for :class:`Bayless2020Model` (mirrors the OpenQuake context).

    Rupture scalars: ``mag``, ``rake``, ``dip`` [deg], ``ztor``/``zbor``
    (top/bottom of rupture depth, km), ``hypo_depth`` [km], ``gc_length``
    (fault length in GC2-U, km), ``gc2u_updip_hypo`` (GC2-U of the up-dip
    hypocentre projection, km).

    Site arrays (same length N): ``gc2u``, ``gc2t`` (GC2 coordinates, km),
    ``ry0`` (off-end distance, km), ``rrup`` (rupture distance, km; only used
    for the sigma reduction).
    """
    mag: float
    rake: float
    dip: float
    ztor: float
    zbor: float
    hypo_depth: float
    gc_length: float
    gc2u_updip_hypo: float
    gc2u: object
    gc2t: object
    ry0: object
    rrup: object = None


class Bayless2020Model(DirectivityModel):
    """Bayless, Somerville & Skarlatoudis (2020) directivity model (Bea20)."""

    @property
    def name(self) -> str:
        return "Bayless et al. (2020) directivity [Bea20]"

    @property
    def period_limits(self):
        return (0.01, 10.0)

    @property
    def input_params(self) -> tuple:
        return ("mag", "gc2u", "gc2t", "ry0", "gc_length", "gc2u_updip_hypo",
                "ztor", "zbor", "hypo_depth", "dip", "rake", "imt")

    def evaluate(self, ctx=None, imt=None, phi=None, **params) -> DirectivityResult:
        """Median directivity adjustment ``fd`` (ln units) at ``imt``.

        Two equivalent call styles:

        * explicit parameters (like Bea25Model)::

              model.evaluate(imt=3.0, mag=7.0, rake=0.0, dip=90.0,
                             ztor=0.0, zbor=12.0, hypo_depth=6.0,
                             gc_length=50.0, gc2u_updip_hypo=0.0,
                             gc2u=..., gc2t=..., ry0=...)

        * a context object (a :class:`Bea20Context`, or any object exposing
          the same attributes, e.g. an OpenQuake RuptureContext)::

              model.evaluate(ctx, imt=3.0)

        ``imt`` is a numeric SA period [s] or "PGA"/"SA(T)". If ``phi``
        (within-event sigma) is given and ``rrup`` is available, the reduced
        sigma is returned in ``extra['phi']``.
        """
        if ctx is None:
            ctx = Bea20Context(**params)        # explicit-parameter call style
        elif params:
            raise TypeError("pass either a context object or keyword "
                            "parameters, not both")
        elif not hasattr(ctx, "gc2t"):
            raise TypeError("first argument must be a Bea20Context (or object "
                            f"with the same attributes), got {type(ctx).__name__}")
        if imt is None:
            raise TypeError("imt is required (numeric SA period or 'PGA'/'SA(T)')")

        is_pga, period = _imt_flags(imt)
        shape = np.asarray(ctx.gc2t, dtype=float).shape
        sof = _get_sof(ctx.rake, shape)
        rmax = _get_rmax(sof, ctx.mag)

        if is_pga:
            fd = np.zeros(shape)            # no directivity adjustment for PGA
        else:
            fd = get_median_adjustment_factor(ctx, period, sof, rmax)

        extra = {"sof": sof, "rmax": rmax, "imt": imt, "period": period}
        if phi is not None and getattr(ctx, "rrup", None) is not None:
            e1 = get_e1(period)
            extra["e1"] = e1
            extra["phi"] = get_within_event_stddev(
                e1, period, is_pga, phi, rmax, ctx.rrup)
        return DirectivityResult(fd=fd, model=self.name, extra=extra)
