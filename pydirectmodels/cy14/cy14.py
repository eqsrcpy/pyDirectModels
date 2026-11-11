"""
cy14 — Chiou & Youngs (2014) near-fault directivity term
========================================================

Standalone (NumPy-only) port of ``get_directivity(C, ctx)`` from OpenQuake
(openquake/hazardlib/gsim/chiou_youngs_2014.py), with the CY14 directivity
coefficients (c8, c8a, c8b) bundled, wrapped as a DirectivityModel.

The CY14 directivity adjustment (added to the ln median ground motion) is

    f_dir = c8 * dist_taper * mag_taper * exp(-c8a * (M - c8b)**2) * cdpp
    mag_taper  = clip((M - 5.5) / 0.8, 0, 1)
    dist_taper = max(1 - max(Rrup - 40, 0) / 30, 0)

where ``cdpp`` is the *centered* direct-point parameter (OpenQuake's
``rcdpp``), the Spudich et al. (2013) directivity predictor. cdpp is an INPUT
here; compute it from rupture geometry with the companion ``cdpp`` module
(``get_dppvalue`` / ``get_cdppvalue``), or set it to 0 to disable directivity
(CY14's default).

Reference: Chiou & Youngs (2014), Earthquake Spectra 30(3), 1117-1153,
DOI:10.1193/072813EQS219M. Original OpenQuake code (C) GEM Foundation, AGPL v3.
"""

import numpy as np

from ..base import DirectivityModel, DirectivityResult, parse_imt

__all__ = [
    "ChiouYoungs2014Directivity",
    "get_directivity",
    "get_coeffs",
    "COEFFS_C8",
]

# ---------------------------------------------------------------------------
# CY14 directivity coefficients (c8, c8a, c8b), extracted from the CY14 COEFFS
# table. c8a == 0.2695 everywhere; c8 == 0 up to ~0.4 s then ramps in.
# ---------------------------------------------------------------------------
#         kind   period(s)   c8        c8a       c8b
_TABLE = [
    ("PGA",  0.0,    0.0,    0.2695, 0.4833),
    ("PGV", -1.0,    0.2154, 0.2695, 5.0),
    ("SA",   0.010,  0.0,    0.2695, 0.4833),
    ("SA",   0.020,  0.0,    0.2695, 1.2144),
    ("SA",   0.030,  0.0,    0.2695, 1.6421),
    ("SA",   0.040,  0.0,    0.2695, 1.9456),
    ("SA",   0.050,  0.0,    0.2695, 2.1810),
    ("SA",   0.075,  0.0,    0.2695, 2.6087),
    ("SA",   0.100,  0.0,    0.2695, 2.9122),
    ("SA",   0.120,  0.0,    0.2695, 3.1045),
    ("SA",   0.150,  0.0,    0.2695, 3.3399),
    ("SA",   0.170,  0.0,    0.2695, 3.4719),
    ("SA",   0.200,  0.0,    0.2695, 3.6434),
    ("SA",   0.250,  0.0,    0.2695, 3.8787),
    ("SA",   0.300,  0.0,    0.2695, 4.0711),
    ("SA",   0.400,  0.0,    0.2695, 4.3745),
    ("SA",   0.500,  0.0991, 0.2695, 4.6099),
    ("SA",   0.750,  0.1982, 0.2695, 5.0376),
    ("SA",   1.000,  0.2154, 0.2695, 5.3411),
    ("SA",   1.500,  0.2154, 0.2695, 5.7688),
    ("SA",   2.000,  0.2154, 0.2695, 6.0723),
    ("SA",   3.000,  0.2154, 0.2695, 6.5000),
    ("SA",   4.000,  0.2154, 0.2695, 6.8035),
    ("SA",   5.000,  0.2154, 0.2695, 7.0389),
    ("SA",   7.500,  0.2154, 0.2695, 7.4666),
    ("SA",  10.000,  0.2154, 0.2695, 7.7700),
]

COEFFS_C8 = {}
_SA_ROWS = {}
for _kind, _per, _c8, _c8a, _c8b in _TABLE:
    _d = {"c8": _c8, "c8a": _c8a, "c8b": _c8b}
    if _kind == "PGA":
        COEFFS_C8["PGA"] = _d
    elif _kind == "PGV":
        COEFFS_C8["PGV"] = _d
    else:
        COEFFS_C8[f"SA({_per:g})"] = _d
        _SA_ROWS[_per] = _d
_SA_PERIODS = np.array(sorted(_SA_ROWS))


def get_coeffs(imt, interpolate=False):
    """Return ``{'c8','c8a','c8b'}`` for an IMT.

    ``imt`` is "PGA", "PGV", "SA(<T>)", or a numeric SA period [s].
    If ``interpolate`` and the SA period is not tabulated, log-period linear
    interpolation is used (a convenience; CY14 is defined only at tabulated
    periods).
    """
    kind, period = parse_imt(imt)
    if kind == "PGA":
        return dict(COEFFS_C8["PGA"])
    if kind == "PGV":
        return dict(COEFFS_C8["PGV"])

    label = f"SA({period:g})"
    if label in COEFFS_C8:
        return dict(COEFFS_C8[label])
    if not interpolate:
        raise KeyError(
            f"period {period} s is not tabulated; pass interpolate=True or "
            f"use one of {list(_SA_PERIODS)}")
    if period < _SA_PERIODS[0] or period > _SA_PERIODS[-1]:
        raise ValueError(f"period {period} s outside tabulated range "
                         f"[{_SA_PERIODS[0]}, {_SA_PERIODS[-1]}] s")
    lp, lps = np.log(period), np.log(_SA_PERIODS)
    return {name: float(np.interp(
                lp, lps, [_SA_ROWS[p][name] for p in _SA_PERIODS]))
            for name in ("c8", "c8a", "c8b")}


def get_directivity(C, ctx):
    """Chiou & Youngs (2014) directivity term, in natural-log units.

    Faithful to openquake.hazardlib.gsim.chiou_youngs_2014.get_directivity.

    ``C`` is a mapping with keys c8, c8a, c8b. ``ctx`` exposes ``mag`` and
    ``rrup`` (scalars or arrays) and optionally ``rcdpp`` (the centered DPP);
    if ``rcdpp`` is absent the term is 0 (matching OpenQuake).
    """
    try:
        cdpp = ctx.rcdpp
    except AttributeError:
        return 0.0  # No directivity term

    mag = np.asarray(ctx.mag, dtype=float)
    rrup = np.asarray(ctx.rrup, dtype=float)
    cdpp = np.asarray(cdpp, dtype=float)

    f_dir = np.exp(-C["c8a"] * (mag - C["c8b"]) ** 2.0) * cdpp
    f_dir = f_dir * np.clip((mag - 5.5) / 0.8, 0.0, 1.0)            # mag taper
    dist_taper = np.maximum(1.0 - np.maximum(rrup - 40.0, 0.0) / 30.0, 0.0)
    return C["c8"] * dist_taper * f_dir


class _Ctx:
    """Minimal stand-in for an OpenQuake context (only the needed fields)."""
    def __init__(self, mag, rrup, rcdpp=None):
        self.mag = mag
        self.rrup = rrup
        if rcdpp is not None:
            self.rcdpp = rcdpp


class ChiouYoungs2014Directivity(DirectivityModel):
    """Chiou & Youngs (2014) near-fault directivity as a DirectivityModel.

    Example
    -------
    >>> m = ChiouYoungs2014Directivity()
    >>> res = m.evaluate(mag=7.0, rrup=10.0, cdpp=2.0, imt=1.0)  # SA(1.0 s)
    >>> float(res.fd)
    0.2052...
    """

    @property
    def name(self) -> str:
        return "Chiou & Youngs (2014) near-fault directivity"

    @property
    def period_limits(self) -> tuple:
        return (0.01, 10.0)

    @property
    def input_params(self) -> tuple:
        return ("mag", "rrup", "cdpp", "imt")

    def evaluate(self, mag, rrup, cdpp, imt, interpolate=False) -> DirectivityResult:
        """Directivity adjustment (ln units) from precomputed rrup & cdpp.

        ``mag``, ``rrup``, ``cdpp`` may be scalars or broadcastable arrays;
        ``imt`` is "PGA"/"PGV"/"SA(T)" or a numeric SA period in seconds.
        Use :meth:`evaluate_from_geometry` to compute rrup/cdpp from a rupture.
        """
        C = get_coeffs(imt, interpolate=interpolate)
        fd = get_directivity(C, _Ctx(mag, rrup, cdpp))
        return DirectivityResult(
            fd=fd, model=self.name,
            extra={"C": C, "imt": imt, "cdpp": np.asarray(cdpp, dtype=float)})

    def evaluate_from_geometry(self, surface, hypocenter, sites, mag, imt,
                               slip_direction=0.0, interpolate=False,
                               buf=1.0, delta=0.05, space=2.0, spacing=1.0):
        """Full pipeline: rupture geometry -> (cdpp, rrup) -> CY14 directivity.

        Parameters
        ----------
        surface : MultiSegmentSurface | SingleSegmentSurface
            The rupture surface (from ``pydirectmodels.cy14``).
        hypocenter : Point
            Hypocenter location.
        sites : Mesh
            Target sites.
        mag : float
            Moment magnitude.
        imt : str or float
            "PGA"/"PGV"/"SA(T)" or numeric SA period [s].
        slip_direction : float
            Slip direction away from strike [deg] (0 = pure strike-slip).
        buf, delta, space, spacing :
            cdpp centering-grid controls (see ``get_cdppvalue``).

        Returns
        -------
        DirectivityResult
            ``fd`` is the ln-units adjustment at each site; ``extra`` carries
            ``cdpp`` and ``rrup``.
        """
        from .cdpp import get_cdppvalue   # lazy: pulls scipy/geometry only here
        cdpp = get_cdppvalue(surface, hypocenter, slip_direction, sites,
                             buf=buf, delta=delta, space=space, spacing=spacing)
        rrup = surface.get_min_distance(sites)
        res = self.evaluate(mag, rrup, cdpp, imt, interpolate=interpolate)
        res.extra.update({"rrup": rrup, "cdpp": cdpp, "imt": imt})
        return res
