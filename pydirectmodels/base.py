"""
base — abstract interface for rupture directivity adjustment models
===================================================================

Defines:
    DirectivityResult  -- flexible output container shared by all models
    DirectivityModel   -- abstract base class defining the model interface

Different models need different inputs and produce different auxiliary
outputs, so the interface is intentionally permissive:

* ``evaluate`` has no fixed signature -- each model declares its own
  parameters (and may advertise them via :attr:`input_params`). A model may
  also expose additional geometry-driven entry points, e.g. the CY14 model's
  ``evaluate_from_geometry(surface, hypocenter, sites, mag, imt, ...)``.
* :class:`DirectivityResult` carries the one quantity every directivity model
  produces -- the ln-units adjustment ``fd`` -- plus an open ``extra`` dict for
  model-specific arrays, accessible by attribute or item lookup.

Concrete models in this package:
    pydirectmodels.bea25.Bea25Model
        Bayless et al. (2025); GC2-coordinate inputs.
    pydirectmodels.cy14.ChiouYoungs2014Directivity
        Chiou & Youngs (2014); rrup/cdpp inputs, or a rupture surface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def parse_imt(imt):
    """Parse an intensity-measure spec shared by all models.

    Accepts a numeric SA period [s], or a string "PGA" / "PGV" / "SA(T)".
    Returns ``(kind, period)`` with ``kind`` in {"PGA", "PGV", "SA"}; for PGA
    and PGV ``period`` is 0.0 and -1.0 respectively (sentinels).

    This gives every model's ``evaluate(..., imt)`` the same ``imt`` argument
    so the spectral target is specified the same way across the package.
    """
    if isinstance(imt, str):
        s = imt.strip().upper()
        if s == "PGA":
            return "PGA", 0.0
        if s == "PGV":
            return "PGV", -1.0
        if s.startswith("SA(") and s.endswith(")"):
            return "SA", float(s[3:-1])
        return "SA", float(s)          # bare numeric string
    return "SA", float(imt)


# ---------------------------------------------------------------------------
# Result container (flexible: common `fd` + model-specific `extra`)
# ---------------------------------------------------------------------------
@dataclass
class DirectivityResult:
    """Output of a directivity-model evaluation.

    Parameters
    ----------
    fd : float or ndarray
        The directivity adjustment in natural-log units -- the single output
        every directivity model produces (added to the ln median ground
        motion). For period-dependent models this is the value at the
        requested period/IMT.
    model : str
        Name of the producing model.
    extra : dict
        Model-specific outputs. Accessible by item (``res['fDi']``), by
        attribute (``res.fDi``; attribute lookup falls through to ``extra``),
        or via :meth:`get`.
    """
    fd: object
    model: str = ""
    extra: dict = field(default_factory=dict)

    # -- mapping-style access to `extra` -----------------------------------
    def __getitem__(self, key):
        return self.extra[key]

    def __contains__(self, key):
        return key in self.extra

    def keys(self):
        return self.extra.keys()

    def get(self, key, default=None):
        """Return ``extra[key]`` if present, else ``default``."""
        return self.extra.get(key, default)

    def __getattr__(self, name):
        # only invoked when normal attribute lookup fails; route to `extra`
        extra = self.__dict__.get("extra")
        if extra is not None and name in extra:
            return extra[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    def __repr__(self):
        fd = np.asarray(self.fd)
        shape = "scalar" if fd.ndim == 0 else f"array{tuple(fd.shape)}"
        keys = ", ".join(self.extra) if self.extra else ""
        tail = f", extra=[{keys}]" if keys else ""
        return f"DirectivityResult(model={self.model!r}, fd={shape}{tail})"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class DirectivityModel(ABC):
    """Abstract base class for rupture directivity adjustment models.

    A directivity model maps a rupture/site description to a period-dependent
    ground-motion adjustment ``fd`` (natural-log units) and, optionally, an
    aleatory-variability (phi) reduction and predictor functions.

    Concrete subclasses must implement :attr:`name` and :meth:`evaluate`.
    They may override the optional metadata (:attr:`m_limits`,
    :attr:`period_limits`, :attr:`input_params`); the defaults impose no
    constraints so models with very different inputs can coexist. Subclasses
    may also add further entry points (e.g. a geometry-driven evaluator) that
    ultimately return a :class:`DirectivityResult`.
    """

    # -- required metadata --------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name/version."""

    # -- optional metadata (override where applicable) ----------------------
    @property
    def m_limits(self):
        """Valid magnitude range ``(Mmin, Mmax)``, or None if unconstrained."""
        return None

    @property
    def period_limits(self):
        """Valid spectral-period range [s], or None if unconstrained."""
        return None

    @property
    def input_params(self) -> tuple:
        """Names of the parameters accepted by :meth:`evaluate` (for tools/UIs)."""
        return ()

    # -- core API (signature is model-specific) -----------------------------
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> DirectivityResult:
        """Evaluate the directivity adjustment. Signature is model-specific."""

    def __call__(self, *args, **kwargs) -> DirectivityResult:
        """Convenience alias for :meth:`evaluate`."""
        return self.evaluate(*args, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"

    # -- shared helpers -----------------------------------------------------
    def _check_magnitude(self, M):
        lim = self.m_limits
        if lim is None:
            return
        mmin, mmax = lim
        if M > mmax:
            raise ValueError(f"Upper M limit is {mmax:.1f}")
        if M < mmin:
            raise ValueError(f"Lower M limit is {mmin:.1f}")

    @staticmethod
    def _mcolon(a, d, b):
        """Replicate MATLAB's ``a:d:b`` colon operator (inclusive of b)."""
        if d == 0:
            return np.array([a])
        n_int = int(np.floor((b - a) / d + 1e-10))
        if n_int < 0:
            return np.array([])
        return a + d * np.arange(n_int + 1)
