"""bea20 — Bayless, Somerville & Skarlatoudis (2020) directivity (Bea20).

GC2-based; predecessor of bea25. Ported from OpenQuake's BaylessEtAl2020.
"""

from .bea20 import (
    Bayless2020Model, Bea20Context, get_e1,
    get_median_adjustment_factor, get_within_event_stddev, CONSTANTS)

__all__ = [
    "Bayless2020Model",
    "Bea20Context",
    "get_e1",
    "get_median_adjustment_factor",
    "get_within_event_stddev",
    "CONSTANTS",
]
