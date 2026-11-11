"""
cdpp — standalone Direct Point Parameter (DPP) and centered DPP (cdpp)
=====================================================================

Standalone (OpenQuake-free) port of OpenQuake's get_dppvalue / get_cdppvalue
(Spudich et al., 2013), built on the local geom.py geometry and the ported
nearfault.py isochrone math. ``cdpp`` is the input the CY14 directivity term
needs (cy14.py).

Multi-segment faults are handled by splitting the (possibly bent) trace into
ordered rectangular patches (geom.MultiSegmentSurface) and walking patch by
patch from the hypocenter toward the site's direct point, advancing the source
to the direct point at each patch boundary and accumulating the contributions
(summed, then log). Unlike OpenQuake's reference loop -- which only propagates
in the +index direction -- this walks in *either* along-strike direction, so a
hypocenter anywhere along the fault is handled (the "bidirectional" subtlety).

VALIDATION NOTE: the isochrone math (nearfault.py) is a faithful port, but this
standalone driver and the surface layer have been checked for self-consistency
(symmetry, forward/back sign, finiteness) -- not bit-matched against OpenQuake.
Validate against OpenQuake for production use.
"""

import numpy as np

from .geom import Point, Mesh, MultiSegmentSurface
from .nearfault import (get_plane_equation, projection_pp, directp,
                        average_s_rad, isochone_ratio)
from .geom import geodetic_distance

__all__ = ["get_dppvalue", "get_cdppvalue", "compute_cdpp"]


def _f(x):
    """Coerce a scalar/len-1 array to float."""
    return float(np.asarray(x).ravel()[0])


def _eval_patch(surface, top, origin, dip, upper, lower, hypo, site,
                slip_dir, index_patch):
    """One patch's DPP contribution; returns (dpp_exp, pd_geo, n_seg)."""
    p0, p1, p2, p3 = surface.get_fault_patch_vertices(
        top, upper, lower, dip, index_patch)
    normal, dtp = get_plane_equation(p0, p1, p2, origin)
    pp = projection_pp(site, normal, dtp, origin)
    pd, e, _go, n_seg = directp(p0, p1, p2, p3, hypo, origin, pp)

    pd0, pd1, pd2 = (_f(pd[0]), _f(pd[1]), _f(pd[2]))
    pd_geo = origin.point_at((pd0 * pd0 + pd1 * pd1) ** 0.5, -pd2,
                             np.degrees(np.arctan2(pd0, pd1)))

    f1 = geodetic_distance(p0.longitude, p0.latitude, p1.longitude, p1.latitude)
    f2 = geodetic_distance(p2.longitude, p2.latitude, p3.longitude, p3.latitude)
    f = max(f1, f2)

    fs, rd, r_hyp = average_s_rad(site, hypo, origin, pp, normal, dtp, e,
                                  p0, p1, slip_dir)
    e_f = _f(e)
    cprime = isochone_ratio(e_f, _f(rd), _f(r_hyp))
    dpp_exp = cprime * max(e_f, 0.1 * f) * max(_f(fs), 0.2)
    return dpp_exp, pd_geo, n_seg


def get_dppvalue(surface, hypocenter, slip_direction, site, spacing=1.0):
    """DPP at a single site (Spudich et al., 2013), bidirectional traversal.

    Parameters
    ----------
    surface : MultiSegmentSurface
    hypocenter : Point
    slip_direction : float
        Slip direction away from strike [deg] (0 for pure strike-slip).
    site : Point
    spacing : float
        Top-edge resampling step [km] (patch size).

    Returns
    -------
    float
        DPP value, log of the summed patch contributions.
    """
    top = surface.get_resampled_top_edge(spacing)
    origin = top[0]
    upper, lower, dip = surface.upper_depth, surface.lower_depth, surface.get_dip()
    n_patch = len(top) - 1
    start = surface.hypocentre_patch_index(hypocenter, top, upper, lower, dip)

    contribs = []
    dpp_exp, pd_geo, n_seg = _eval_patch(
        surface, top, origin, dip, upper, lower, hypocenter, site, slip_direction,
        start)
    contribs.append(dpp_exp)

    # direction of the direct point: +1 far/forward edge, -1 near/back edge
    direction = 1 if n_seg == 2 else (-1 if n_seg == 4 else 0)
    ipatch, hypo = start, pd_geo
    while direction != 0:
        ipatch += direction
        if ipatch < 1 or ipatch > n_patch:
            break
        dpp_exp, pd_geo, n_seg = _eval_patch(
            surface, top, origin, dip, upper, lower, hypo, site, slip_direction,
            ipatch)
        contribs.append(dpp_exp)
        keep = (n_seg == 2) if direction == 1 else (n_seg == 4)
        if not keep:
            break
        hypo = pd_geo

    finite = [c for c in contribs if np.isfinite(c) and c > 0]
    if not finite:
        return np.nan        # fully degenerate (e.g. site on the fault plane)
    return float(np.log(np.sum(finite)))


def get_cdppvalue(surface, hypocenter, slip_direction, target,
                  buf=1.0, delta=0.05, space=2.0, spacing=1.0):
    """Centered DPP (cdpp / rcdpp) at the target sites.

    Centers each site's DPP on the mean DPP over a grid of comparable
    rupture-distance sites (Spudich et al., 2013). ``target`` is a Mesh.

    Note: ``delta`` defaults to 0.05 deg here (OpenQuake uses 0.01) to keep the
    pure-Python centering grid tractable; tighten it for production accuracy.
    """
    min_lon, max_lon, max_lat, min_lat = surface.get_bounding_box()
    min_lon -= buf
    max_lon += buf
    min_lat -= buf
    max_lat += buf

    lons = np.arange(min_lon, max_lon + delta, delta)
    lats = np.arange(min_lat, max_lat + delta, delta)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid = Mesh(grid_lon, grid_lat)
    grid_rup = surface.get_min_distance(grid).reshape(grid.shape)
    target_rup = surface.get_min_distance(target)

    flat_lon = grid.lons.ravel()
    flat_lat = grid.lats.ravel()
    flat_rup = grid_rup.ravel()

    # Precompute DPP at every centering-grid point ONCE (OpenQuake recomputes
    # these per target; the result is identical but this is far faster for maps).
    min_rrup = 0.5  # exclude on-/in-fault grid points (degenerate DPP)
    grid_dpp = np.full(flat_lon.shape, np.nan)
    ok = flat_rup > min_rrup
    for j in np.nonzero(ok)[0]:
        grid_dpp[j] = get_dppvalue(surface, hypocenter, slip_direction,
                                   Point(flat_lon[j], flat_lat[j]), spacing)

    cdpp = np.zeros_like(np.asarray(target.lons, dtype=float))
    for i, (t_lon, t_lat) in enumerate(zip(target.lons, target.lats)):
        band = (flat_rup <= target_rup[i] + space) & \
               (flat_rup >= target_rup[i] - space) & ok & np.isfinite(grid_dpp)
        dpp_target = get_dppvalue(surface, hypocenter, slip_direction,
                                  Point(t_lon, t_lat), spacing)
        dpp_mean = np.nanmean(grid_dpp[band]) if band.any() else np.nan
        cdpp[i] = dpp_target - dpp_mean
    return cdpp


def compute_cdpp(trace, dip, upper_depth, lower_depth, hypocenter,
                 site_lons, site_lats, slip_direction=0.0,
                 buf=1.0, delta=0.05, space=2.0, spacing=1.0):
    """Convenience: build a surface and return (cdpp, rrup) at the sites.

    ``trace`` is a list of (lon, lat); ``hypocenter`` is (lon, lat, depth).
    ``rrup`` is the rupture distance [km] for the CY14 distance taper.
    """
    surface = MultiSegmentSurface(trace, dip, upper_depth, lower_depth)
    hypo = Point(*hypocenter)
    sites = Mesh(np.asarray(site_lons, float), np.asarray(site_lats, float))
    cdpp = get_cdppvalue(surface, hypo, slip_direction, sites,
                         buf=buf, delta=delta, space=space, spacing=spacing)
    rrup = surface.get_min_distance(sites)
    return cdpp, rrup
