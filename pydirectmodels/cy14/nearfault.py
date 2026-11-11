"""
nearfault — standalone port of openquake.hazardlib.near_fault
=============================================================

Faithful NumPy/SciPy port of the Spudich et al. (2013) isochrone-directivity
functions used to compute DPP: ``get_xyz_from_ll``, ``get_plane_equation``,
``projection_pp``, ``vectors2angle``, ``average_s_rad``, ``isochone_ratio``,
``_intersection`` and ``directp``. Geodesy comes from geom.py instead of
OpenQuake, so this has no OpenQuake dependency.

``directp`` additionally returns ``n_seg`` (which patch edge the direct point
lands on: 2 = far/forward edge, 4 = near/back edge) so the DPP driver can
follow rupture propagation in *either* along-strike direction.

Original OpenQuake code: Copyright (C) GEM Foundation, GNU AGPL v3.
"""

import math
import numpy as np
import scipy.spatial.distance as dst

from .geom import azimuth, geodetic_distance

__all__ = ["get_xyz_from_ll", "get_plane_equation", "projection_pp",
           "vectors2angle", "average_s_rad", "isochone_ratio", "directp"]


def get_xyz_from_ll(projected, reference):
    """x, y, z (km) of a Point relative to a reference Point."""
    azims = azimuth(reference.longitude, reference.latitude,
                    projected.longitude, projected.latitude)
    depths = np.subtract(reference.depth, projected.depth)
    dists = geodetic_distance(reference.longitude, reference.latitude,
                              projected.longitude, projected.latitude)
    return np.array([dists * math.sin(math.radians(azims)),
                     dists * math.cos(math.radians(azims)),
                     depths])


def get_plane_equation(p0, p1, p2, reference):
    """Plane through p0,p1,p2: returns (normal, dist_to_plane) with
    normal=(a,b,c) and ax+by+cz = dist_to_plane."""
    p0_xyz = get_xyz_from_ll(p0, reference)
    p1_xyz = get_xyz_from_ll(p1, reference)
    p2_xyz = get_xyz_from_ll(p2, reference)
    u = np.array(p1_xyz) - np.array(p0_xyz)
    v = np.array(p2_xyz) - np.array(p0_xyz)
    normal = np.cross(u, v)
    dist_to_plane = np.dot(np.array(p0_xyz), normal)
    return normal, dist_to_plane


def projection_pp(site, normal, dist_to_plane, reference):
    """Perpendicular projection Pp of the site onto the fault plane (xyz)."""
    site_x, site_y, site_z = get_xyz_from_ll(site, reference)
    a = np.array([(1, 0, 0, -normal[0]),
                  (0, 1, 0, -normal[1]),
                  (0, 0, 1, -normal[2]),
                  (normal[0], normal[1], normal[2], 0)], dtype=float)
    b = np.array([site_x, site_y, site_z, dist_to_plane], dtype=float)
    x = np.linalg.solve(a, b)
    return np.array([x[0], x[1], x[2]])


def vectors2angle(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


def average_s_rad(site, hypocenter, reference, pp, normal, dist_to_plane,
                  e, p0, p1, delta_slip):
    """Average S-wave radiation pattern fs, plus rd and r_hyp (Spudich 2013)."""
    site_xyz = get_xyz_from_ll(site, reference)
    zs = dst.pdist([pp, site_xyz])
    if (site_xyz[0] * normal[0] + site_xyz[1] * normal[1] +
            site_xyz[2] * normal[2] - dist_to_plane) > 0:
        zs = -zs
    hyp_xyz = np.array(get_xyz_from_ll(hypocenter, reference)).reshape(1, 3).flatten()
    l2 = dst.pdist([pp, hyp_xyz])
    rd = ((l2 - e) ** 2 + zs ** 2) ** 0.5
    r_hyp = (l2 ** 2 + zs ** 2) ** 0.5
    p0_xyz = get_xyz_from_ll(p0, reference)
    p1_xyz = get_xyz_from_ll(p1, reference)
    u = np.array(p1_xyz) - np.array(p0_xyz)
    v = pp - hyp_xyz
    phi = vectors2angle(u, v) - np.deg2rad(delta_slip)
    with np.errstate(divide='ignore', invalid='ignore'):
        ix = np.cos(phi) * (2 * zs * (l2 / r_hyp - (l2 - e) / rd) -
                            zs * np.log((l2 + r_hyp) / (l2 - e + rd)))
        inn = np.cos(phi) * (-2 * zs ** 2 * (1 / r_hyp - 1 / rd) - (r_hyp - rd))
        iphi = np.sin(phi) * (zs * np.log((l2 + r_hyp) / (l2 - e + rd)))
        fs = (ix ** 2 + inn ** 2 + iphi ** 2) ** 0.5 / e
    return fs, rd, r_hyp


def isochone_ratio(e, rd, r_hyp):
    """Isochrone ratio c' (Spudich 2013, p.88)."""
    if e == 0.0:
        return 0.8
    return 1.0 / ((1.0 / 0.8) - ((r_hyp - rd) / e))


def _intersection(seg1_start, seg1_end, seg2_start, seg2_end):
    """Closest intersection of two 3-D segments (Cartesian)."""
    pa = np.array([seg1_start, seg2_start])
    pb = np.array([seg1_end, seg2_end])
    si = pb - pa
    ni = si / np.power(
        np.dot(np.sum(si ** 2, axis=1).reshape(2, 1), np.ones((1, 3))), 0.5)
    nx = ni[:, 0].reshape(2, 1)
    ny = ni[:, 1].reshape(2, 1)
    nz = ni[:, 2].reshape(2, 1)
    sxx = np.sum(nx ** 2 - 1)
    syy = np.sum(ny ** 2 - 1)
    szz = np.sum(nz ** 2 - 1)
    sxy = np.sum(nx * ny)
    sxz = np.sum(nx * nz)
    syz = np.sum(ny * nz)
    s = np.array([sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz]).reshape(3, 3)
    cx = np.sum(pa[:, 0].reshape(2, 1) * (nx ** 2 - 1) +
                pa[:, 1].reshape(2, 1) * (nx * ny) +
                pa[:, 2].reshape(2, 1) * (nx * nz))
    cy = np.sum(pa[:, 0].reshape(2, 1) * (nx * ny) +
                pa[:, 1].reshape(2, 1) * (ny ** 2 - 1) +
                pa[:, 2].reshape(2, 1) * (ny * nz))
    cz = np.sum(pa[:, 0].reshape(2, 1) * (nx * nz) +
                pa[:, 1].reshape(2, 1) * (ny * nz) +
                pa[:, 2].reshape(2, 1) * (nz ** 2 - 1))
    c = np.array([cx, cy, cz]).reshape(3, 1)
    p_intersect = np.linalg.solve(s, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        v1 = (p_intersect.flatten() - seg2_end) / \
            np.sum((p_intersect.flatten() - seg2_end) ** 2) ** 0.5
        v2 = (seg2_start - seg2_end) / \
            np.sum((seg2_start - seg2_end) ** 2) ** 0.5
        v3 = (seg1_end - seg1_start) / \
            np.sum((seg1_end - seg1_start) ** 2) ** 0.5
        v4 = (p_intersect.flatten() - seg1_start) / \
            np.sum((p_intersect.flatten() - seg1_start) ** 2) ** 0.5
    return p_intersect, v1, v2, v3, v4


def directp(node0, node1, node2, node3, hypocenter, reference, pp,
            max_loop=1000):
    """Direct point Pd and E-path. Returns (pd, e, go_next_patch, n_seg).

    ``n_seg`` is the patch edge the direct point lies on: 1 top, 2 far/forward,
    3 bottom, 4 near/back. OpenQuake's ``go_next_patch`` == (n_seg == 2).
    """
    node0_xyz = get_xyz_from_ll(node0, reference)
    node1_xyz = get_xyz_from_ll(node1, reference)
    node2_xyz = get_xyz_from_ll(node2, reference)
    node3_xyz = get_xyz_from_ll(node3, reference)
    hyp_xyz = np.array(get_xyz_from_ll(hypocenter, reference)).flatten()
    pp_xyz = pp

    segment_s = [node0_xyz, node1_xyz, node2_xyz, node3_xyz]
    segment_e = [node1_xyz, node2_xyz, node3_xyz, node0_xyz]

    x_min = min(node0_xyz[0], node1_xyz[0], node2_xyz[0], node3_xyz[0])
    x_max = max(node0_xyz[0], node1_xyz[0], node2_xyz[0], node3_xyz[0])
    y_min = min(node0_xyz[1], node1_xyz[1], node2_xyz[1], node3_xyz[1])
    y_max = max(node0_xyz[1], node1_xyz[1], node2_xyz[1], node3_xyz[1])
    buf = 0.01  # km, generous in-bounds tolerance in the projected frame
    ppph = float(np.asarray(dst.pdist([pp, hyp_xyz])).ravel()[0])

    # Among the 4 patch edges, the direct point is the intersection that (a) is
    # between the hypocentre and Pp (ppph >= pdph), (b) lies within the patch
    # bounds, and (c) best aligns the direction vectors. OpenQuake reaches the
    # same edge by slowly loosening a tolerance until the first edge matches;
    # picking the minimum-mismatch valid edge is equivalent but single-pass.
    best = None  # (mismatch, n_seg, p_intersect, pdph)
    for i, (seg_s, seg_e) in enumerate(zip(segment_s, segment_e), start=1):
        seg_s = np.asarray(seg_s).flatten()
        seg_e = np.asarray(seg_e).flatten()
        p_int, v1, v2, v3, v4 = _intersection(seg_s, seg_e, pp_xyz, hyp_xyz)
        pdph = float(np.asarray(dst.pdist([p_int.flatten(), hyp_xyz])).ravel()[0])
        if ppph + 1e-9 < pdph:
            continue
        if not (x_min - buf <= p_int[0] <= x_max + buf and
                y_min - buf <= p_int[1] <= y_max + buf):
            continue
        m = max(np.linalg.norm(v1.flatten() - v2),
                np.linalg.norm(v3.flatten() - v4))
        if not np.isfinite(m):
            continue
        if best is None or m < best[0]:
            best = (m, i, p_int, pdph)

    if best is not None and best[0] <= 0.5:    # accepted direct point on an edge
        n_seg = best[1]
        pd = best[2]
        e = np.array([best[3]])
    else:                                       # Pp inside the patch -> Pd = Pp
        pd = pp_xyz
        e = dst.pdist([pp, hyp_xyz])
        n_seg = 0

    go_next_patch = (n_seg == 2)
    return pd, e, go_next_patch, n_seg
