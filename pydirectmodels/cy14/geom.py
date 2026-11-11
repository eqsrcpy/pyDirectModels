"""
geom — minimal standalone geodesy + multi-segment fault surface
===============================================================

Provides the small slice of OpenQuake's geometry that the DPP/cdpp calculation
needs, so it can run without OpenQuake:

* spherical geodesy: ``geodetic_distance``, ``azimuth``, ``point_at``
  (faithful to openquake.hazardlib.geo.geodetic, EARTH_RADIUS = 6371 km)
* :class:`Point` (lon, lat, depth) with ``point_at`` / ``azimuth``
* :class:`Mesh` (arrays of lon/lat/depth)
* :class:`MultiSegmentSurface`: a fault built from a (possibly bent) surface
  trace plus a uniform dip and seismogenic depths. It exposes the methods the
  DPP loop calls -- ``get_resampled_top_edge``, ``get_fault_patch_vertices``,
  ``hypocentre_patch_index``, ``get_dip``, ``get_min_distance``,
  ``get_bounding_box`` -- splitting the (multi-segment) trace into ordered
  rectangular patches.

Distances used by the directivity math are computed in a local ENU frame
(consistent with near_fault, which also works in that projected frame); at
near-fault scales this matches OpenQuake's spherical Rrup closely.
"""

import math
import numpy as np

EARTH_RADIUS = 6371.0

__all__ = ["EARTH_RADIUS", "geodetic_distance", "azimuth", "point_at",
           "Point", "Mesh", "MultiSegmentSurface", "SingleSegmentSurface"]


# ---------------------------------------------------------------------------
# spherical geodesy (faithful to openquake.hazardlib.geo.geodetic)
# ---------------------------------------------------------------------------
def geodetic_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    d = np.arcsin(np.sqrt(
        np.sin((lat1 - lat2) / 2.0) ** 2 +
        np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2.0) ** 2))
    return 2.0 * EARTH_RADIUS * d


def azimuth(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    cos_lat2 = np.cos(lat2)
    true_course = np.degrees(np.arctan2(
        np.sin(lon1 - lon2) * cos_lat2,
        np.cos(lat1) * np.sin(lat2) -
        np.sin(lat1) * cos_lat2 * np.cos(lon1 - lon2)))
    return (360.0 - true_course) % 360.0


def point_at(lon, lat, azim, distance):
    """Destination lon/lat from (lon, lat) along ``azim`` for ``distance`` km."""
    lon, lat = math.radians(lon), math.radians(lat)
    tc = math.radians(azim)
    sd = distance / EARTH_RADIUS
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lat2 = sin_lat * math.cos(sd) + cos_lat * math.sin(sd) * math.cos(tc)
    sin_lat2 = max(-1.0, min(1.0, sin_lat2))
    lat2 = math.asin(sin_lat2)
    dlon = math.atan2(math.sin(tc) * math.sin(sd) * cos_lat,
                      math.cos(sd) - sin_lat * sin_lat2)
    lon2 = (math.degrees(lon + dlon) + 180.0) % 360.0 - 180.0
    return lon2, math.degrees(lat2)


# ---------------------------------------------------------------------------
# Point / Mesh
# ---------------------------------------------------------------------------
class Point:
    def __init__(self, longitude, latitude, depth=0.0):
        self.longitude = float(longitude)
        self.latitude = float(latitude)
        self.depth = float(depth)

    def point_at(self, horizontal_distance, vertical_increment, azim):
        lon, lat = point_at(self.longitude, self.latitude, azim,
                            horizontal_distance)
        return Point(lon, lat, self.depth + vertical_increment)

    def azimuth(self, other):
        return azimuth(self.longitude, self.latitude,
                       other.longitude, other.latitude)

    def distance(self, other):
        hd = geodetic_distance(self.longitude, self.latitude,
                               other.longitude, other.latitude)
        return math.hypot(hd, self.depth - other.depth)

    def __repr__(self):
        return f"Point({self.longitude:.5f}, {self.latitude:.5f}, {self.depth:.3f})"


class Mesh:
    """Holds arrays of lon/lat/depth (any shape)."""
    def __init__(self, lons, lats, depths=None):
        self.lons = np.asarray(lons, dtype=float)
        self.lats = np.asarray(lats, dtype=float)
        self.depths = (np.zeros_like(self.lons) if depths is None
                       else np.asarray(depths, dtype=float))

    @property
    def shape(self):
        return self.lons.shape


def _xyz(lon, lat, depth, ref):
    """Local ENU (km) of (lon,lat,depth) relative to a reference Point."""
    az = azimuth(ref.longitude, ref.latitude, lon, lat)
    d = geodetic_distance(ref.longitude, ref.latitude, lon, lat)
    return np.array([d * math.sin(math.radians(az)),
                     d * math.cos(math.radians(az)),
                     ref.depth - depth])


# ---------------------------------------------------------------------------
# Multi-segment fault surface
# ---------------------------------------------------------------------------
class MultiSegmentSurface:
    """A fault surface from a (multi-segment) surface trace + uniform dip.

    Parameters
    ----------
    trace : sequence of (lon, lat)
        Ordered surface-trace vertices (the top edge). Two or more points;
        bends between segments are allowed.
    dip : float
        Dip angle [deg] (90 = vertical), uniform along the fault.
    upper_depth, lower_depth : float
        Top and bottom seismogenic depths [km].
    """

    def __init__(self, trace, dip, upper_depth=0.0, lower_depth=15.0):
        if len(trace) < 2:
            raise ValueError("trace needs at least two (lon, lat) vertices")
        self.upper_depth = float(upper_depth)
        self.lower_depth = float(lower_depth)
        self.dip = float(dip)
        self.trace = [Point(lo, la, self.upper_depth) for lo, la in trace]

    # -- methods used by the DPP loop --------------------------------------
    def get_dip(self):
        return self.dip

    def get_resampled_top_edge(self, spacing=1.0):
        """Resample the trace into ~``spacing`` km steps (vertices preserved)."""
        pts = [self.trace[0]]
        for a, b in zip(self.trace[:-1], self.trace[1:]):
            d = geodetic_distance(a.longitude, a.latitude,
                                  b.longitude, b.latitude)
            n = max(int(round(d / spacing)), 1)
            az = azimuth(a.longitude, a.latitude, b.longitude, b.latitude)
            for i in range(1, n + 1):
                lon, lat = point_at(a.longitude, a.latitude, az, d * i / n)
                pts.append(Point(lon, lat, self.upper_depth))
        return pts

    @staticmethod
    def get_fault_patch_vertices(top_edge, upper, lower, dip, index_patch=1):
        """Rectangle for patch ``index_patch`` (1-based), clockwise:
        p0,p1 on the top edge; p2 below p1; p3 below p0."""
        p0 = top_edge[index_patch - 1]
        p1 = top_edge[index_patch]
        vert = lower - upper
        horiz = 0.0 if dip >= 89.99999 else vert / math.tan(math.radians(dip))
        ddir = (p0.azimuth(p1) + 90.0) % 360.0            # down-dip direction
        p2 = p1.point_at(horiz, vert, ddir)
        p3 = p0.point_at(horiz, vert, ddir)
        return p0, p1, p2, p3

    def hypocentre_patch_index(self, hypo, top_edge, upper, lower, dip):
        """Index (1-based) of the patch whose top-edge point is nearest the
        hypocenter (horizontally)."""
        n_patch = len(top_edge) - 1
        dists = [geodetic_distance(hypo.longitude, hypo.latitude,
                                   p.longitude, p.latitude) for p in top_edge]
        k = int(np.argmin(dists))                          # 0-based edge point
        return min(max(k, 1), n_patch)

    # -- distance / bounding box -------------------------------------------
    def _surface_cloud(self, spacing=2.0):
        """3-D point cloud (lon, lat, depth arrays) sampling the whole surface."""
        top = self.get_resampled_top_edge(spacing)
        vert = self.lower_depth - self.upper_depth
        horiz = 0.0 if self.dip >= 89.99999 else \
            vert / math.tan(math.radians(self.dip))
        width = math.hypot(vert, horiz)
        ndown = max(int(round(width / spacing)), 1)
        lons, lats, deps = [], [], []
        for i in range(1, len(top)):
            p0, p1, p2, p3 = self.get_fault_patch_vertices(
                top, self.upper_depth, self.lower_depth, self.dip, i)
            for j in range(ndown + 1):
                f = j / ndown
                # linear blend of the two down-dip edges (near-fault accuracy)
                for (a, b) in ((p0, p3), (p1, p2)):
                    lons.append(a.longitude + f * (b.longitude - a.longitude))
                    lats.append(a.latitude + f * (b.latitude - a.latitude))
                    deps.append(a.depth + f * (b.depth - a.depth))
        return np.array(lons), np.array(lats), np.array(deps)

    def get_min_distance(self, mesh):
        """Rrup: min distance [km] from each mesh point to the surface."""
        from scipy.spatial.distance import cdist
        ref = self.trace[0]
        clon, clat, cdep = self._surface_cloud()
        cloud = np.array([_xyz(lo, la, de, ref)
                          for lo, la, de in zip(clon, clat, cdep)])
        lons = np.asarray(mesh.lons).ravel()
        lats = np.asarray(mesh.lats).ravel()
        deps = np.asarray(mesh.depths).ravel()
        sites = np.array([_xyz(lo, la, de, ref)
                          for lo, la, de in zip(lons, lats, deps)])
        return cdist(sites, cloud).min(axis=1)

    def get_bounding_box(self):
        """Returns (min_lon, max_lon, max_lat, min_lat) over the surface."""
        clon, clat, _ = self._surface_cloud()
        return clon.min(), clon.max(), clat.max(), clat.min()


class SingleSegmentSurface(MultiSegmentSurface):
    """Planar single-segment fault: one straight trace, one rectangular face.

    Construct from the two surface-trace endpoints::

        SingleSegmentSurface((lon0, lat0), (lon1, lat1), dip, upper, lower)

    or from a midpoint + strike + length via :meth:`from_strike`.
    """

    def __init__(self, p_start, p_end, dip, upper_depth=0.0, lower_depth=15.0):
        super().__init__([p_start, p_end], dip, upper_depth, lower_depth)

    @classmethod
    def from_strike(cls, midpoint, strike, length, dip,
                    upper_depth=0.0, lower_depth=15.0):
        """Build from a trace midpoint ``(lon, lat)``, ``strike`` [deg] and
        ``length`` [km]."""
        mid = Point(midpoint[0], midpoint[1], upper_depth)
        half = length / 2.0
        a = mid.point_at(half, 0.0, (strike + 180.0) % 360.0)
        b = mid.point_at(half, 0.0, strike)
        return cls((a.longitude, a.latitude), (b.longitude, b.latitude),
                   dip, upper_depth, lower_depth)
