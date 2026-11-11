"""
dmtk: tools for classifying and tracing multi-segment / multi-fault ruptures
for directivity modelling.

Main entry points
  get_fsegments(...)          sections of a rupture, grouped by ParentID
  check_multifault(...)       True/False: junction criteria (Weatherill & Lilienkamp, 2023,
                              or Al Atik et al., 2023), then strand-configuration criteria
  classify_rupture(...)       full result: strands, ordered sections, breaks, relations
  check_segment_relation(...) configuration of extra strands relative to their nearest strand
  get_ruptures_inpoly(...)    ruptures that cross or lie within a polygon
  plot_faults(...)            sections of a set of ruptures, coloured by rake
  rupture_spline_traces(...)  smoothed trace for each fault strand, points every spacing_km
  strand_magnitudes(...)      area-weighted magnitude of each fault strand
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import geopandas as gpd

import shapely
from scipy.interpolate import splprep, splev
from scipy.stats import circmean

EARTH_RADIUS_KM = 6371.0088


# =============================================================== geometry helpers
def _geoms(seg):
    """Shapely geometries of a segment's sections (parses WKT if needed)."""
    g = np.asarray(seg['geometry'])
    if isinstance(g[0], str):
        g = shapely.from_wkt(g)
    return g


def to_km(lonlat, lon0, lat0, R=EARTH_RADIUS_KM):
    """lon/lat (deg) -> local tangent-plane x/y (km) around (lon0, lat0)."""
    lonlat = np.asarray(lonlat, dtype=float)
    return np.column_stack([np.deg2rad(lonlat[:, 0] - lon0) * R * np.cos(np.deg2rad(lat0)),
                            np.deg2rad(lonlat[:, 1] - lat0) * R])


def to_lonlat(xy, lon0, lat0, R=EARTH_RADIUS_KM):
    """Inverse of to_km."""
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([np.rad2deg(xy[:, 0] / (R * np.cos(np.deg2rad(lat0)))) + lon0,
                            np.rad2deg(xy[:, 1] / R) + lat0])


# =============================================================== distances
def haversine_km(lon1, lat1, lon2, lat2, EARTH_RADIUS_KM=EARTH_RADIUS_KM):
    """Great-circle distance in km; inputs in degrees, broadcastable."""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def segment_min_distance_haversine(pts_a, pts_b):
    """Minimum great-circle distance (km) between two (N, 2) lon/lat point arrays."""
    d = haversine_km(pts_a[:, None, 0], pts_a[:, None, 1],
                     pts_b[None, :, 0], pts_b[None, :, 1])
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(d[i, j])


def distance_between_two_segments(seg1, seg2):
    """Minimum great-circle distance (km) between the upper edges of two segments."""
    return segment_min_distance_haversine(segment_trace(seg1), segment_trace(seg2))


# =============================================================== angles
def first_quadrant_rake(rake):
    """Fold a rake angle into the first quadrant [0, 90]°:
    0 = pure strike-slip, 90 = pure dip-slip."""
    r = abs((rake + 180) % 360 - 180)   # wrap to [-180, 180], magnitude -> [0, 180]
    return 180 - r if r > 90 else r


def dip_change(dip1, dipdir1, dip2, dipdir2, use_dipdir=True):
    """
    Change in dip (°) between two planes.
    use_dipdir=False : plain |dip2 - dip1|, as in Al Atik et al. (2023)
    use_dipdir=True  : opposite-facing dips measured from the same side (0–180°)
    """
    if use_dipdir:
        ddir = abs((dipdir2 - dipdir1 + 180) % 360 - 180)   # 0–180°
        if ddir > 90:
            dip2 = 180 - dip2
    return abs(dip2 - dip1)


def _mean_dip(seg):
    return float(np.mean(seg['DipDeg']))


def _mean_dipdir(seg):
    """Circular mean so that e.g. 350° and 10° average to 0°, not 180°."""
    return float(circmean(seg['DipDir'], high=360, low=0))


def _mean_rake(seg):
    return float(circmean(seg['Rake'], high=180, low=-180))


def dip_difference_two_segments(seg1, seg2, use_dipdir=True):
    """Change in mean dip between two segments (see dip_change)."""
    return dip_change(_mean_dip(seg1), _mean_dipdir(seg1),
                      _mean_dip(seg2), _mean_dipdir(seg2), use_dipdir)


def rake_difference_two_segments(seg1, seg2):
    """Change in first-quadrant (circular-mean) rake between two segments."""
    return abs(first_quadrant_rake(_mean_rake(seg2)) - first_quadrant_rake(_mean_rake(seg1)))


def section_area_km2(sections):
    """
    Fault-plane area (km²) of each section: upper-edge length x down-dip width,
    with width = (LowDepth - UpDepth) / sin(dip).
    Do NOT use geometry.area: the polygons are surface projections, so vertical
    sections have zero area and dipping ones are underestimated.
    """
    edges = [_upper_edge(g) for g in _geoms(sections)]
    length = np.array([haversine_km(e[0, 0], e[0, 1], e[1, 0], e[1, 1]) for e in edges])
    width = ((sections['LowDepth'] - sections['UpDepth']).to_numpy(dtype=float)
             / np.sin(np.deg2rad(sections['DipDeg'].to_numpy(dtype=float))))
    return length * width


def strand_magnitude(M_rupture, strand_area, rupture_area):
    """Strand magnitude with moment split by area (Al Atik et al., 2023):
    M_i = M + (2/3) log10(A_i / A). Use fault-plane areas (section_area_km2),
    not polygon (surface-projection) areas."""
    return M_rupture + (2.0 / 3.0) * np.log10(strand_area / rupture_area)


def strand_magnitudes(result, M_rupture):
    """Magnitude of each fault strand of a classify_rupture() result.
    Returns a DataFrame with fault_strand, area_km2, area_frac, M."""
    sec = result['sections']
    area = pd.Series(section_area_km2(sec), index=sec.index)
    a = area.groupby(sec['fault_strand']).sum()
    out = pd.DataFrame({'area_km2': a, 'area_frac': a / a.sum()})
    out['M'] = strand_magnitude(M_rupture, out['area_km2'], a.sum())
    return out.reset_index()



# =============================================================== rupture segments
def sort_fsegments_WENS(fsegs):
    """
    Sort segments west -> east (ties north -> south) by their mean upper-edge midpoint.
    Use for listing/plotting only: it interleaves side-by-side faults, so it is NOT
    a valid along-rupture order. Use classify_rupture (strands from build_strands) for that.
    """
    def segment_point(seg):
        return np.mean([np.asarray(x.exterior.coords)[:2].mean(axis=0) for x in _geoms(seg)], axis=0)

    seg_pts = np.array([segment_point(seg) for seg in fsegs])
    order = np.lexsort((-seg_pts[:, 1], seg_pts[:, 0]))
    return [fsegs[i] for i in order]


def get_fsegments(rupindex, rupture_sections, fault_sections, do_sort_WENS=False):
    """Sections of one rupture grouped into segments (one DataFrame per ParentID)."""
    rsections = rupture_sections.loc[rupture_sections['rupture'] == rupindex, 'section'].tolist()
    sel = fault_sections[fault_sections['FaultID'].isin(rsections)]
    fsegments = [grp for _, grp in sel.groupby('ParentID', sort=False)]
    if do_sort_WENS:
        fsegments = sort_fsegments_WENS(fsegments)
    return fsegments


# =============================================================== strands
def segment_trace(seg):
    """Upper-edge vertices (lon, lat) of a segment, ordered along its end-to-end direction."""
    p = np.unique(np.vstack([np.asarray(x.exterior.coords)[:2] for x in _geoms(seg)]).round(6), axis=0)
    q = p * [np.cos(np.deg2rad(p[:, 1].mean())), 1.0]
    d = np.linalg.norm(q[:, None] - q[None], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return p[np.argsort((q - q[i]) @ (q[j] - q[i]))]


def _pair_geometry(A, B):
    """Along-strike overlap (km) of two ordered segment traces, measured on the longer
    one's axis, and their closest pair of end points (distance, end of A, end of B)."""
    LA, LB = np.linalg.norm(A[-1] - A[0]), np.linalg.norm(B[-1] - B[0])
    P, Q = (A, B) if LA >= LB else (B, A)
    L = max(np.linalg.norm(P[-1] - P[0]), 1e-9)
    u = (P[-1] - P[0]) / L
    s = (Q - P[0]) @ u
    overlap = max(0.0, min(L, s.max()) - max(0.0, s.min()))
    ends_a, ends_b = (A[0], A[-1]), (B[0], B[-1])
    d, ea, eb = min((np.linalg.norm(ends_a[i] - ends_b[j]), i, j) for i in (0, 1) for j in (0, 1))
    return overlap, d, ea, eb


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def _continues(A, ea, B, eb, behind_km=10.0, touch_km=1.0):
    """True if B continues A through the end points A[ea] -> B[eb]: neither starts more
    than behind_km back along the other (overlaps are tested separately), and the strike
    bends by less than 90°, so a strand never folds back on itself."""
    pa, pb = (A[0], A[-1])[ea], (B[0], B[-1])[eb]
    va = _unit((A[-1] - A[0]) if ea == 0 else (A[0] - A[-1]))   # from linked end into A
    vb = _unit((B[-1] - B[0]) if eb == 0 else (B[0] - B[-1]))   # from linked end into B
    if va @ vb >= 0:                                            # both run the same way
        return False
    g = pb - pa
    if np.linalg.norm(g) < touch_km:
        return True
    return (g @ va < behind_km) and (g @ vb > -behind_km)


def _link_offset(A, ea, B, eb):
    """Sideways (across-strike) offset of a link: distance of each linked end from the
    other segment's line, whichever is larger."""
    ua, ub = _unit(A[-1] - A[0]), _unit(B[-1] - B[0])
    pa, pb = (A[0], A[-1])[ea], (B[0], B[-1])[eb]
    return max(abs(np.cross(ua, pb - A[0])), abs(np.cross(ub, pa - B[0])))


def _aligned(A, ea, B, eb, max_dstrike=30.0):
    """For long jumps: B continues A at a similar strike."""
    ua, ub = _unit(A[-1] - A[0]), _unit(B[-1] - B[0])
    return np.degrees(np.arccos(min(1.0, abs(ua @ ub)))) <= max_dstrike


def fault_name(parent_name):
    """Fault a segment belongs to: its parent name before the first ':'
    ('Awatere: Southwest' -> 'Awatere'; 'Cloudy 2' -> 'Cloudy 2')."""
    return str(parent_name).split(':')[0].strip()


def build_strands(traces_km, max_jump_km=20.0, overlap_km=10.0, max_dstrike=30.0,
                  max_offset_km=10.0, spur_km=5.0, fault_names=None):
    """
    Link segments end to end into strands, in two passes.

    A link joins the closest pair of end points of two segments when
      - their along-strike overlap is at most overlap_km (larger overlaps are
        side-by-side strands, never linked),
      - B continues A without folding back (_continues), and
      - the link's sideways offset is at most max_offset_km (a wider step-over
        separates strands, however short the gap).
    Gaps up to max_jump_km are linked; longer jumps only if B also continues A at a
    similar strike (within max_dstrike), so a gap alone does not separate strands.
    If fault_names are given, two segments of the same fault are linked across an
    overlapping step-over (overlap is not tested), as long as they do not fold back
    and the step is within max_offset_km: a step-over within one fault is one strand.
    Links between segments longer than spur_km are made first, so a short spur near a
    junction cannot take the end point that the main line continues through. Within
    each group links are made shortest-first; each end point takes one link and no
    loops are formed, so every strand is a simple chain. The longest strand comes first.
    Returns a list of (segment_indices, oriented_traces_km, flipped_flags).
    """
    n = len(traces_km)
    seg_len = [np.linalg.norm(np.diff(t, axis=0), axis=1).sum() for t in traces_km]
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            A, B = traces_km[i], traces_km[j]
            ov, d, ei, ej = _pair_geometry(A, B)
            same_fault = fault_names is not None and fault_names[i] == fault_names[j]
            if same_fault:
                if not _continues(A, ei, B, ej, behind_km=np.inf):
                    continue
            elif ov > overlap_km or not _continues(A, ei, B, ej, behind_km=overlap_km):
                continue
            if _link_offset(A, ei, B, ej) > max_offset_km:
                continue
            if d <= max_jump_km:
                far_jump = False
            elif _aligned(A, ei, B, ej, max_dstrike):
                far_jump = True
            else:
                continue
            spur = min(seg_len[i], seg_len[j]) <= spur_km
            cands.append((spur, far_jump, d, i, ei, j, ej))
    cands = [c[2:] for c in sorted(cands)]

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    link = {}                                   # (seg, end) -> (other seg, other end)
    for d, i, ei, j, ej in cands:
        if (i, ei) in link or (j, ej) in link or find(i) == find(j):
            continue
        link[(i, ei)], link[(j, ej)] = (j, ej), (i, ei)
        parent[find(i)] = find(j)

    strands, seen = [], set()
    starts = [i for i in range(n) if ((i, 0) in link) + ((i, 1) in link) <= 1]
    for s0 in starts:
        if s0 in seen:
            continue
        entry = 0 if (s0, 0) not in link else 1     # enter at the free end
        idx, ori, fl, cur = [], [], [], s0
        while True:
            seen.add(cur)
            t = traces_km[cur]
            idx.append(cur)
            ori.append(t if entry == 0 else t[::-1])
            fl.append(entry == 1)
            nxt = link.get((cur, 1 - entry))
            if nxt is None or nxt[0] in seen:
                break
            cur, entry = nxt
        strands.append((idx, ori, fl))

    def length(st):
        pts = np.vstack(st[1])
        return np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    strands.sort(key=length, reverse=True)
    return _orient_strands(strands)


def _reverse_strand(strand):
    idx, oriented, flipped = strand
    return (idx[::-1], [o[::-1] for o in oriented[::-1]], [not f for f in flipped[::-1]])


def _orient_strands(strands):
    """Fix strand directions so results do not depend on the input order of segments:
    the main strand runs west -> east (south -> north if it trends N-S), and every
    other strand runs the same general way as the main strand."""
    def vec(st):
        return st[1][-1][-1] - st[1][0][0]
    main = strands[0]
    v = vec(main)
    ref = np.array([1.0, 0.0]) if abs(v[0]) >= abs(v[1]) else np.array([0.0, 1.0])
    if v @ ref < 0:
        main = _reverse_strand(main)
    main_vec = vec(main)
    out = [main]
    for st in strands[1:]:
        out.append(_reverse_strand(st) if vec(st) @ main_vec < 0 else st)
    return out


def strand_relation(main_xy, other_xy, window_km=20.0, overlap_km=10.0, max_dstrike=30.0):
    """
    Relation of a strand to the nearby part of the main strand (both in km, ordered):
      'parallel'   : overlaps the main strand along strike at a similar strike
      'branching'  : overlaps it at a different strike
      'end-to-end' : no significant overlap (offset_km gives the across-strike step)
      'separate'   : further than (strand length + window_km) from any main-strand point
    Also returns overlap, separation and strike difference.
    """
    dist = np.min(np.linalg.norm(main_xy[:, None] - other_xy[None], axis=-1), axis=1)
    sep = float(dist.min())
    L_other = np.linalg.norm(other_xy[-1] - other_xy[0])
    uo = (other_xy[-1] - other_xy[0]) / max(L_other, 1e-9)

    i_m = int(np.argmin(dist))
    i_o = int(np.argmin(np.linalg.norm(other_xy - main_xy[i_m], axis=1)))
    near = main_xy[dist <= L_other + window_km]
    if len(near) < 2 or np.linalg.norm(near[-1] - near[0]) < 1e-6:
        near, rel = main_xy, 'separate'          # too far for a local comparison
    else:
        rel = None
    u = (near[-1] - near[0]) / np.linalg.norm(near[-1] - near[0])
    s_main, s_oth = (near - near[0]) @ u, (other_xy - near[0]) @ u
    overlap = max(0.0, min(s_main.max(), s_oth.max()) - max(s_main.min(), s_oth.min()))
    dstrike = np.degrees(np.arccos(min(1.0, abs(u @ uo))))
    offset = abs(np.cross(u, other_xy[i_o] - main_xy[i_m]))
    if rel is None:
        if overlap > overlap_km:
            rel = 'parallel' if dstrike <= max_dstrike else 'branching'
        else:
            rel = 'end-to-end'
    return dict(overlap_km=float(overlap), separation_km=sep, offset_km=float(offset),
                dstrike_deg=float(dstrike), relation=rel)


def _strands_for(fsegments, max_jump_km=20.0, overlap_km=10.0, max_dstrike=30.0,
                 max_offset_km=10.0, same_fault_stepovers=True):
    tr_ll = [segment_trace(s) for s in fsegments]
    lon0, lat0 = np.vstack(tr_ll).mean(axis=0)
    tr_km = [to_km(t, lon0, lat0) for t in tr_ll]
    names = [fault_name(s['ParentName'].iloc[0]) for s in fsegments] if same_fault_stepovers else None
    return build_strands(tr_km, max_jump_km, overlap_km, max_dstrike, max_offset_km,
                         fault_names=names), (lon0, lat0)


def strand_relations(strands, overlap_km=10.0, max_dstrike=30.0, max_offset_km=10.0,
                     min_branch_km=None):
    """
    Relation of every strand after the first (main) one to its NEAREST other strand.
    A non-overlapping strand offset sideways by more than max_offset_km is a 'stepover'.
    Otherwise a strand that stays separate (its end meets a strand whose end is already
    taken) is a 'branching' junction if longer than min_branch_km (default overlap_km);
    shorter spurs stay 'end-to-end' and belong to the fault strand they touch.
    Each relation also gives 'strand' and 'to_strand'.
    """
    if min_branch_km is None:
        min_branch_km = overlap_km
    xy = [np.vstack(o) for _, o, _ in strands]
    out = []
    for k in range(1, len(strands)):
        best = None
        for m in range(len(strands)):
            if m == k:
                continue
            r = strand_relation(xy[m], xy[k], overlap_km=overlap_km, max_dstrike=max_dstrike)
            if best is None or r['separation_km'] < best['separation_km']:
                best = dict(r, to_strand=m)
        L = np.linalg.norm(np.diff(xy[k], axis=0), axis=1).sum()
        if best['relation'] == 'end-to-end':
            if best['offset_km'] > max_offset_km:
                best['relation'] = 'stepover'
            elif L > min_branch_km:
                best['relation'] = 'branching'
        best['strand'] = k
        out.append(best)
    return out


def check_segment_relation(fsegments, max_jump_km=20.0, overlap_km=10.0, max_dstrike=30.0,
                           max_offset_km=10.0, same_fault_stepovers=True):
    """Strand-configuration relations of the extra strands to their nearest strand
    (['end-to-end'] if there is only one strand)."""
    strands, _ = _strands_for(fsegments, max_jump_km, overlap_km, max_dstrike, max_offset_km,
                              same_fault_stepovers)
    rels = strand_relations(strands, overlap_km, max_dstrike, max_offset_km)
    return [r['relation'] for r in rels] or ['end-to-end']


# =============================================================== multi-fault classification
def _upper_edge(geom):
    return np.asarray(geom.exterior.coords)[:2]


def classify_rupture(fsegments, rule='weatherill', dip_thresh=50.0, rake_thresh=30.0,
                     gap_thresh_km=10.0, use_dipdir=True, include_configuration=True,
                     max_jump_km=20.0, overlap_km=10.0, max_dstrike=30.0, max_offset_km=10.0,
                     same_fault_stepovers=True, include_parallel=None):
    """
    Classify a rupture as multi-fault or multi-segment.

    Step 1, junction criteria (dip change, rake change, gap) between CONSECUTIVE
    SUBSECTIONS of a strand:
      rule='weatherill' (default; Weatherill & Lilienkamp, 2023, as applied to NZ NSHM 2022):
          a break where |Δdip| > dip_thresh AND |Δrake (1st quadrant)| > rake_thresh
          AND gap > gap_thresh_km. Any such break makes the rupture multi-fault.
      rule='al_atik' (Al Atik et al., 2023, Section 5):
          multi-fault if |Δdip| > dip_thresh AND |Δrake₁Q| > rake_thresh at any pair;
          breaks at those pairs and wherever gap > gap_thresh_km.
    Step 2, strand-configuration criteria (include_configuration=True): an extra strand
      that is parallel to, branching from, stepped over from, or separate from its
      nearest strand also makes the rupture multi-fault and forms its own fault strand(s).

    Dip change: with use_dipdir=True (default) opposite-facing dips are measured from the
      same side (0-180 deg), so 40 deg east next to 40 deg west is a 100 deg change;
      use_dipdir=False gives the plain |dip change| used in the published rules.

    Strands are built by linking segments end to end (see build_strands): segments
    overlapping by more than overlap_km, or stepping sideways by more than
    max_offset_km, are never linked; gaps up to max_jump_km are linked, and longer
    jumps when they continue the strand in line, so a gap alone does not separate strands.
    With same_fault_stepovers=True, an overlapping step-over between segments of the same
    fault (same ParentName before ':') is part of one strand. Each extra strand is compared with its
    nearest strand (see strand_relations).

    Returns dict with:
      is_multifault            : final classification (step 1 or step 2)
      junction_multifault      : step 1 under the chosen rule
      al_atik_multifault, weatherill_multifault : step 1 under each rule, for comparison
      configuration_multifault : step 2
      rule, relations
      (old keys criteria_multifault and parallel_multifault are kept as aliases;
       the old argument include_parallel overrides include_configuration if given)
      sections : ordered sections with strand (contiguous piece), fault_strand,
                 dDip, dRake1Q, gap_km, break
    """
    if rule not in ('weatherill', 'al_atik'):
        raise ValueError("rule must be 'weatherill' or 'al_atik'")
    if include_parallel is not None:                 # old argument name
        include_configuration = include_parallel

    strands, _ = _strands_for(fsegments, max_jump_km, overlap_km, max_dstrike, max_offset_km,
                              same_fault_stepovers)
    relations = strand_relations(strands, overlap_km, max_dstrike, max_offset_km)

    # sections in along-strand order
    parts = []
    for sid, (idx, oriented, _) in enumerate(strands):
        for k, o in zip(idx, oriented):
            seg = fsegments[k].copy()
            mids = np.array([_upper_edge(g).mean(axis=0) for g in _geoms(seg)])
            lon0, lat0 = mids.mean(axis=0)
            pos = to_km(mids, lon0, lat0) @ (o[-1] - o[0])           # along oriented segment
            seg = seg.iloc[np.argsort(pos)]
            seg['strand'] = sid
            parts.append(seg)
    sec = pd.concat(parts, ignore_index=True)

    # consecutive-subsection changes within each strand
    edges = [_upper_edge(g) for g in _geoms(sec)]
    first_in_strand = sec['strand'].ne(sec['strand'].shift()).to_numpy()
    d_dip, d_rake, gap = (np.full(len(sec), np.nan) for _ in range(3))
    for i in range(1, len(sec)):
        if first_in_strand[i]:
            continue
        a, b = sec.iloc[i - 1], sec.iloc[i]
        d_dip[i] = dip_change(a['DipDeg'], a['DipDir'], b['DipDeg'], b['DipDir'], use_dipdir)
        d_rake[i] = abs(first_quadrant_rake(b['Rake']) - first_quadrant_rake(a['Rake']))
        gap[i] = segment_min_distance_haversine(edges[i - 1], edges[i])
    sec['dDip'], sec['dRake1Q'], sec['gap_km'] = d_dip, d_rake, gap

    angle = (sec['dDip'] > dip_thresh) & (sec['dRake1Q'] > rake_thresh)
    wide_gap = sec['gap_km'] > gap_thresh_km
    weatherill = bool((angle & wide_gap).any())
    al_atik = bool(angle.any())

    configuration = include_configuration and len(relations) > 0 and any(
        r['relation'] in ('parallel', 'branching', 'stepover', 'separate') for r in relations)
    junction = weatherill if rule == 'weatherill' else al_atik
    is_mf = junction or configuration

    # breaks within strands, then a new fault strand at each break and each remaining strand
    brk = (angle & wide_gap) if rule == 'weatherill' else (angle | wide_gap)
    brk = brk & is_mf
    sec['break'] = brk
    fstrand = ((brk | pd.Series(first_in_strand)).cumsum() - 1) if is_mf else pd.Series(0, index=sec.index)

    # a strand that only continues or spurs off another ('end-to-end') is part of the
    # fault strand it touches: take that of its nearest section on the other strand
    if is_mf:
        for r in relations:
            if r['relation'] != 'end-to-end':
                continue
            mine = np.where(sec['strand'] == r['strand'])[0]
            other = np.where(sec['strand'] == r['to_strand'])[0]
            d = [min(segment_min_distance_haversine(edges[i], edges[j]) for i in mine) for j in other]
            fstrand.iloc[mine] = fstrand.iloc[other[int(np.argmin(d))]]
        fstrand = pd.Series(pd.factorize(fstrand)[0], index=sec.index)
    sec['fault_strand'] = fstrand

    return dict(is_multifault=is_mf,
                junction_multifault=junction,
                configuration_multifault=bool(configuration),
                al_atik_multifault=al_atik, weatherill_multifault=weatherill,
                rule=rule, relations=relations, sections=sec,
                # old key names, kept so existing code keeps working
                criteria_multifault=junction, parallel_multifault=bool(configuration))


def check_multifault(fsegments, **kwargs):
    """True if the rupture is multi-fault (see classify_rupture for options)."""
    return classify_rupture(fsegments, **kwargs)['is_multifault']


# =============================================================== splines
def spline_xy(xy, smooth_km=1.0, spacing_km=1.0):
    """Parametric smoothing spline through ordered km points, resampled every spacing_km.
    Returns (points, original_length_km, smoothed_length_km)."""
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.01]
    xy = xy[keep]
    seglen = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    u = np.r_[0, np.cumsum(seglen)] / seglen.sum()
    tck, _ = splprep(xy.T, u=u, k=min(3, len(xy) - 1), s=len(xy) * smooth_km ** 2)
    dense = np.array(splev(np.linspace(0, 1, 3000), tck)).T
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    s_out = np.linspace(0, s[-1], max(2, int(round(s[-1] / spacing_km)) + 1))
    out = np.column_stack([np.interp(s_out, s, dense[:, 0]), np.interp(s_out, s, dense[:, 1])])
    return out, float(seglen.sum()), float(s[-1])


def _ordered_edge_points(sections):
    """Upper-edge vertices of ordered sections, each edge oriented to continue the previous one."""
    edges = [_upper_edge(g) for g in _geoms(sections)]
    if len(edges) > 1:                                  # orient first edge towards the second
        nxt = edges[1].mean(axis=0)
        if np.linalg.norm(edges[0][0] - nxt) < np.linalg.norm(edges[0][1] - nxt):
            edges[0] = edges[0][::-1]
    for i in range(1, len(edges)):
        prev_end = edges[i - 1][-1]
        if np.linalg.norm(edges[i][1] - prev_end) < np.linalg.norm(edges[i][0] - prev_end):
            edges[i] = edges[i][::-1]
    # at a junction between segments, drop the leading points of the next segment that
    # lie behind the end of the previous one (an overlapping step-over), so the spline
    # jumps across the step instead of looping back; points within a segment are kept
    pid = sections['ParentID'].to_numpy()
    lon0, lat0 = np.vstack(edges).mean(axis=0)
    blocks, start = [], 0
    for i in range(1, len(edges) + 1):
        if i == len(edges) or pid[i] != pid[start]:
            blocks.append(to_km(np.vstack(edges[start:i]), lon0, lat0))
            start = i
    out = [blocks[0]]
    for blk in blocks[1:]:
        prev = np.vstack(out)
        prev_blk = out[-1]
        d = prev_blk[-1] - prev_blk[0]
        ahead = (blk - prev[-1]) @ d >= 0
        first = int(np.argmax(ahead)) if ahead.any() else len(blk)
        out.append(blk[first:])
    return to_lonlat(np.vstack([o for o in out if len(o)]), lon0, lat0)


def rupture_spline_traces(fsegments, smooth_km=1.0, spacing_km=1.0, result=None,
                          M_rupture=None, **classify_kwargs):
    """
    Smoothed trace for each fault strand of a rupture, consistent with classify_rupture.
    A fault strand that contains more than one contiguous piece (a parallel strand when
    include_configuration=False) gets one trace per piece, so no spline cuts between strands.

    result    : an existing classify_rupture() result (otherwise it is computed)
    M_rupture : rupture magnitude; if given, each fault strand's magnitude is added to
                info (total moment split by fault-plane area)

    Returns (traces, info):
      traces : one row per trace point: fault_strand, strand, dist_along_km, lon, lat
      info   : one row per trace piece: fault_strand, strand, segments,
               length_original_km, length_smoothed_km,
               strand_area_km2, strand_area_frac[, strand_M]   (per fault strand)
    """
    if result is None:
        result = classify_rupture(fsegments, **classify_kwargs)
    sec = result['sections']

    rows, info = [], []
    for (fsid, sid), grp in sec.groupby(['fault_strand', 'strand'], sort=True):
        pts = _ordered_edge_points(grp)
        lon0, lat0 = pts.mean(axis=0)
        sm, L0, L1 = spline_xy(to_km(pts, lon0, lat0), smooth_km, spacing_km)
        ll = to_lonlat(sm, lon0, lat0)
        rows.append(pd.DataFrame({'fault_strand': fsid, 'strand': sid,
                                  'dist_along_km': np.linspace(0, L1, len(ll)),
                                  'lon': ll[:, 0], 'lat': ll[:, 1]}))
        info.append({'fault_strand': fsid, 'strand': sid,
                     'segments': ' -> '.join(dict.fromkeys(grp['ParentName'])),
                     'length_original_km': L0, 'length_smoothed_km': L1})
    traces, info = pd.concat(rows, ignore_index=True), pd.DataFrame(info)

    # per-fault-strand area (and magnitude) from fault-plane areas
    area = pd.Series(section_area_km2(sec), index=sec.index).groupby(sec['fault_strand']).sum()
    info['strand_area_km2'] = info['fault_strand'].map(area)
    info['strand_area_frac'] = info['strand_area_km2'] / area.sum()
    if M_rupture is not None:
        info['strand_M'] = strand_magnitude(M_rupture, info['strand_area_km2'], area.sum())
    return traces, info


# =============================================================== plotting
def plot_fsection(fseg, color=None, ax=None, fig=None):
    """Plot a segment's section polygons, with upper edges in thick black."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    xx, yy = [], []
    for geom in _geoms(fseg):
        x, y = geom.exterior.xy
        xx, yy = xx + list(x), yy + list(y)
        ax.plot(x, y, '-', color=color) if color is not None else ax.plot(x, y, '-')
        ax.plot(x[:2], y[:2], '-', color='k', linewidth=2)
    return xx, yy


def plot_fsegements(fsegs, is_cmap=False, ax=None, fig=None, cmap_name='rainbow'):
    """Plot all segments of a rupture with their names listed in the corner."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    n = len(fsegs)
    colors = plt.colormaps[cmap_name](np.linspace(0, 1, n)) if is_cmap else ['b'] * n

    fnames, xxx, yyy = [], [], []
    for seg, c in zip(fsegs, colors):
        for name in seg['ParentName']:
            if name not in fnames:
                fnames.append(name)
        xx, yy = plot_fsection(seg, color=c, ax=ax, fig=fig)
        xxx, yyy = xxx + xx, yyy + yy

    for k, name in enumerate(fnames):
        ax.text(0.05, 0.95 - 0.04 * k, name, transform=ax.transAxes, fontsize=9)

    ax.set_xlim(min(xxx) - 0.5, max(xxx) + 0.5)
    ax.set_ylim(min(yyy) - 0.5, max(yyy) + 0.5)
    ax.set_xlabel('Longitude (\u00B0E)')
    ax.set_ylabel('Latitude (\u00B0N)')
    return fig, ax


#--------------------------------------------------------------
def rake_classification(rake):
    # strikeslip - 0, reverse= -1, normal = 1
    # Bayless et al. 2025
    if ((rake>=-180) & (rake<=-150)) | ((rake>=-30) & (rake<=30)) | ((rake>=150) & (rake<=180)): 
        return ('strikeslip')
    else:
        return('Not implemented')
#--------------------------------------------------------------

def plot_faults(dataframe, rupture_sections, fault_sections,
                ax=None, fig=None, doplot_rake=True,
                xlim=(165, 179), ylim=(-47, -34), cmap=plt.cm.twilight):
    """
    Plot the sections of all ruptures in `dataframe` (needs 'Rupture Index').
    doplot_rake=True colours sections by rake (range from 'Average Rake (degrees)'
    when present, else from the section rakes) and adds a colour bar;
    doplot_rake=False draws plain outlines.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    fig = ax.figure

    section_ids = rupture_sections.loc[
        rupture_sections['rupture'].isin(dataframe['Rupture Index']), 'section'].unique()
    sel = fault_sections[fault_sections['FaultID'].isin(section_ids)]

    verts, vals = [], []
    for geom, v in zip(_geoms(sel), sel['Rake']):
        parts = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for p in parts:
            verts.append(np.asarray(p.exterior.coords))
            vals.append(v)

    if doplot_rake:
        rng = dataframe['Average Rake (degrees)'] if 'Average Rake (degrees)' in dataframe else sel['Rake']
        norm = mpl.colors.Normalize(vmin=rng.min(), vmax=rng.max())
        pc = PolyCollection(verts, cmap=cmap, norm=norm, edgecolor='black', linewidth=0.5)
        pc.set_array(np.asarray(vals))
    else:
        pc = PolyCollection(verts, facecolor='none', edgecolor='black', linewidth=0.5)
    ax.add_collection(pc)
    ax.autoscale_view()

    if doplot_rake:
        cax = ax.inset_axes([0.05, 0.55, 0.015, 0.3])   # [x0, y0, width, height] in axes fraction
        cbar = fig.colorbar(pc, cax=cax)
        cbar.set_label('Rake Angle')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Longitude (\u00B0E)')
    ax.set_ylabel('Latitude (\u00B0N)')
    return fig, ax


def plot_point(ax, xy):
    ax.plot(xy['lon'], xy['lat'], 'o')

#---------------------------
def get_ruptures_inpoly(data_set, rupture_sections, fault_sections, geojson_file):
    # 1. Make fault_sections a proper GeoDataFrame (parses WKT if needed)
    fs = fault_sections.copy()
    g = np.asarray(fs['geometry'])
    if isinstance(g[0], str):
        fs['geometry'] = shapely.from_wkt(g)
    fs = gpd.GeoDataFrame(fs, geometry='geometry', crs='EPSG:4326')  # assuming lon/lat

    # 2. Define the region (lon, lat), or load it from a file
    #region = Polygon([(174.5, -41.5), (175.5, -41.5), (175.5, -40.8), (174.5, -40.8)])
    region = gpd.read_file(geojson_file).to_crs(fs.crs).union_all()

    # 3. Spatial test per section
    fs['hit'] = fs.intersects(region)    # crosses, touches, overlaps or lies inside
    fs['inside'] = fs.within(region)     # completely inside

    # 4. Roll up to ruptures
    m = (
        rupture_sections[['rupture', 'section']].drop_duplicates()
        .merge(fs[['FaultID', 'hit', 'inside']], left_on='section', right_on='FaultID')
    )
    rel = m.groupby('rupture').agg(
        n_sections=('section', 'size'),
        n_hit=('hit', 'sum'),
        n_inside=('inside', 'sum'),
    )
    rel['relation'] = np.select(
        [rel['n_inside'] == rel['n_sections'], rel['n_hit'] > 0],
        ['within', 'crosses'],
        default='outside',
    )

    # 5. Select ruptures
    _ruptures = (
        data_set
            .join(rel[['relation', 'n_hit', 'n_sections']], on='Rupture Index')
            .query("relation != 'outside'")
    )
    return _ruptures

def plot_singlepoly(geojson_file, ax, color='red', linewidth=1.5):
    """Outline of the (merged) polygon(s) in a GeoJSON file, in lon/lat."""
    region = gpd.read_file(geojson_file).to_crs('EPSG:4326').union_all()
    parts = region.geoms if region.geom_type == 'MultiPolygon' else [region]
    for p in parts:
        x, y = p.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth)
