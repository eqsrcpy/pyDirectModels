import numpy as np
import matplotlib as plt
import shapely 

# need to put some of functions from itfm_crust here
def haversine_km(lon1, lat1, lon2, lat2, EARTH_RADIUS_KM = 6371.0088):
    """Great-circle distance in km; inputs in degrees, broadcastable."""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
#-------------------------------------------------------------
def segment_min_distance_haversine(pts_a, pts_b):
    """Minimum great-circle distance (km) between two sets of lon/lat points."""
    d = haversine_km(pts_a[:, None, 0], pts_a[:, None, 1],
                     pts_b[None, :, 0], pts_b[None, :, 1])
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(d[i, j]) # pts_a[i], pts_b[j]

#-------------------------------------------------------------
def distance_between_two_segments(seg1, seg2):
    def get_pts(sects):
        geoms = np.asarray(sects['geometry'])
        if isinstance(geoms[0], str):
            geoms = shapely.from_wkt(geoms)   
        pts = np.vstack([np.asarray(g.exterior.coords)[:2] for g in geoms]) 
        return pts
        
    pts1 = get_pts(seg1)
    pts2 = get_pts(seg2)
    return segment_min_distance_haversine(pts1, pts2)

#-------------------------------------------------------------
def sort_fsegments_WENS(fsegs):
    def segment_point(seg):
        """Mean of the upper-edge midpoints of a segment's sections (lon, lat)."""
        g = np.asarray(seg['geometry'])
        if isinstance(g[0], str):
            g = shapely.from_wkt(g)
        return np.mean([np.asarray(x.exterior.coords)[:2].mean(axis=0) for x in g], axis=0)

    seg_pts = np.array([segment_point(seg) for seg in fsegs])   # (n_segments, 2)

    order = np.lexsort((-seg_pts[:, 1], seg_pts[:, 0]))   # west -> east, then north -> south
    fsegments_sorted = [fsegs[i] for i in order]
    return fsegments_sorted

#--------------------------------------------------------
def plot_fsection(fseg, color = None, ax=None, fig=None):
    if ax is None:
       fig, ax = plt.subplots(figsize=(6,6))
    geoms = np.asarray(fseg['geometry'])
    if isinstance(geoms[0], str):
       geoms = shapely.from_wkt(geoms)   
    
    xx, yy = [],[]
    for geom in geoms:
        x,y = geom.exterior.xy
        xx, yy = xx+list(x), yy+list(y)
        if color is None:
           ax.plot(x,y, '-')
        else:
           ax.plot(x,y, '-', color=color)
        ax.plot(x[:2],y[:2], '-', color='k', linewidth=2)   
    return xx, yy
#--------------------------------------------------------
def plot_fsegements(fsegs, is_cmap=False, ax=None, fig=None):
   if ax is None:
       fig, ax = plt.subplots(figsize=(6,6))
    
   nsegments = len(fsegs)
   fnames,xxx, yyy = [],[],[]
   if is_cmap:
        #colors = plt.colormaps['turbo'].resampled(nsegments).colors
        cmap = plt.colormaps['rainbow']
        colors = cmap(np.linspace(0, 1, nsegments)) 
        for k in range(nsegments):
           fnames = fnames + list(fsegs[k]['ParentName'])
           xx, yy = plot_fsection(fsegs[k], color = colors[k], ax=ax, fig=fig)
           xxx, yyy = xxx+xx, yyy+yy 
   else:
        for k in range(nsegments):
           fnames = fnames + list(fsegs[k]['ParentName'])
           xx, yy  = plot_fsection(fsegs[k], color = 'b', ax=ax, fig=fig)
           xxx, yyy = xxx+xx, yyy+yy
    
   fnames = list(set(fnames))
   k=0
   for name in fnames:
        ax.text(0.05,0.95-k, name, transform=ax.transAxes, fontsize=9)
        k=k+0.04
    
   ax.set_xlim(min(xxx)-0.5,max(xxx)+0.5)
   ax.set_ylim(min(yyy)-0.5,max(yyy)+0.5)
   ax.set_xlabel('Longitude (\u00B0E)')
   ax.set_ylabel('Latitude (\u00B0N)')
#-------------------------------------------------------------
def dip_differnce_two_segments(seg1, seg2):
    dip1 = float(np.mean(list(seg1['DipDeg'])))
    dipdir1 = float(np.mean(list(seg1['DipDir'])))             
    dip2 = float(np.mean(list(seg2['DipDeg'])))
    dipdir2 = float(np.mean(list(seg2['DipDir'])))

    # treating opposite-facing dips as measured from the same side.
    ddir = abs((dipdir2 - dipdir1 + 180) % 360 - 180) 
    # dip-direction difference, 0–180°
    if ddir > 90: dip2 = 180 - dip2
        
    return abs(dip2 - dip1)
#-------------------------------------------------------------
def first_quadrant_rake(rake):
    """Fold a rake angle into the first quadrant [0, 90]°:
    0 = pure strike-slip, 90 = pure dip-slip."""
    r = abs((rake + 180) % 360 - 180)   # wrap to [-180, 180], take magnitude -> [0, 180]
    return 180 - r if r > 90 else r

#-------------------------------------------------------------
def rake_differnce_two_segments(seg1, seg2):
    rake1 = float(np.mean(list(seg1['Rake'])))
    rake2 = float(np.mean(list(seg2['Rake'])))
    return abs(first_quadrant_rake(rake2) - first_quadrant_rake(rake1))

#-------------------------------------------------------------
def get_fsegments(rupindex, rupture_sections, fault_sections, do_sort_WENS=True):
    rsections = rupture_sections[rupture_sections['rupture']==rupindex]['section'].tolist()
    # get fault sections descrp. involved in the rupture 
    sel_fsections = fault_sections[fault_sections['FaultID'].isin(rsections)]   
    parent_ids = list(set(list(sel_fsections['ParentID'])))
    fsegments = []
    for pid in parent_ids:
        fseg = sel_fsections[sel_fsections['ParentID']==pid]
        fsegments.append(fseg)
    if do_sort_WENS:
        fsegments = sort_fsegments_WENS(fsegments)
    return fsegments
    
#-------------------------------------------------------------
def check_multifault(fsegments):
    nseg = len(fsegments)
    is_multifault = False    
    for k in range(nseg-1):  
        dist_km = distance_between_two_segments(fsegments[k], fsegments[k+1])
        dip_diff = dip_differnce_two_segments(fsegments[k], fsegments[k+1])
        rake_diff = rake_differnce_two_segments(fsegments[k], fsegments[k+1])
        if (dist_km > 10) & (dip_diff>50) & (rake_diff>30):
            is_multifault = True    
            break
    return is_multifault
#-------------------------------------------------------------
def segment_trace(seg):
    """Upper-edge vertices (lon, lat) of all sections in a segment."""
    g = np.asarray(seg['geometry'])
    if isinstance(g[0], str):
        g = shapely.from_wkt(g)
    return np.vstack([np.asarray(x.exterior.coords)[:2] for x in g])

#-------------------------------------------------------------
def segment_relation(trace_a, trace_b, overlap_km=5.0, overlap_frac=0.2, max_dstrike=30.0):
    """
    Geometric relation between two consecutive segments:
      'end-to-end' : B follows A (subsequent)
      'parallel'   : B overlaps A along strike at a similar strike (step-over / en echelon)
      'branching'  : B overlaps A along strike at a different strike (splay / branch)
    """
    pts = np.vstack([trace_a, trace_b])
    lat0, lon0 = np.deg2rad(pts[:, 1].mean()), pts[:, 0].mean()
    def to_km(p, R = 6371.0088):   # local tangent plane in km, accurate over a few hundred km
        return np.column_stack([np.deg2rad(p[:, 0] - lon0) * R * np.cos(lat0),
                                np.deg2rad(p[:, 1]) * R])
    A, B = to_km(trace_a), to_km(trace_b)

    def axis(P):    # end-to-end axis from the farthest pair of points
        d = np.linalg.norm(P[:, None] - P[None], axis=-1)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        v = P[j] - P[i]
        return P[i], v / np.linalg.norm(v), np.linalg.norm(v)

    a0, ua, La = axis(A)
    _, ub, Lb = axis(B)

    s = (B - a0) @ ua                                    # B's positions along A's strike
    ov = max(0.0, min(La, s.max()) - max(0.0, s.min()))  # along-strike overlap
    dstrike = np.degrees(np.arccos(min(1.0, abs(ua @ ub))))   # 0–90°, ignores direction
    sep = np.min(np.linalg.norm(A[:, None] - B[None], axis=-1))

    if not (ov > overlap_km and ov > overlap_frac * min(La, Lb)):
        rel = 'end-to-end'
    elif dstrike <= max_dstrike:
        rel = 'parallel'
    else:
        rel = 'branching'
    return rel
    #return dict(overlap_km=ov, separation_km=sep, dstrike_deg=dstrike,
    #            length_a_km=La, length_b_km=Lb, relation=rel)

#-------------------------------------------------------------
def check_segment_relation(fsegments):
    nseg = len(fsegments)
    relations = []
    for k in range(nseg-1):  
        rel = segment_relation(segment_trace(fsegments[k]), segment_trace(fsegments[k+1]),
                        overlap_km=5.0, overlap_frac=0.2, max_dstrike=30.0)
        relations.append(rel)
        
    return list(set(relations))

