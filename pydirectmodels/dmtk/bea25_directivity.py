"""
bea25_directivity — Bea25 directivity amplification and phi_red for a rupture
=============================================================================

Computes, at each site, the Bea25 (Bayless et al., 2024) directivity:
    amp     = exp(fD)   median directivity amplification
    phi_red = PhiRed    within-event sigma reduction
for a rupture given as one or more fault strands (trace polylines) + sites.

Key features
------------
* Only sites within the directivity footprint (Rrup <= Rmax(M) + margin) are
  computed; the rest are directivity-neutral and omitted by default.
* Per-strand magnitude / rake / Ztor can be read straight from a `traceinfo`
  DataFrame (as written by dmtk.rupture_spline_traces).
* Multi-fault: per-strand hypocentres by the closest-approach cascade; strands
  combined at each site by the max over fD (Bea24 multi-fault rule). Reverse/
  normal strands and strands with M < 6 are skipped (printed) as neutral.
* `directivity_grid()` evaluates on a regular grid (for contouring);
  `plot_contour()` draws it with the RdBu_r colormap.
* OpenQuake hand-off: stable coordinate-based `custom_site_id` + `rupture_id`.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from pydirectmodels import Bea25Model, gc2

KM_PER_DEG = 111.195
M_MIN, M_MAX = 6.0, 8.0        # Bea24 applicable magnitude range (M6.0-8.0)


# --------------------------------------------------------------------------- ids / classify
def site_id_from_coords(lon, lat, ndp=4):
    """Stable per-site id from coordinates (same lon/lat -> same id, any order)."""
    lon = np.round(np.asarray(lon, float), ndp)
    lat = np.round(np.asarray(lat, float), ndp)
    def f(v):
        return ("m" if v < 0 else "p") + f"{abs(v):0{3+1+ndp}.{ndp}f}".replace(".", "")
    return np.array([f(a) + "_" + f(b) for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat))])


def is_strike_slip(rake):
    """Bea24 strike-slip test: rake in [-180,-150] u [-30,30] u [150,180]."""
    r = abs(float(rake)) % 360.0
    return (r <= 30.0) or (r >= 150.0)


def rmax_of_M(M):
    """Bea24 footprint radius Rmax(M): 80 km for M>=7, 20M-60 for 6<=M<7."""
    M = min(float(M), M_MAX)
    return 80.0 if M >= 7.0 else max(20.0 * M - 60.0, 0.0)


# --------------------------------------------------------------------------- geometry
def _project(lon, lat, lon0, lat0):
    x = (np.asarray(lon, float) - lon0) * np.cos(np.radians(lat0)) * KM_PER_DEG
    y = (np.asarray(lat, float) - lat0) * KM_PER_DEG
    return x, y


def _build_ftrace(xy):
    seg = np.diff(xy, axis=0)
    l = np.hypot(seg[:, 0], seg[:, 1])
    strike = np.degrees(np.arctan2(seg[:, 0], seg[:, 1])) % 360.0
    return {'trace': xy, 'strike': strike, 'l': l}


def affected_mask(trace_df, sites_df, rmax, margin=5.0):
    """Boolean mask: sites within (rmax + margin) km of any strand trace.

    Horizontal distance to the (1 km-dense) trace vertices -- a tight screen for
    vertical, surface-rupturing faults (Rrup ~ distance to trace). Over-includes
    slightly (safe); the model zeroes directivity beyond each strand's own Rmax.
    """
    lon0, lat0 = float(trace_df['lon'].mean()), float(trace_df['lat'].mean())
    tx, ty = _project(trace_df['lon'], trace_df['lat'], lon0, lat0)
    sx, sy = _project(sites_df['lon'], sites_df['lat'], lon0, lat0)
    d, _ = cKDTree(np.column_stack([tx, ty])).query(np.column_stack([sx, sy]))
    return d <= (rmax + margin)


# --------------------------------------------------------------------------- hypocentres
def assign_strand_hypocenters(strands, primary_id, primary_frac):
    """Hypocentre on every strand by the closest-approach cascade (see docstring)."""
    xy, dist = strands[primary_id]
    L = dist[-1]
    hx = np.interp(primary_frac * L, dist, xy[:, 0])
    hy = np.interp(primary_frac * L, dist, xy[:, 1])
    assigned = {primary_id: dict(hypo_xy=np.array([hx, hy]), frac=float(primary_frac),
                                 from_strand=None, jump_km=0.0)}
    remaining = [s for s in strands if s != primary_id]
    while remaining:
        best = None
        for s in remaining:
            s_xy, _ = strands[s]
            for a in assigned:
                a_xy, _ = strands[a]
                D = cdist(s_xy, a_xy)
                i = int(D.min(axis=1).argmin())
                d = float(D[i].min())
                if best is None or d < best[0]:
                    best = (d, s, i, a)
        d, s, i, a = best
        s_xy, s_dist = strands[s]
        assigned[s] = dict(hypo_xy=s_xy[i], frac=float(s_dist[i] / s_dist[-1]),
                           from_strand=a, jump_km=d)
        remaining.remove(s)
    return assigned


# --------------------------------------------------------------------------- model
def _strand_directivity(xy, sx, sy, mag, rake, imt, hypo_xy, ztor, model):
    """fD, PhiRed, U, T, R for one strand with hypocentre at hypo_xy [km]."""
    ft = [_build_ftrace(xy)]
    type_ = {'epi': [float(hypo_xy[0]), float(hypo_xy[1])],
             'po':  [float(hypo_xy[0]), float(hypo_xy[1])], 'str': 'JB'}
    T, U, *_r, nom, Upo, Tpo = gc2(ft, sx, sy, type_, False, False)
    a = nom['a']
    _, Ua, *_ = gc2(ft, a[0, 0], a[0, 1], type_, False, False)
    _, Ub, *_ = gc2(ft, a[1, 0], a[1, 1], type_, False, False)
    Ua, Ub = float(np.ravel(Ua)[0]), float(np.ravel(Ub)[0])
    S1, S2 = min(Ua, Ub), max(Ua, Ub)
    r = model.evaluate(mag=min(mag, M_MAX), gc2u=U, gc2t=T, smax1=S1, smax2=S2,
                       ztor=ztor, rake=rake, imt=imt)
    fDi, phired = np.asarray(r.fd), np.asarray(r['PhiRedi'])
    Ry = np.where(U > S2, U - S2, np.where(U < S1, np.abs(U) - np.abs(S1), 0.0))
    R = np.sqrt(np.asarray(T) ** 2 + Ry ** 2 + ztor ** 2)
    return fDi, phired, np.asarray(U), np.asarray(T), R


def _strand_inputs(traceinfo, strand_mags, strand_rakes, strand_ztors):
    """Merge per-strand dicts from an optional traceinfo DataFrame (explicit dicts win)."""
    sm, sr, sz = dict(strand_mags or {}), dict(strand_rakes or {}), dict(strand_ztors or {})
    if traceinfo is not None:
        ti = traceinfo
        for sid, row in ti.set_index('fault_strand').iterrows():
            if 'strand_M' in ti.columns:
                sm.setdefault(sid, float(row['strand_M']))
            if 'Rake' in ti.columns:
                sr.setdefault(sid, float(row['Rake']))
            if 'Ztor' in ti.columns:
                sz.setdefault(sid, float(row['Ztor']))
    return sm, sr, sz


def compute_directivity(trace_df, sites_df, mag=None, rake=None, imt=None, *,
                        traceinfo=None, ztor=0.0, version=2,
                        strand_mags=None, strand_rakes=None, strand_ztors=None,
                        primary_id=None, primary_frac=0.5, rupture_id=None,
                        margin_km=5.0, return_affected_only=True,
                        verbose=True, return_hypocenters=False):
    """Bea25 amp & phi_red at the sites.

    ``imt`` (spectral period, s) is required. Per-strand magnitude/rake/Ztor come
    from ``traceinfo`` (dmtk) if given, or from ``strand_mags``/``strand_rakes``/
    ``strand_ztors`` dicts, or the scalar ``mag``/``rake``/``ztor``.

    Only sites within Rmax(M)+``margin_km`` of the trace are computed; with
    ``return_affected_only`` (default True) just those rows are returned (the
    rest are directivity-neutral). Set it False to return every input site.
    """
    if imt is None:
        raise ValueError("imt (spectral period in s) is required")
    strand_mags, strand_rakes, strand_ztors = _strand_inputs(
        traceinfo, strand_mags, strand_rakes, strand_ztors)

    # build projected strands + hypocentres
    lon0, lat0 = float(trace_df['lon'].mean()), float(trace_df['lat'].mean())
    strands = {}
    for sid, g in trace_df.groupby('fault_strand'):
        g = g.sort_values('dist_along_km')
        xy = np.column_stack(_project(g['lon'], g['lat'], lon0, lat0))
        strands[sid] = (xy, g['dist_along_km'].to_numpy(float))
    if primary_id is None:
        primary_id = max(strands, key=lambda s: strands[s][1][-1])
    hypos = assign_strand_hypocenters(strands, primary_id, primary_frac)

    # scalar fallbacks
    if mag is None:
        mag = max(strand_mags.values()) if strand_mags else 7.0
    if rake is None:
        rake = strand_rakes.get(primary_id, 180.0)

    # footprint filter: largest Rmax over contributing (SS, M>=6) strands
    contrib_M = [strand_mags.get(s, mag) for s in strands
                 if is_strike_slip(strand_rakes.get(s, rake)) and strand_mags.get(s, mag) >= M_MIN]
    rmax = max((rmax_of_M(m) for m in contrib_M), default=rmax_of_M(mag))
    mask = affected_mask(trace_df, sites_df, rmax, margin_km)

    work = sites_df.loc[mask].reset_index(drop=True)
    lon = work['lon'].to_numpy(float); lat = work['lat'].to_numpy(float)
    sx, sy = _project(lon, lat, lon0, lat0)

    model = Bea25Model(version=version)
    n = len(work); best = np.full(n, -np.inf)
    cols = ('amp', 'phi_red', 'fd', 'gc2u', 'gc2t', 'rrup')
    out = {k: np.zeros(n) for k in cols}; out['strand'] = np.full(n, -1, int)
    for sid, (xy, _dist) in strands.items():
        m = strand_mags.get(sid, mag); rk = strand_rakes.get(sid, rake); zt = strand_ztors.get(sid, ztor)
        tag = f"rupture {rupture_id} " if rupture_id is not None else ""
        if not is_strike_slip(rk):
            if verbose:
                print(f"  [skip] {tag}strand {sid}: rake {rk:.0f} is not strike-slip "
                      f"(Bea24 is strike-slip only) -> treated as neutral")
            continue
        if m < M_MIN:
            if verbose:
                print(f"  [skip] {tag}strand {sid}: M {m:.2f} < {M_MIN:.1f} "
                      f"(below Bea24 range M{M_MIN:.0f}-{M_MAX:.0f}) -> treated as neutral")
            continue
        if n == 0:
            continue
        fd, phir, U, T, R = _strand_directivity(xy, sx, sy, m, rk, imt,
                                                hypos[sid]['hypo_xy'], zt, model)
        take = fd > best                         # MAX RULE
        best[take] = fd[take]
        out['fd'][take] = fd[take]; out['amp'][take] = np.exp(fd[take])
        out['phi_red'][take] = phir[take]
        out['gc2u'][take] = U[take]; out['gc2t'][take] = T[take]; out['rrup'][take] = R[take]
        out['strand'][take] = sid
    out['amp'][out['amp'] == 0.0] = 1.0

    def _assemble(df, arrs, nrow):
        r = df.copy()
        if 'custom_site_id' not in r.columns:
            ids = site_id_from_coords(df['lon'], df['lat'])
            ndup = len(ids) - len(set(ids))
            if ndup and verbose:
                print(f"  [warn] {ndup} duplicate site coordinate(s) share a custom_site_id; "
                      f"dedupe allsites (drop_duplicates(['lon','lat'])) before OpenQuake.")
            r.insert(0, 'custom_site_id', ids)
        if rupture_id is not None:
            r.insert(1, 'rupture_id', rupture_id)
        for k in (*cols, 'strand'):
            r[k] = arrs[k] if nrow else np.array([], dtype=float)
        return r

    if return_affected_only:
        res = _assemble(work, out, n)
    else:                                        # all sites: neutral fill for the rest
        full = {k: (np.ones(len(sites_df)) if k == 'amp' else np.zeros(len(sites_df)))
                for k in cols}
        full['strand'] = np.full(len(sites_df), -1, int)
        idx = np.flatnonzero(mask.to_numpy() if hasattr(mask, 'to_numpy') else mask)
        for k in (*cols, 'strand'):
            full[k][idx] = out[k]
        res = _assemble(sites_df, full, len(sites_df))
    return (res, hypos) if return_hypocenters else res


# --------------------------------------------------------------------------- grid / contour
def directivity_grid(trace_df, imt, *, traceinfo=None, mag=None, rake=None, ztor=0.0,
                     version=2, strand_mags=None, strand_rakes=None, strand_ztors=None,
                     primary_id=None, primary_frac=0.5, spacing_km=2.0, pad_km=None,
                     rupture_id=None, verbose=False):
    """Evaluate Bea25 directivity on a regular lon/lat grid covering the footprint.

    Returns a DataFrame with lon, lat, amp, phi_red, ... on the grid (for
    contouring). ``spacing_km`` sets the grid step; ``pad_km`` the margin beyond
    the fault bounding box (default: Rmax, so the whole footprint is covered).
    """
    sm, sr, sz = _strand_inputs(traceinfo, strand_mags, strand_rakes, strand_ztors)
    m_for_rmax = max(sm.values()) if sm else (mag or 7.0)
    pad = pad_km if pad_km is not None else rmax_of_M(m_for_rmax)
    lat0 = float(trace_df['lat'].mean())
    dlon = spacing_km / (KM_PER_DEG * np.cos(np.radians(lat0)))
    dlat = spacing_km / KM_PER_DEG
    lons = np.arange(trace_df['lon'].min() - pad * dlon / spacing_km,
                     trace_df['lon'].max() + pad * dlon / spacing_km + dlon, dlon)
    lats = np.arange(trace_df['lat'].min() - pad * dlat / spacing_km,
                     trace_df['lat'].max() + pad * dlat / spacing_km + dlat, dlat)
    GLon, GLat = np.meshgrid(lons, lats)
    grid = pd.DataFrame({'lon': GLon.ravel(), 'lat': GLat.ravel()})
    return compute_directivity(trace_df, grid, mag=mag, rake=rake, imt=imt,
                               traceinfo=traceinfo, ztor=ztor, version=version,
                               strand_mags=sm, strand_rakes=sr, strand_ztors=sz,
                               primary_id=primary_id, primary_frac=primary_frac,
                               rupture_id=rupture_id, return_affected_only=False,
                               verbose=verbose)


def plot_contour(grid_df, trace_df=None, hypos=None, column='amp',
                 vmin=0.8, vmax=1.2, nlevels=25, save=None, title=None, ax=None):
    """Filled contour of a gridded directivity field, using the RdBu_r colormap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G = grid_df.pivot_table(index='lat', columns='lon', values=column)
    Lon, Lat = np.meshgrid(G.columns.values, G.index.values)
    Z = G.values
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 8))
    cf = ax.contourf(Lon, Lat, np.clip(Z, vmin, vmax),
                     levels=np.linspace(vmin, vmax, nlevels), cmap='RdBu_r', extend='both')
    lat0 = float(trace_df['lat'].mean()) if trace_df is not None else float(np.mean(Lat))
    if trace_df is not None:
        for sid, g in trace_df.groupby('fault_strand'):
            g = g.sort_values('dist_along_km')
            ax.plot(g['lon'], g['lat'], 'k-', lw=2)
        lon0 = float(trace_df['lon'].mean())
        if hypos is not None:
            for sid, h in hypos.items():
                hx, hy = h['hypo_xy']
                ax.plot(lon0 + hx / (KM_PER_DEG * np.cos(np.radians(lat0))),
                        lat0 + hy / KM_PER_DEG, 'r*', ms=14, mec='w')
    ax.set_aspect(1.0 / np.cos(np.radians(lat0)))
    ax.set_xlabel('lon'); ax.set_ylabel('lat')
    if title:
        ax.set_title(title)
    if created:
        fig.colorbar(cf, ax=ax, label=column, fraction=0.046, pad=0.04)
        if save:
            fig.savefig(save, dpi=130, bbox_inches='tight')
        return fig
    return cf


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    up = "/mnt/user-data/uploads"
    trace = pd.read_csv(f"{up}/179359_traces.csv")
    info = pd.read_csv(f"{up}/179359_traceinfo.csv")
    sites = pd.read_csv(f"{up}/allsites.csv"); sites.columns = [c.strip().strip('"') for c in sites.columns]

    print("=== 179359: scattered sites, inputs from traceinfo, affected only ===")
    out, hyp = compute_directivity(trace, sites, imt=3.0, traceinfo=info,
                                   primary_id=0, primary_frac=0.5, rupture_id=179359,
                                   return_hypocenters=True)
    print(f"  affected sites returned: {len(out)} of {len(sites)}  "
          f"amp [{out['amp'].min():.3f}, {out['amp'].max():.3f}]")

    print("\n=== gridded field + RdBu_r contour ===")
    grid = directivity_grid(trace, imt=3.0, traceinfo=info, primary_id=0,
                            primary_frac=0.5, spacing_km=2.0, rupture_id=179359)
    print(f"  grid points: {len(grid)}  amp [{grid['amp'].min():.3f}, {grid['amp'].max():.3f}]")
    plot_contour(grid, trace, hypos=hyp, save="179359_contour.png",
                 title="Rupture 179359 — Bea25 directivity SA(3s)")
    print("  wrote 179359_contour.png")


def plot_traces(trace_df, hypos=None, traceinfo=None, ax=None, save=None, title=None,
                strand_colors=None, label_strands=True):
    """Plot fault-strand traces with per-strand hypocentres as stars.

    trace_df  : columns fault_strand, strand, dist_along_km, lon, lat.
    hypos     : dict strand_id -> {'hypo_xy', ...} from compute_directivity
                (return_hypocenters=True); if None, no stars are drawn.
    traceinfo : optional DataFrame (fault_strand, segments, Rake, ...) used to
                label each strand and dash non-strike-slip (reverse/normal) ones.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 8))

    lon0 = float(trace_df['lon'].mean()); lat0 = float(trace_df['lat'].mean())
    name = rake = {}
    if traceinfo is not None:
        if 'segments' in traceinfo.columns:
            name = dict(zip(traceinfo['fault_strand'], traceinfo['segments']))
        if 'Rake' in traceinfo.columns:
            rake = dict(zip(traceinfo['fault_strand'], traceinfo['Rake']))
    cmap = plt.get_cmap('tab10')

    for i, (sid, g) in enumerate(trace_df.groupby('fault_strand')):
        g = g.sort_values('dist_along_km')
        col = (strand_colors or {}).get(sid, cmap(i % 10))
        ss = is_strike_slip(rake[sid]) if sid in rake else True
        lbl = None
        if label_strands:
            nm = name.get(sid, f"strand {sid}")
            lbl = f"{nm}" + ("" if ss else f" (rake {rake[sid]:.0f}, reverse)")
        ax.plot(g['lon'], g['lat'], '-' if ss else '--', color=col, lw=2.5, label=lbl, zorder=2)
        # start/end ticks
        ax.plot(g['lon'].iloc[0], g['lat'].iloc[0], 'o', color=col, ms=4, zorder=3)
        if hypos is not None and sid in hypos:
            hx, hy = hypos[sid]['hypo_xy']
            hlon = lon0 + hx / (KM_PER_DEG * np.cos(np.radians(lat0)))
            hlat = lat0 + hy / KM_PER_DEG
            ax.plot(hlon, hlat, '*', ms=18, mfc=col, mec='k', mew=1.0, zorder=4)

    ax.set_aspect(1.0 / np.cos(np.radians(lat0)))
    ax.set_xlabel('lon'); ax.set_ylabel('lat')
    if title:
        ax.set_title(title)
    if label_strands:
        ax.legend(loc='best', fontsize=8)
    if created and save:
        fig.savefig(save, dpi=130, bbox_inches='tight')
    return ax


def _A_of_T(M, T, Amax, SigG):
    Tpeak = 10.0 ** (-2.15 + 0.404 * M)
    return Amax * np.exp(-(np.log10(np.asarray(T, float) / Tpeak)) ** 2 / (2 * SigG ** 2))


def unknown_hypocenter(trace_df, sites_df, imt, *, traceinfo=None, spacing_km=5.0,
                       n_hypo=None, mag=None, rake=None, ztor=0.0, version=2,
                       strand_mags=None, strand_rakes=None, strand_ztors=None,
                       margin_km=5.0, rupture_id=None, return_realizations=False,
                       verbose=True):
    """Unknown-hypocentre moment modifiers (Flavor A), period-factored.

    ``imt`` may be a single period or a LIST of periods. The period-independent
    geometry (GC2 + centering -> fGprime) is computed once per strand/hypocentre;
    each period is then an analytic A(M,T)*logistic, so an N-period spectrum costs
    ~1 geometry pass, not N. See prior docstring for the Flavor-A / length-weight
    / max-rule semantics.

    Returns affected sites with mu_fd, phi_iuh, amp, phi_red. For a single period:
    one row per site. For a list: long format with a 'period' column.
    """
    periods = np.atleast_1d(imt).astype(float)
    multi = np.ndim(imt) > 0

    strand_mags, strand_rakes, strand_ztors = _strand_inputs(
        traceinfo, strand_mags, strand_rakes, strand_ztors)
    lon0, lat0 = float(trace_df['lon'].mean()), float(trace_df['lat'].mean())
    strands = {}
    for sid, g in trace_df.groupby('fault_strand'):
        g = g.sort_values('dist_along_km')
        xy = np.column_stack(_project(g['lon'], g['lat'], lon0, lat0))
        strands[sid] = (xy, g['dist_along_km'].to_numpy(float))
    if mag is None:
        mag = max(strand_mags.values()) if strand_mags else 7.0
    if rake is None:
        rake = strand_rakes.get(max(strands, key=lambda s: strands[s][1][-1]), 180.0)

    def contributes(sid):
        return is_strike_slip(strand_rakes.get(sid, rake)) and strand_mags.get(sid, mag) >= M_MIN
    nucleators = [s for s in strands if contributes(s)]
    if not nucleators:
        # No strand is strike-slip AND M>=6 -> no Bea25 directivity at all.
        if verbose:
            reasons = []
            for s in strands:
                rk = strand_rakes.get(s, rake); m = strand_mags.get(s, mag)
                why = "not strike-slip" if not is_strike_slip(rk) else f"M {m:.2f}<{M_MIN:.0f}"
                reasons.append(f"strand {s} ({why})")
            print(f"  [note] rupture {rupture_id}: no directivity computed -- "
                  f"{', '.join(reasons)}. Returning None.")
        return None

    def n_for(sid):
        if isinstance(n_hypo, dict): return max(1, int(n_hypo.get(sid, 1)))
        if n_hypo is not None:       return max(1, int(n_hypo))
        return max(1, int(round(strands[sid][1][-1] / spacing_km)))

    rmax = max((rmax_of_M(strand_mags.get(s, mag)) for s in nucleators), default=rmax_of_M(mag))
    mask = affected_mask(trace_df, sites_df, rmax, margin_km)
    work = sites_df.loc[np.asarray(mask)].reset_index(drop=True)
    n = len(work)
    sx, sy = _project(work['lon'], work['lat'], lon0, lat0)
    model = Bea25Model(version=version)
    klog, Amax, SigG = model.k, model.Amax, model.SigG

    def strand_TU(xy, origin_xy):
        ft = [_build_ftrace(xy)]
        ty = {'epi': list(map(float, origin_xy)), 'po': list(map(float, origin_xy)), 'str': 'JB'}
        T, U, *_r, nom, _Up, _Tp = gc2(ft, sx, sy, ty, False, False)
        a = nom['a']
        _, Ua, *_ = gc2(ft, a[0, 0], a[0, 1], ty, False, False)
        _, Ub, *_ = gc2(ft, a[1, 0], a[1, 1], ty, False, False)
        return np.asarray(T), np.asarray(U), float(np.ravel(Ua)[0]), float(np.ravel(Ub)[0])

    def geom(T, U, S1, S2, sid):
        """Period-independent logistic(k*fGprime) and Footprint for a strand."""
        r = model.evaluate(mag=min(strand_mags.get(sid, mag), M_MAX), gc2u=U, gc2t=T,
                           smax1=min(S1, S2), smax2=max(S1, S2),
                           ztor=strand_ztors.get(sid, ztor),
                           rake=strand_rakes.get(sid, rake), imt=float(periods[0]))
        fg = np.asarray(r.extra['predic_funcs']['fGprime'])
        foot = np.asarray(r.extra['other']['Footprint'], float)
        return 2.0 / (1.0 + np.exp(-klog * fg)) - 1.0, foot

    # ---- geometry pass (once): store logistic + M + footprint per realization ----
    foot_any = np.zeros(n)
    realz = []          # each: list of (logistic, M) over contributing strands, + weight
    N_total = sum(n_for(k) for k in nucleators) or 1
    for k in nucleators:
        hypos_k = assign_strand_hypocenters(strands, k, 0.5)
        sec = []                                   # (logistic, M) for secondaries of k
        for j in nucleators:
            if j == k: continue
            Tj, Uj, Uaj, Ubj = strand_TU(strands[j][0], hypos_k[j]['hypo_xy'])
            lg, ft = geom(Tj, Uj, Uaj, Ubj, j)
            sec.append((lg, strand_mags.get(j, mag))); foot_any = np.maximum(foot_any, ft)
        xyk, distk = strands[k]
        Tk, U0, Ua0, Ub0 = strand_TU(xyk, xyk[0])
        Lk = distk[-1]; nk = n_for(k); fracs = (np.arange(nk) + 0.5) / nk
        hx = np.interp(fracs * Lk, distk, xyk[:, 0]); hy = np.interp(fracs * Lk, distk, xyk[:, 1])
        ftp = [_build_ftrace(xyk)]; typ = {'epi': list(xyk[0]), 'po': list(xyk[0]), 'str': 'JB'}
        _, Uh, *_ = gc2(ftp, hx, hy, typ, False, False); Uh = np.asarray(Uh).ravel()
        Mk = strand_mags.get(k, mag)
        for i in range(nk):
            lg, ft = geom(Tk, U0 - Uh[i], Ua0 - Uh[i], Ub0 - Uh[i], k)
            foot_any = np.maximum(foot_any, ft)
            realz.append([(lg, Mk)] + sec)         # all contributing strands this realization

    # ---- period pass (analytic): max over strands, mean/std over realizations ----
    e1T = np.interp(np.log(periods), np.log(model.PhiPer), model.e1)
    rows = []
    w = 1.0 / N_total
    for p_idx, T in enumerate(periods):
        sum_fd = np.zeros(n); sumsq = np.zeros(n)
        for strand_list in realz:
            fd_comb = np.full(n, -np.inf)
            for lg, M in strand_list:
                fd_comb = np.maximum(fd_comb, _A_of_T(M, T, Amax, SigG) * lg)
            sum_fd += w * fd_comb; sumsq += w * fd_comb ** 2
        mu = sum_fd if realz else np.zeros(n)       # wsum == 1
        phi = np.sqrt(np.maximum(sumsq - mu ** 2, 0.0)) if realz else np.zeros(n)
        rows.append((T, mu, phi, e1T[p_idx] * foot_any))

    # ---- assemble ----
    base = work.copy()
    if 'custom_site_id' not in base.columns:
        base.insert(0, 'custom_site_id', site_id_from_coords(work['lon'], work['lat']))
    if rupture_id is not None:
        base.insert(1, 'rupture_id', rupture_id)
    if not multi:
        T, mu, phi, phr = rows[0]
        base['mu_fd'] = mu; base['phi_iuh'] = phi; base['amp'] = np.exp(mu); base['phi_red'] = phr
        return base
    out = []
    for T, mu, phi, phr in rows:
        d = base.copy(); d['period'] = T
        d['mu_fd'] = mu; d['phi_iuh'] = phi; d['amp'] = np.exp(mu); d['phi_red'] = phr
        out.append(d)
    return pd.concat(out, ignore_index=True)
