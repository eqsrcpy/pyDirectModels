"""
GC2 — Python translation of GC2.m
=================================

Conversion to Python (NumPy) of Jeff Bayless' MATLAB implementation of
Brian Chiou's R functions for the calculation of GC2 coordinates.

Reference
---------
Spudich, Paul and Chiou, Brian, 2015, Strike-parallel and strike-normal
coordinate system around geometrically complicated rupture traces for use by
NGA-West2 and further improvements: U.S. Geological Survey Open-File Report
2015-1028, 20 p., https://dx.doi.org/10.3133/ofr20151028.

Original MATLAB by Jeff Bayless (jeff.bayless@aecom.com), May 2020.

Input parameters
----------------
ftraces : list of dict
    One entry per individual rupture strand. For each strand i:
        ftraces[i]['trace']  : (n, 2) array of cartesian X,Y coords [km]
        ftraces[i]['l']      : (n-1,) array of segment lengths [km]
        ftraces[i]['strike'] : (n-1,) array of segment strikes [degrees]
SiteX, SiteY : array-like
    Cartesian site coordinates [km]. If ``gridflag`` is True, U and T are
    computed for every (SiteY, SiteX) combination (shape S_y x S_x); otherwise
    they are computed point-wise (paired) for the S coordinates.
type_ : dict
    type_['str'] : str
        Method for the coordinate shift (Eq 12, 13 in the OFR). Can be
        'NGA2', 'JB', or any other string (defaults to the OFR version).
        If 'JB' then type_['po'] must be provided.
    type_['po']  : (2,) array-like, only used when type_['str'] == 'JB'.
        Shifts the origin (p_origin) of U and T to this location instead of
        placing it at the endpoint of the nominal strike.
discordant : bool
    If True, check/correct segment discordance.
gridflag : bool
    If True, compute on a grid; otherwise element-wise.

Output parameters
-----------------
T, U, W          : GC2 parameters (km, km, unitless), grid or 1-D arrays
reference_axis   : nominal strike direction after discordance correction (2,)
p_origin         : coordinate system origin (2,)
nominal_strike   : dict with keys 'a', 'b', 'e', 'E'
Upo, Tpo         : U, T of coordinate type_['po'] before shift (the applied shift)
"""

import numpy as np


def gc2(ftraces, SiteX, SiteY, type_, discordant, gridflag):
    """Main GC2 driver. Mirrors the MATLAB function GC2()."""

    # (1) nominal strike
    nominal_strike = comp_nominal_strike(ftraces)

    # (2) link traces
    single_trace, reference_axis, p_origin = linktraces(
        ftraces, nominal_strike, type_, discordant
    )

    SiteX = np.atleast_1d(np.asarray(SiteX, dtype=float))
    SiteY = np.atleast_1d(np.asarray(SiteY, dtype=float))

    # (3) compute GC2 for stations
    if gridflag:  # coordinates on a grid defined by SiteX and SiteY
        T = np.zeros((len(SiteY), len(SiteX)))
        U = np.zeros((len(SiteY), len(SiteX)))
        W = np.zeros((len(SiteY), len(SiteX)))
        for ii in range(len(SiteX)):
            for jj in range(len(SiteY)):
                T[jj, ii], U[jj, ii], W[jj, ii] = compute_gc2(
                    SiteX[ii], SiteY[jj], single_trace
                )
    else:  # point-wise at the coordinates paired in SiteX and SiteY
        T = np.zeros(len(SiteX))
        U = np.zeros(len(SiteX))
        W = np.zeros(len(SiteX))
        for ii in range(len(SiteX)):
            T[ii], U[ii], W[ii] = compute_gc2(SiteX[ii], SiteY[ii], single_trace)

    # (4) compute GC2 for type_['po'] and apply the shift to U and T
    if type_.get('str') == 'JB':
        po = np.asarray(type_['po'], dtype=float)
        Tpo, Upo, _ = compute_gc2(po[0], po[1], single_trace)
        U = U - Upo
        T = T - Tpo
    else:
        Upo = 0.0
        Tpo = 0.0

    return T, U, W, reference_axis, p_origin, nominal_strike, Upo, Tpo


# ---------------------------------------------------------------------------
# Step 2 — Compute GC2 nominal strike
# ---------------------------------------------------------------------------
def comp_nominal_strike(ftraces):
    """Calculate the nominal strike and helper parameters used by linktraces."""
    m = len(ftraces)

    # matrix of trace ends, two rows per trace
    trace_ends = np.zeros((m * 2, 2))
    for jj in range(m):
        nseg = len(np.atleast_1d(ftraces[jj]['strike']))
        trace = np.asarray(ftraces[jj]['trace'], dtype=float)
        # first point (index 0) and last point (index nseg)
        trace_ends[2 * jj:2 * jj + 2, :] = trace[[0, nseg], :]

    # find the two endpoints most distant from each other
    n = trace_ends.shape[0]
    maxdist = -1.0
    i1 = i2 = 0
    for ii in range(n - 1):
        for kk in range(ii + 1, n):
            dist = np.linalg.norm(trace_ends[kk, :] - trace_ends[ii, :], 2)
            if dist > maxdist:
                i1, i2 = ii, kk
                maxdist = dist

    # trial vector-a, connecting the two most distant endpoints
    a = trace_ends[[i1, i2], :].copy()
    if a[1, 0] - a[0, 0] < 0:
        a = trace_ends[[i2, i1], :].copy()
    a_hat = a[1, :] - a[0, :]
    a_hat = a_hat / np.linalg.norm(a_hat, 2)  # unit vector

    # projection of end-to-end vector onto a_hat
    e = np.zeros(m)
    for jj in range(m):
        e[jj] = (trace_ends[2 * jj + 1, :] - trace_ends[2 * jj, :]) @ a_hat
    E = np.sum(e)

    # vector-b with strike discordance corrected
    b = np.array([0.0, 0.0])
    for jj in range(m):
        seg = trace_ends[2 * jj + 1, :] - trace_ends[2 * jj, :]
        if np.sign(e[jj]) == np.sign(E):
            b = b + seg
        else:
            b = b - seg

    return {'a': a, 'b': b, 'e': e, 'E': E}


# ---------------------------------------------------------------------------
# Step 3 — Link traces; reverse the strike of any discordant trace
# ---------------------------------------------------------------------------
def linktraces(ftraces, nominal_strike, type_, discordant):
    m = len(ftraces)
    a = nominal_strike['a']
    b = nominal_strike['b'].reshape(2, 1)  # column vector
    e = nominal_strike['e']
    E = nominal_strike['E']

    # work on local copies so the caller's data is not mutated
    work = []
    for ft in ftraces:
        work.append({
            'trace': np.array(ft['trace'], dtype=float),
            'l': np.atleast_1d(np.array(ft['l'], dtype=float)),
            'strike': np.atleast_1d(np.array(ft['strike'], dtype=float)),
        })

    if discordant:
        for jj in range(m):
            if e[jj] * E < 0:  # reverse strike, current trace is discordant
                work[jj]['trace'] = np.flipud(work[jj]['trace'])
                work[jj]['l'] = work[jj]['l'][::-1].copy()
                work[jj]['strike'] = work[jj]['strike'] - 180.0

    # reference axis and origin for the coordinate shift
    if type_.get('str') == 'NGA2':
        reference_axis = np.sign(E) * (a[1, :] - a[0, :])
        if E < 0:
            p_origin = a[1, :].copy()
        else:
            p_origin = a[0, :].copy()
    else:  # default OFR case, also used for 'JB'
        if float((a[1, :] - a[0, :]) @ b.ravel()) >= 0:
            p_origin = a[0, :].copy()
        else:
            p_origin = a[1, :].copy()
        reference_axis = nominal_strike['b'].copy()
    reference_axis = reference_axis / np.linalg.norm(reference_axis, 2)

    # compute U'_p1 for each trace
    Uprime_p1 = np.full(m, np.nan)
    for jj in range(m):
        Uprime_p1[jj] = (work[jj]['trace'][0, :] - p_origin) @ reference_axis

    # link all traces into a single merged trace, separated by NaNs
    Trace = np.zeros((0, 2))
    Strike = []
    Len = []
    s = []
    for jj in range(m):
        ftr = work[jj]
        Trace = np.vstack([Trace, ftr['trace']]) if Trace.size else ftr['trace'].copy()
        Strike.extend(list(ftr['strike']) + [np.nan])
        Len.extend(list(ftr['l']) + [np.nan])
        s_block = Uprime_p1[jj] + np.concatenate([[0.0], np.cumsum(ftr['l'])])
        s_block = s_block.tolist()
        s.extend(s_block)
        s[-1] = np.nan  # overwrite the last appended value (separator)

    # remove the final trailing NaN separator
    Strike = np.array(Strike[:-1])
    Len = np.array(Len[:-1])
    s = np.array(s[:-1])

    single_trace = {
        'strike': Strike,
        'l': Len,
        's': s,
        'ftrace': Trace,
        'ref_axis': reference_axis,
    }
    return single_trace, reference_axis, p_origin


# ---------------------------------------------------------------------------
# Step 4 — Compute GC2 (single-site version)
# ---------------------------------------------------------------------------
def compute_gc2(StaX, StaY, single_trace):
    strike = single_trace['strike']
    l = single_trace['l']
    s = single_trace['s']
    nseg = len(l)
    ftrace = single_trace['ftrace']
    p_origin = ftrace[:nseg, :]  # discard the last (nseg+1) coordinate

    # (t, u, wgt) w.r.t. each of the nseg coordinate systems
    seg_tuw_list = []
    for iseg in range(nseg):
        seg_tuw_list.append(
            comp_segment_tuw(p_origin[iseg, :], strike[iseg], l[iseg], StaX, StaY)
        )

    GC2_U = 0.0
    GC2_T = 0.0
    Wgt = 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        for iseg in range(nseg):
            if np.isnan(l[iseg]):
                continue  # skip bogus (separator) segments
            seg = seg_tuw_list[iseg]
            GC2_U += (seg['u'] + s[iseg]) * seg['wgt']
            GC2_T += seg['t'] * seg['wgt']
            Wgt += seg['wgt']

        # guard against division by zero / inf
        GC2_U = GC2_U / Wgt
        GC2_T = GC2_T / Wgt

    # rule #3 for sites located on the fault (Wgt == inf)
    if np.isinf(Wgt):
        GC2_T = 0.0
        for ii in range(nseg):
            if np.isinf(seg_tuw_list[ii]['wgt']):
                GC2_U = seg_tuw_list[ii]['u'] + s[ii]
                Wgt = np.nan
                break

    return GC2_T, GC2_U, Wgt


def comp_segment_tuw(origin, strike, l, StaX, StaY):
    """Compute (t, u, w) w.r.t. fault axes defined by (origin, strike, l)."""
    strikerad = strike / 180.0 * np.pi

    uhat = np.array([np.sin(strikerad), np.cos(strikerad)])
    that = np.array([np.sin(strikerad + np.pi / 2.0), np.cos(strikerad + np.pi / 2.0)])

    t = (StaX - origin[0]) * that[0] + (StaY - origin[1]) * that[1]
    u = (StaX - origin[0]) * uhat[0] + (StaY - origin[1]) * uhat[1]

    # closed-form solution to Equation 1 of the OFR
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(t):
            wgt = np.nan
        elif abs(t) > 1e-6:  # rule 1
            wgt = (np.arctan((l - u) / t) - np.arctan(-u / t)) / t
        elif u < 0 or u > l:  # rule 2
            wgt = 1.0 / (u - l) - 1.0 / u
        else:  # rule 3; T = 0
            wgt = np.inf

    return {'t': t, 'u': u, 'wgt': wgt}


# Step 5 — Gradient of T: not implemented (matches the original).
