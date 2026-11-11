"""
Combine per-rupture Bea25 moment-modifier files into ONE physical Parquet file
for the OpenQuake hand-off.

The output is sorted by (rupture_id, custom_site_id, period) so that reading a
single rupture back is fast via predicate pushdown on row-group statistics:

    import pandas as pd
    df = pd.read_parquet("bea25_momodifiers_v1.parquet",
                         filters=[("rupture_id", "=", 167377)])

Build it either by (A) combining the per-rupture .parquet files you already
wrote, or (B) streaming straight from unknown_hypocenter in one pass.
"""

import glob
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# tidy dtypes keep the file small and OQ-friendly
def _tidy(df):
    df = df.copy()
    if "rupture_id" in df:
        df["rupture_id"] = df["rupture_id"].astype("int64")
    if "custom_site_id" in df:
        df["custom_site_id"] = df["custom_site_id"].astype("string")
    for c in ("lon", "lat", "period", "mu_fd", "phi_iuh", "amp", "phi_red"):
        if c in df:
            df[c] = df[c].astype("float32" if c in ("period", "mu_fd", "phi_iuh",
                                                    "amp", "phi_red") else "float64")
    return df


# ---------------------------------------------------------------------------
# (A) Combine the per-rupture files you ALREADY wrote, into one sorted file.
#     Streams with a single ParquetWriter so memory stays low even for the
#     whole country x all periods.
# ---------------------------------------------------------------------------
def combine_existing(in_glob, out_path, compression="zstd", row_group_size=250_000):
    files = sorted(glob.glob(in_glob),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not files:
        raise FileNotFoundError(f"no files matched {in_glob}")
    writer = None
    n_rows = 0
    try:
        for f in files:                      # already one rupture per file -> sorted by rupture
            df = _tidy(pd.read_parquet(f))
            df = df.sort_values([c for c in ("custom_site_id", "period") if c in df])
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema,
                                          compression=compression)
            writer.write_table(tbl, row_group_size=row_group_size)
            n_rows += len(df)
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote {out_path}: {n_rows:,} rows from {len(files)} ruptures "
          f"({os.path.getsize(out_path)/1e6:.1f} MB, {compression})")
    return out_path


# ---------------------------------------------------------------------------
# (B) Build in one pass directly from unknown_hypocenter (no per-rupture files).
# ---------------------------------------------------------------------------
def build_from_scratch(rupture_ids, sites, periods, out_path, *, bd,
                       trace_tmpl="fault_ruptures/%d_traces.csv",
                       info_tmpl="fault_ruptures/%d_traceinfo.csv",
                       version=1, spacing_km=5.0,
                       compression="zstd", row_group_size=250_000):
    writer = None
    n_rows = n_rup = 0
    try:
        for rid in rupture_ids:              # iterate in sorted order -> file sorted by rupture
            trace = pd.read_csv(trace_tmpl % rid)
            info = pd.read_csv(info_tmpl % rid)
            mm = bd.unknown_hypocenter(trace, sites, imt=periods, traceinfo=info,
                                       spacing_km=spacing_km, rupture_id=rid,
                                       verbose=False, version=version)
            if mm is None:                   # reverse/normal or sub-M6 -> no directivity
                continue
            mm = _tidy(mm)
            mm = mm.sort_values([c for c in ("custom_site_id", "period") if c in mm])
            tbl = pa.Table.from_pandas(mm, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression=compression)
            writer.write_table(tbl, row_group_size=row_group_size)
            n_rows += len(mm); n_rup += 1
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote {out_path}: {n_rows:,} rows from {n_rup} ruptures with directivity "
          f"({os.path.getsize(out_path)/1e6:.1f} MB, {compression})")
    return out_path


if __name__ == "__main__":
    # (A) combine the files your loop already produced:
    combine_existing("momodifers/bea25-version1/*.parquet",
                     "momodifers/bea25_momodifiers_v1.parquet")

    # (B) one-pass build (uncomment to use instead):
    # import bea25_directivity as bd
    # sites = pd.read_csv("NZ_inside_10km.csv")
    # periods = [0.3,0.4,0.5,0.75,1,1.5,2,3,4,5,7.5,10]
    # rupture_ids = sorted(rupture_ids)
    # build_from_scratch(rupture_ids, sites, periods,
    #                    "momodifers/bea25_momodifiers_v1.parquet", bd=bd, version=1)
