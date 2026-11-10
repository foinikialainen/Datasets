#!/usr/bin/env python3
"""
compare_apps.py
===============
Compare the IMU (inertial) and GNSS data recorded by TWO Android apps that ran
at the same time on the same phone:

  1. GNSSLogger (the well-known app)
       -> files named like  gnss_log_2026_11_02_12_13_23.txt
  2. your own app ("MIMIR" in the example data)
       -> many small files named like  log_02112026-121604.txt
          (the app splits one recording into many fixed-size chunk files)

What the script does
--------------------
* You only pass the two PARENT directories; the script finds all data files
  inside them automatically (you never have to list files by hand).
* It reads both different file formats.
* Main focus - the IMU data (accelerometer, gyroscope, magnetometer):
    - plots both apps' IMU signals on the same time axis,
    - checks each app's IMU clock for gaps/steps (periods of a couple of
      seconds where nothing was logged),
    - compares the IMU values of the two apps sample by sample.
* It also compares the GNSS "Fix" data (the phone's position estimate, ~1 Hz):
    - plots the position track of both apps,
    - looks for clock jumps to 1970 (timestamps near zero) and other jumps,
    - measures the position difference between the two apps over time.
* Finally a short comparison of the raw GNSS measurements (counts, signal
  strength).

About time and file names
-------------------------
Every data line contains a "Unix time in milliseconds" (ms since 1 Jan 1970,
UTC), so the two apps are aligned exactly using those in-line timestamps -
file names are NOT needed for the alignment. File names are only used to sort
the chunk files of the second app. Note that:
  - GNSSLogger writes the file name in UTC        (gnss_log_2026_11_02_12_13_23)
  - the second app writes the file name in LOCAL time (log_02112026-121604)
The script measures and prints the difference between the file-name time and
the timestamps inside each file, so you can see this offset yourself.

A "1970 clock jump" means a timestamp that suddenly becomes near zero (the
phone clock was reset to the Unix epoch). A "gap" means the time between two
consecutive logged samples is much larger than usual - as if the app did not
log anything for a while.

Usage
-----
    python3 compare_apps.py <gnsslogger_parent_dir> <other_app_parent_dir> [--out OUTPUT_DIR]

Example (from the repository root):
    python3 compare_apps.py GNSSLogger MIMIR --out comparison_output

Requires Python 3 with pandas, numpy and matplotlib (see requirements.txt).

Everything is printed to the screen, written to report.txt, and plotted as
PNG files inside the output directory.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw plots to files, no screen needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants and small helpers
# ---------------------------------------------------------------------------

# Any timestamp smaller than this is considered a "1970-style clock jump":
# 946684800000 ms = 1 Jan 2000 UTC. A clock that jumped to 1970 produces values
# near 0 ms, which are always below this limit.
EPOCH_JUMP_MAX_MS = 946_684_800_000

# Column names (normalized: lower case, only letters+digits) that hold the
# Unix time in milliseconds.
TIME_COL_NAMES = {"unixtimemillis", "unixtimemillisecond", "utctimemillis", "timestampunixms"}

# Map raw record tag -> canonical sensor name used for comparison.
# GNSSLogger writes "UncalAccel" etc., the second app writes "ACC_UNCAL" etc.
SENSOR_BASE = {"ACCEL": "ACC", "ACC": "ACC", "GYRO": "GYR", "GYR": "GYR", "MAG": "MAG", "MAGNETOMETER": "MAG"}


def norm(name):
    """Normalize a column name: lower case and keep only letters/digits.
    This lets us match 'UnixTimeMillis', 'UnixTime_millisecond', ... alike."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def to_int(s):
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def fmt_ts(ms):
    """Format a Unix-ms timestamp as a readable UTC string."""
    if ms is None:
        return "-"
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def to_dt64(ms_array):
    """Convert Unix-ms values to numpy datetime64 (UTC) for plotting."""
    return np.asarray(ms_array, dtype="int64").astype("datetime64[ms]")


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points (haversine formula)."""
    r = 6_371_000.0  # Earth radius in meters
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_files(parent, pattern):
    """Recursively find all files under `parent` whose NAME matches `pattern`."""
    return sorted(glob.glob(os.path.join(parent, "**", pattern), recursive=True))


def parse_gnsslogger_filename(path):
    """GNSSLogger file name: gnss_log_YYYY_MM_DD_HH_MM_SS.txt -> naive datetime."""
    m = re.search(r"gnss_log_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})", os.path.basename(path))
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


def parse_mimir_filename(path):
    """Second app file name: log_DDMMYYYY-HHMMSS.txt (local time) -> naive datetime."""
    m = re.search(r"(\d{2})(\d{2})(\d{4})-(\d{2})(\d{2})(\d{2})", os.path.basename(path))
    if not m:
        return None
    d, mo, y, h, mi, s = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Generic log parser (works for BOTH apps)
#
# Both apps write CSV-like text files where each line starts with a record tag
# ("Fix", "Raw", "UncalAccel", "ACC", ...). Each file also contains comment
# lines of the form "# Tag,column1,column2,..." that describe the columns of
# the following records. We use those in-file headers to find columns by NAME,
# so we do not have to hard-code column positions (the two apps order their
# columns differently).
# ---------------------------------------------------------------------------

def canonical_sensor(tag):
    """Map a record tag to a canonical sensor name, e.g.
    'UncalAccel' -> 'ACC_UNCAL', 'ACC_UNCAL' -> 'ACC_UNCAL', 'Gyro' -> 'GYR'."""
    t = tag.upper()
    if t.startswith("UNCAL"):          # GNSSLogger style: UncalAccel, UncalGyro, ...
        base, uncal = t[5:], True
    elif t.endswith("_UNCAL"):         # second app style: ACC_UNCAL, GYR_UNCAL, ...
        base, uncal = t[:-6], True
    else:
        base, uncal = t, False
    base = SENSOR_BASE.get(base)
    if base is None:
        return None
    return base + ("_UNCAL" if uncal else "")


def extract_record(tag, colmap, row):
    """Turn one CSV row into a dict, using the column-name map of its tag.
    Returns a dict with 'kind' in {'fix', 'imu', 'raw'} or None if the record
    type is not needed."""
    if colmap is None or not row:
        return None

    def col(pred):
        for name, idx in colmap.items():
            if idx < len(row) and pred(name):
                return row[idx]
        return None

    # Unix time in milliseconds of this record.
    time_ms = to_int(col(lambda n: n in TIME_COL_NAMES))

    # Fix = the phone's position estimate (lat/lon/altitude at ~1 Hz).
    if tag == "FIX":
        return {
            "kind": "fix",
            "provider": col(lambda n: n == "provider"),
            "lat": to_float(col(lambda n: "latitude" in n)),
            "lon": to_float(col(lambda n: "longitude" in n)),
            "alt": to_float(col(lambda n: "altitude" in n)),
            "speed": to_float(col(lambda n: "speed" in n and "accuracy" not in n and "drift" not in n)),
            "accuracy": to_float(col(lambda n: "accuracy" in n and "meter" in n)),
            "time_ms": time_ms,
            "elapsed_ns": to_int(col(lambda n: n.startswith("elapsedrealtime"))),
        }

    # Raw = one raw GNSS measurement from one satellite.
    if tag == "RAW":
        cn0 = col(lambda n: n == "cn0dbhz") or col(lambda n: "cn0" in n and "baseband" not in n)
        return {
            "kind": "raw",
            "time_ms": time_ms,
            "svid": to_int(col(lambda n: n == "svid")),
            "cn0": to_float(cn0),
            "constellation": to_int(col(lambda n: n == "constellationtype")),
        }

    # IMU sensors (accelerometer / gyroscope / magnetometer, calibrated or not).
    sensor = canonical_sensor(tag)
    if sensor is not None:
        # Find the x/y/z value columns. Two naming schemes exist:
        #  - per-record chunk files: columns right after 'elapsedRealtime...'
        #    (e.g. UncalAccel,utcTimeMillis,elapsedRealtimeNanos,UncalAccelXMps2,...)
        #  - the merged wide CSV: one header for all record types, where each
        #    row only fills ITS OWN sensor columns (e.g. x_uncalibrated_radPerSecond
        #    for GYR_UNCAL rows, empty for ACC rows).
        # So we pick columns by the physical UNIT of the sensor (m/s^2 for the
        # accelerometer, rad/s for the gyroscope, uT for the magnetometer) and
        # then by axis letter - this works for both naming schemes.
        base = sensor.replace("_UNCAL", "")
        unit_tokens = {"ACC": ("mps2", "meterpersecond"),
                       "GYR": ("radpersec",),
                       "MAG": ("microt",)}[base]
        elapsed_idx = next((i for n, i in colmap.items() if n.startswith("elapsedrealtime")), -1)
        tag_norm = norm(tag)
        axis_col = {}
        for n, i in sorted(colmap.items(), key=lambda kv: kv[1]):
            if i <= elapsed_idx or "bias" in n or "drift" in n or "accuracy" in n:
                continue
            if not any(tok in n for tok in unit_tokens):
                continue
            axis = None
            if tag_norm and n.startswith(tag_norm) and len(n) > len(tag_norm) and n[len(tag_norm)] in "xyz":
                axis = n[len(tag_norm)]          # e.g. 'UncalAccelXMps2' -> 'x'
            elif n and n[0] in "xyz":
                axis = n[0]                       # e.g. 'x_uncalibrated_...' -> 'x'
            if axis and axis not in axis_col:
                axis_col[axis] = i
        if not axis_col:
            # fallback: first three non-bias/drift/accuracy columns after elapsed
            fallback = sorted(i for n, i in colmap.items()
                              if i > elapsed_idx and "bias" not in n and "drift" not in n and "accuracy" not in n)[:3]
            axis_col = dict(zip("xyz", fallback))
        vals = []
        for a in "xyz":
            i = axis_col.get(a)
            vals.append(to_float(row[i]) if (i is not None and i < len(row)) else None)
        return {
            "kind": "imu",
            "sensor": sensor,
            "time_ms": time_ms,
            "elapsed_ns": to_int(col(lambda n: n.startswith("elapsedrealtime"))),
            "x": vals[0], "y": vals[1], "z": vals[2],
        }

    return None  # Nav, Status, Agc, PHOTO, ... - not needed for this comparison


def parse_log_file(path):
    """Parse one text log file (GNSSLogger style or second-app chunk style).
    Returns a dict with lists of records plus some file-level info."""
    colmaps = {}          # record tag (upper case) -> {normalized column name: index}
    global_colmap = None  # used by the merged *.csv of the second app
    out = {"fix": [], "imu": [], "raw": [], "other": Counter(), "first_time_ms": None}

    with open(path, newline="", errors="replace") as f:
        for row in csv.reader(f):
            if not row:
                continue
            tag0 = row[0].strip()

            # First line of a merged *.csv is a normal header row.
            if global_colmap is None and norm(tag0) == "recordtype" and len(row) > 1:
                global_colmap = {norm(name): i for i, name in enumerate(row)}
                continue

            # Comment lines. A line like "# ACC,utcTimeMillis,x,..." is the
            # column header for the "ACC" records that follow in this file.
            if tag0.startswith("#"):
                tag = tag0.lstrip("#").strip().upper()
                if tag and len(row) > 1:
                    colmaps[tag] = {norm(name): i for i, name in enumerate(row[1:], start=1)}
                continue

            tag = tag0.upper()
            rec = extract_record(tag, colmaps.get(tag, global_colmap), row)
            if rec is None:
                out["other"][tag0[:24] if tag0 else "<empty>"] += 1
                continue
            if rec["time_ms"] is not None and (out["first_time_ms"] is None or rec["time_ms"] < out["first_time_ms"]):
                out["first_time_ms"] = rec["time_ms"]
            out[rec["kind"]].append(rec)

    return out


# ---------------------------------------------------------------------------
# Loading a whole app directory
# ---------------------------------------------------------------------------

def load_app(parent, role):
    """Load every data file of one app found under `parent`.
    role = 'gnsslogger' or 'other'. Returns a dict with DataFrames + info."""
    info = {"parent": parent, "role": role, "files": [], "skipped": []}

    if role == "gnsslogger":
        # GNSSLogger produces one main .txt log plus .nmea / .26o files.
        # For this comparison we only need the .txt log.
        all_logs = find_files(parent, "gnss_log_*")
        log_files = [p for p in all_logs if p.endswith(".txt")]
        info["skipped"] = [os.path.basename(p) for p in all_logs if p not in log_files]
        name_parser = parse_gnsslogger_filename
    else:
        # The second app splits the recording into fixed-size chunk files
        # (log_DDMMYYYY-HHMMSS.txt). Some versions of the app also write one
        # merged session *.csv with a single header row (easier to read).
        # We parse BOTH: the chunks are the primary source, and the CSV is
        # cross-checked against them (see crosscheck_chunks_vs_csv) so that
        # any difference between the two is reported instead of assumed.
        # If no chunk files exist, the CSV alone is used as the primary source.
        chunk_files = find_files(parent, "log_*.txt")
        csv_files = find_files(parent, "*.csv")
        if chunk_files:
            log_files = chunk_files
            info["csv_files"] = csv_files
        else:
            log_files = csv_files
            info["csv_files"] = []
        info["skipped"] = []
        name_parser = parse_mimir_filename

    fixes, imus, raws = [], [], []
    for path in log_files:
        parsed = parse_log_file(path)
        fname_dt = name_parser(path)
        info["files"].append({
            "path": path,
            "name": os.path.basename(path),
            "filename_dt": fname_dt,                 # naive time from the file name
            "first_record_ms": parsed["first_time_ms"],  # first Unix-ms timestamp inside the file
            "n_fix": len(parsed["fix"]),
            "n_imu": len(parsed["imu"]),
            "n_raw": len(parsed["raw"]),
        })
        for rec in parsed["fix"]:
            rec["src"] = os.path.basename(path)
            fixes.append(rec)
        for rec in parsed["imu"]:
            rec["src"] = os.path.basename(path)
            imus.append(rec)
        for rec in parsed["raw"]:
            rec["src"] = os.path.basename(path)
            raws.append(rec)

    # Sort files by their file-name time (this is how the chunk files of the
    # second app are ordered; for GNSSLogger there is usually just one file).
    info["files"].sort(key=lambda d: (d["filename_dt"] is None, d["filename_dt"] or datetime.min, d["name"]))

    info["fixes"] = pd.DataFrame(fixes)
    info["imu"] = pd.DataFrame(imus)
    info["raw"] = pd.DataFrame(raws)

    # If the second app also wrote merged CSV file(s), parse them separately
    # so they can be cross-checked against the chunk files.
    info["csv"] = None
    if role == "other" and info.get("csv_files"):
        cf, ci, cr = [], [], []
        for path in info["csv_files"]:
            parsed = parse_log_file(path)
            for rec in parsed["fix"]:
                rec["src"] = os.path.basename(path)
                cf.append(rec)
            for rec in parsed["imu"]:
                rec["src"] = os.path.basename(path)
                ci.append(rec)
            for rec in parsed["raw"]:
                rec["src"] = os.path.basename(path)
                cr.append(rec)
        info["csv"] = {"fixes": pd.DataFrame(cf), "imu": pd.DataFrame(ci), "raw": pd.DataFrame(cr)}
    return info


def detect_role(parent):
    """Guess whether a directory holds GNSSLogger data or the other app's data."""
    if find_files(parent, "gnss_log_*.txt"):
        return "gnsslogger"
    if find_files(parent, "log_*.txt") or find_files(parent, "*.csv"):
        return "other"
    return None


def crosscheck_chunks_vs_csv(app, say):
    """Cross-check the second app's fixed-size chunk files against its merged
    CSV file. They are supposed to contain the same records; here we VERIFY
    that instead of assuming it: we compare record counts, the set of
    timestamps, and the actual values at matching timestamps. Any difference
    is printed."""
    csvd = app.get("csv")
    say("\n--- cross-check for '" + app["parent"] + "': chunk files vs merged CSV ---")
    if not csvd:
        say("  no separate CSV file found; nothing to cross-check")
        return
    say(f"  chunk files: {len(app['files'])}, CSV files: {len(app['csv_files'])} "
        f"e.g. {os.path.basename(app['csv_files'][0])}")
    for label, cdf, vdf in (("Fix", app["fixes"], csvd["fixes"]),
                            ("IMU", app["imu"], csvd["imu"]),
                            ("Raw", app["raw"], csvd["raw"])):
        c_times = cdf["time_ms"].dropna().astype("int64") if len(cdf) else pd.Series(dtype="int64")
        v_times = vdf["time_ms"].dropna().astype("int64") if len(vdf) else pd.Series(dtype="int64")
        only_chunks = sorted(set(c_times) - set(v_times))
        only_csv = sorted(set(v_times) - set(c_times))
        say(f"  {label:4s}: records in chunks={len(c_times)}, in CSV={len(v_times)}, "
            f"timestamps only in chunks={len(only_chunks)}, only in CSV={len(only_csv)}")
        if only_csv:
            say(f"       CSV-only timestamps (first 5): {[fmt_ts(t) for t in only_csv[:5]]}")
        if only_chunks:
            say(f"       chunk-only timestamps (first 5): {[fmt_ts(t) for t in only_chunks[:5]]}")
    # value agreement at matching timestamps
    if len(app["fixes"]) and len(csvd["fixes"]):
        c = app["fixes"].drop_duplicates(subset=["time_ms"])
        v = csvd["fixes"].drop_duplicates(subset=["time_ms"])
        m = c.merge(v, on="time_ms", suffixes=("_chunk", "_csv"), how="inner")
        for col in ("lat", "lon", "alt"):
            a, b = m[f"{col}_chunk"], m[f"{col}_csv"]
            both = a.notna() & b.notna()
            mism = int((both & ((a - b).abs() > 1e-9)).sum())
            say(f"  Fix.{col}: matched timestamps={int(both.sum())}, value mismatches={mism}, "
                f"missing in CSV={int((a.notna() & b.isna()).sum())}")
    if len(app["imu"]) and len(csvd["imu"]):
        c = app["imu"].drop_duplicates(subset=["time_ms", "sensor"])
        v = csvd["imu"].drop_duplicates(subset=["time_ms", "sensor"])
        m = c.merge(v, on=["time_ms", "sensor"], suffixes=("_chunk", "_csv"), how="inner")
        for sensor in sorted(m["sensor"].unique()):
            ms = m[m["sensor"] == sensor]
            for col in ("x", "y", "z"):
                a, b = ms[f"{col}_chunk"], ms[f"{col}_csv"]
                both = a.notna() & b.notna()
                mism = int((both & ((a - b).abs() > 1e-9)).sum())
                empty_csv = int((a.notna() & b.isna()).sum())
                say(f"  IMU {sensor}.{col}: matched={int(both.sum())}, value mismatches={mism}, "
                    f"value empty in CSV={empty_csv}")


# ---------------------------------------------------------------------------
# Clock-health analysis (1970 jumps, backward jumps, gaps)
# ---------------------------------------------------------------------------

def clock_health(times_ms, gap_min_ms, backward_min_ms):
    """Analyse a series of timestamps (in the order they were logged).

    Returns a dict with statistics and a list of 'events':
      - 'epoch'   : timestamp jumped to ~1970 (below EPOCH_JUMP_MAX_MS)
      - 'backward': clock jumped backwards by more than the threshold
      - 'gap'     : no data logged for longer than the threshold
    The thresholds are adaptive: normally samples arrive every `p90` ms, so a
    gap is anything much larger than that, but at least `gap_min_ms`.
    """
    t = np.asarray(times_ms, dtype="int64")
    result = {
        "n": len(t),
        "min_ms": int(t.min()) if len(t) else None,
        "max_ms": int(t.max()) if len(t) else None,
        "median_dt": None, "p90_dt": None, "max_dt": None,
        "gap_thr_ms": None, "backward_thr_ms": None,
        "n_epoch": 0, "n_backward": 0, "n_gap": 0,
        "events": [],  # (index, t_prev, t, kind, delta_ms)
    }
    if len(t) < 2:
        return result
    dts = np.diff(t)
    pos = dts[dts > 0]
    med = float(np.median(pos)) if len(pos) else 0.0
    p90 = float(np.percentile(pos, 90)) if len(pos) else 0.0
    gap_thr = max(5.0 * p90, gap_min_ms)        # "nothing logged for a while"
    back_thr = max(2.0 * p90, backward_min_ms)   # clock jumped backwards
    result.update(median_dt=med, p90_dt=p90, max_dt=float(dts.max()),
                  gap_thr_ms=gap_thr, backward_thr_ms=back_thr)
    for i, d in enumerate(dts):
        kind = None
        if t[i + 1] < EPOCH_JUMP_MAX_MS:
            kind = "epoch"
            result["n_epoch"] += 1
        elif d < -back_thr:
            kind = "backward"
            result["n_backward"] += 1
        elif d > gap_thr:
            kind = "gap"
            result["n_gap"] += 1
        if kind:
            result["events"].append((i, int(t[i]), int(t[i + 1]), kind, int(d)))
    return result


def wall_vs_uptime_clock(imu_df):
    """Sanity check: each IMU record carries both the wall clock (Unix ms) and
    the device uptime clock (elapsedRealtimeNanos). Their difference should be
    constant. If the wall clock jumps to 1970, this difference suddenly changes,
    so we count records that disagree with the typical difference by > 60 s."""
    df = imu_df.dropna(subset=["time_ms", "elapsed_ns"])
    if df.empty:
        return None
    diff_ms = df["time_ms"].to_numpy() - df["elapsed_ns"].to_numpy() / 1e6
    typical = float(np.median(diff_ms))
    n_bad = int(np.sum(np.abs(diff_ms - typical) > 60_000))
    return {"typical_offset_ms": typical, "n_mismatch": n_bad, "n": len(df)}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel("time of day (UTC)")


def downsample(times_ms, values, max_points=20000):
    """Keep at most max_points (time, value) pairs for plotting (every k-th)."""
    n = len(times_ms)
    if n <= max_points:
        return times_ms, values
    k = int(math.ceil(n / max_points))
    return times_ms[::k], values[::k]


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare IMU + GNSS data of GNSSLogger and a second app. "
                    "Pass the two PARENT directories; all files inside are found automatically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example: python3 compare_apps.py GNSSLogger MIMIR --out comparison_output",
    )
    ap.add_argument("dir1", help="parent directory of the first app's data (GNSSLogger)")
    ap.add_argument("dir2", help="parent directory of the second app's data")
    ap.add_argument("--out", default="comparison_output", help="output directory for plots/report (default: comparison_output)")
    ap.add_argument("--gap-ms", type=float, default=2000.0,
                    help="minimum gap (ms) counted as 'no data logged for a while' (default: 2000)")
    ap.add_argument("--backward-ms", type=float, default=1000.0,
                    help="minimum backward clock jump (ms) that is reported (default: 1000)")
    args = ap.parse_args()

    for d in (args.dir1, args.dir2):
        if not os.path.isdir(d):
            sys.exit(f"error: '{d}' is not a directory")
    os.makedirs(args.out, exist_ok=True)

    report_lines = []

    def say(text=""):
        """Print to screen AND collect into the report file."""
        print(text)
        report_lines.append(text)

    # ------------------------------------------------------------------ load
    role1, role2 = detect_role(args.dir1), detect_role(args.dir2)
    if role2 == "gnsslogger" and role1 != "gnsslogger":
        role1, role2 = role2, role1          # user swapped the two directories
        args.dir1, args.dir2 = args.dir2, args.dir1
    app1 = load_app(args.dir1, role1 or "other")
    app2 = load_app(args.dir2, role2 or "other")
    name1 = "GNSSLogger" if app1["role"] == "gnsslogger" else "app-1"
    name2 = "GNSSLogger" if app2["role"] == "gnsslogger" else "app-2 (your app)"
    apps = [(name1, app1), (name2, app2)]

    say("=" * 78)
    say("COMPARE TWO APPS - IMU + GNSS DATA")
    say("=" * 78)

    for name, app in apps:
        say(f"\n--- {name}: directory '{app['parent']}' ---")
        say(f"  data files found: {len(app['files'])}"
            + (f"  (first: {app['files'][0]['name']}, last: {app['files'][-1]['name']})" if app["files"] else ""))
        if app["skipped"]:
            say(f"  files skipped (not needed for this comparison): {len(app['skipped'])} "
                f"e.g. {app['skipped'][:3]}")
        if app.get("csv_files"):
            say(f"  merged CSV file(s) also found and parsed (cross-checked below): "
                f"{len(app['csv_files'])} e.g. {os.path.basename(app['csv_files'][0])}")
        for kind in ("fixes", "imu", "raw"):
            df = app[kind]
            if len(df):
                say(f"  {kind:6s}: {len(df):7d} records, "
                    f"time {fmt_ts(df['time_ms'].min())}  ->  {fmt_ts(df['time_ms'].max())}")
            else:
                say(f"  {kind:6s}: 0 records")
        if len(app["imu"]):
            say("  IMU records per sensor: " + ", ".join(
                f"{s}={n}" for s, n in app["imu"]["sensor"].value_counts().items()))
        # File-name time vs. the timestamps inside the files. This shows the
        # UTC-vs-local-time difference of the file names without relying on it.
        offsets_h = []
        for fi in app["files"]:
            if fi["filename_dt"] and fi["first_record_ms"]:
                data_utc = datetime.fromtimestamp(fi["first_record_ms"] / 1000.0, timezone.utc).replace(tzinfo=None)
                offsets_h.append((fi["filename_dt"] - data_utc).total_seconds() / 3600.0)
        if offsets_h:
            med = float(np.median(offsets_h))
            say(f"  file-name time minus in-file UTC time: {med:+.2f} h "
                f"(file names are ahead of the UTC timestamps inside the files by this amount)")

    # cross-check the second app's chunk files against its merged CSV (if any)
    for name, app in apps:
        if app.get("csv"):
            crosscheck_chunks_vs_csv(app, say)

    # ------------------------------------------------------------- overlap
    def time_range(app):
        parts = [app[k]["time_ms"] for k in ("fixes", "imu", "raw") if len(app[k])]
        if not parts:
            return None
        all_ms = pd.concat(parts)
        return int(all_ms.min()), int(all_ms.max())

    r1, r2 = time_range(app1), time_range(app2)
    if r1 and r2:
        ov_lo, ov_hi = max(r1[0], r2[0]), min(r1[1], r2[1])
        say(f"\nTime overlap of the two recordings: {fmt_ts(ov_lo)} -> {fmt_ts(ov_hi)} "
            f"({(ov_hi - ov_lo) / 1000.0:.1f} s)")
        say("(all comparisons below use only this overlap, and always use the Unix-ms")
        say(" timestamps stored inside the data lines - never the file names)")
    else:
        ov_lo = ov_hi = None
        say("\nWARNING: could not determine a time overlap")

    # ------------------------------------------------- clock health per app
    say("\n" + "=" * 78)
    say("CLOCK HEALTH (1970 jumps / backward jumps / gaps in the logged timestamps)")
    say("=" * 78)
    health = {}
    for name, app in apps:
        say(f"\n--- {name} ---")
        groups = {}
        if len(app["fixes"]):
            groups["Fix"] = app["fixes"]["time_ms"].to_numpy()
        if len(app["raw"]):
            groups["Raw"] = app["raw"]["time_ms"].to_numpy()
        if len(app["imu"]):
            for sensor, g in app["imu"].groupby("sensor"):
                groups[f"IMU:{sensor}"] = g["time_ms"].to_numpy()
        # We analyse the timestamps in the order they were logged (records in
        # file order, chunk files in file-name order). That is what reveals
        # clock jumps and logging gaps. (For a chunked app, records are written
        # in bursts per chunk file, so a step of about one chunk length between
        # files is normal.)
        for label, times in groups.items():
            h = clock_health(times, args.gap_ms, args.backward_ms)
            health[(name, label)] = h
            rate = (1000.0 / h["median_dt"]) if h["median_dt"] else float("nan")
            say(f"  {label:14s} n={h['n']:7d}  median dt={h['median_dt']:8.1f} ms (~{rate:6.2f} Hz)  "
                f"max dt={h['max_dt']:9.1f} ms  |  1970-jumps={h['n_epoch']}  "
                f"backward={h['n_backward']}  gaps(>{h['gap_thr_ms']:.0f} ms)={h['n_gap']}")
        # wall-clock vs uptime-clock check on IMU records
        for sensor, g in (app["imu"].groupby("sensor") if len(app["imu"]) else []):
            chk = wall_vs_uptime_clock(g)
            if chk:
                say(f"  IMU:{sensor:9s} wall-clock vs uptime-clock: typical offset "
                    f"{chk['typical_offset_ms'] / 1000.0:.1f} s, records disagreeing by >60 s: {chk['n_mismatch']}")
        # detailed events (in log order; consecutive duplicates collapsed, because
        # a chunked app may re-log the identical stale record in every chunk)
        say("  detailed events (in log order, duplicates collapsed):")
        for label, times in groups.items():
            h = health[(name, label)]
            if not h["events"]:
                continue
            say(f"    [{label}]")
            shown, seen = 0, set()
            for idx, tp, t, kind, d in h["events"]:
                key = (tp, t, kind, d)
                if key in seen:
                    continue
                seen.add(key)
                if shown >= 20:
                    say(f"      ... and {len(seen) - shown} more unique events")
                    break
                say(f"      {kind:8s} {fmt_ts(tp)} -> {fmt_ts(t)}  (delta {d:+d} ms)")
                shown += 1

    # ------------------------------------------------------------- fix plots
    say("\n" + "=" * 78)
    say("GNSS FIX COMPARISON (position estimates, ~1 Hz)")
    say("=" * 78)
    fix_pairs_rows = []
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, app in apps:
        df = app["fixes"].dropna(subset=["time_ms", "lat", "lon"])
        if ov_lo is not None:
            df = df[(df["time_ms"] >= ov_lo) & (df["time_ms"] <= ov_hi)]
        if len(df) == 0:
            continue
        for prov, g in df.groupby(df["provider"].fillna("?")):
            ax.scatter(g["lon"], g["lat"], s=4, alpha=0.5, label=f"{name} [{prov}]")
        say(f"  {name}: {len(df)} fixes in overlap, providers: "
            + ", ".join(f"{p}={n}" for p, n in df["provider"].value_counts().items()))
        for prov, g in df.groupby(df["provider"].fillna("?")):
            gt = g["time_ms"].to_numpy()
            if len(gt) > 1:
                med = float(np.median(np.diff(gt)))
                say(f"    provider {prov}: {len(g)} fixes, ~{1000.0 / med:.2f} Hz (median dt {med:.0f} ms)")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("Position track of both apps (overlap window)")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    p = os.path.join(args.out, "01_fix_track.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    say(f"  saved plot: {p}")

    # fix clock plot: clock value vs record number (left) + dt between consecutive fixes (right)
    # NOTE: fixes are plotted in the order they were LOGGED (not sorted by time),
    # so clock jumps show as vertical drops and logging gaps as vertical steps.
    # The x-axis is the record number (a plain count), the y-axis is the clock
    # value (a real time), so the time-of-day formatter belongs on the Y axis.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, (name, app) in enumerate(apps):
        t = app["fixes"].dropna(subset=["time_ms"])["time_ms"].to_numpy()  # log order
        if len(t) == 0:
            continue
        dts = np.diff(t)
        h = health.get((name, "Fix"))
        ax_t, ax_d = axes[row][0], axes[row][1]
        ax_t.plot(np.arange(len(t)), to_dt64(t), lw=0.8)
        ax_t.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax_t.set_xlabel("fix number (log order)")
        ax_t.set_ylabel("fix clock value (UTC)")
        ax_t.set_title(f"{name}: fix clock in log order (jumps drop down, gaps step up)")
        if h and h["events"]:
            for idx, tp, tt, kind, d in h["events"]:
                ax_t.axvline(idx, color="red", alpha=0.3)
        ax_d.semilogy(np.arange(1, len(t)), np.abs(dts) + 1, ".", ms=2)
        if h:
            ax_d.axhline(h["gap_thr_ms"], color="orange", ls="--", label=f"gap threshold {h['gap_thr_ms']:.0f} ms")
            ax_d.axhline(h["backward_thr_ms"], color="red", ls="--", label=f"backward threshold {h['backward_thr_ms']:.0f} ms")
            ax_d.legend(fontsize=7)
        ax_d.set_ylabel("|dt| between consecutive fixes (ms, log)")
        ax_d.set_xlabel("fix number (log order)")
        for idx, tp, tt, kind, d in (h["events"] if h else [])[:200]:
            if kind == "backward":
                ax_d.plot(idx + 1, abs(d) + 1, "rv", ms=4)
            elif kind == "gap":
                ax_d.plot(idx + 1, d + 1, "r^", ms=4)
            elif kind == "epoch":
                ax_d.plot(idx + 1, 1, "ks", ms=4)
    fig.suptitle("GNSS fix clock per app: clock value vs record number (left), time steps (right)")
    fig.tight_layout()
    p = os.path.join(args.out, "02_fix_clock.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    say(f"  saved plot: {p}")

    # position difference between the two apps (nearest fix in time)
    f1 = app1["fixes"].dropna(subset=["time_ms", "lat", "lon"]).sort_values("time_ms")
    f2 = app2["fixes"].dropna(subset=["time_ms", "lat", "lon"]).sort_values("time_ms")
    if len(f1) and len(f2) and ov_lo is not None:
        t1 = f1["time_ms"].to_numpy()
        lo, hi = max(t1.min(), f2["time_ms"].min()), min(t1.max(), f2["time_ms"].max())
        f2o = f2[(f2["time_ms"] >= lo) & (f2["time_ms"] <= hi)]
        t2 = f2o["time_ms"].to_numpy()
        idx = np.searchsorted(t1, t2)
        idx = np.clip(idx, 1, len(t1) - 1)
        left = np.abs(t1[idx - 1] - t2)
        right = np.abs(t1[idx] - t2)
        nearest = np.where(left <= right, idx - 1, idx)
        dt_match = np.abs(t1[nearest] - t2)
        ok = dt_match <= 2000  # only compare fixes that are at most 2 s apart
        ok_idx = np.where(ok)[0]
        dists = np.array([
            haversine_m(f2o["lat"].iloc[i], f2o["lon"].iloc[i],
                        f1["lat"].iloc[nearest[i]], f1["lon"].iloc[nearest[i]])
            for i in ok_idx
        ])
        t_ok = t2[ok]
        for row_i, i in enumerate(ok_idx):
            fix_pairs_rows.append({
                "time_utc": fmt_ts(int(t2[i])), "time_ms": int(t2[i]),
                "provider": f2o["provider"].iloc[i] if "provider" in f2o else "",
                f"lat_{name2}": float(f2o["lat"].iloc[i]), f"lon_{name2}": float(f2o["lon"].iloc[i]),
                f"lat_{name1}": float(f1["lat"].iloc[nearest[i]]), f"lon_{name1}": float(f1["lon"].iloc[nearest[i]]),
                "dt_to_other_app_ms": int(dt_match[i]), "distance_m": round(float(dists[row_i]), 3),
            })
        if len(dists):
            say(f"  position difference {name2} vs {name1} (fixes matched within 2 s): "
                f"n={len(dists)}, mean={dists.mean():.2f} m, median={np.median(dists):.2f} m, max={dists.max():.2f} m")
            if fix_pairs_rows and "provider" in fix_pairs_rows[0]:
                for prov, g in pd.DataFrame(fix_pairs_rows).groupby("provider"):
                    say(f"    provider {prov}: n={len(g)}, mean distance={g['distance_m'].mean():.2f} m, max={g['distance_m'].max():.2f} m")
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(to_dt64(t_ok), dists, ".", ms=3)
            ax.set_ylabel("distance between the two apps' fixes (m)")
            ax.set_title(f"Position difference over time ({name2} vs {name1}, matched within 2 s)")
            plot_time_axis(ax)
            fig.tight_layout()
            p = os.path.join(args.out, "03_fix_position_diff.png")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            say(f"  saved plot: {p}")
            pd.DataFrame(fix_pairs_rows).to_csv(os.path.join(args.out, "fix_pairs.csv"), index=False)
            say(f"  saved table: {os.path.join(args.out, 'fix_pairs.csv')}")

    # ------------------------------------------------------------- IMU plots
    say("\n" + "=" * 78)
    say("IMU COMPARISON (accelerometer / gyroscope / magnetometer)")
    say("=" * 78)
    imu1, imu2 = app1["imu"], app2["imu"]
    sensors1 = set(imu1["sensor"].unique()) if len(imu1) else set()
    sensors2 = set(imu2["sensor"].unique()) if len(imu2) else set()
    common = sorted(sensors1 & sensors2)
    say(f"  sensors in {name1}: {sorted(sensors1)}")
    say(f"  sensors in {name2}: {sorted(sensors2)}")
    say(f"  sensors compared (present in both): {common}")

    axis_labels = {"ACC": "m/s^2", "ACC_UNCAL": "m/s^2", "GYR": "rad/s", "GYR_UNCAL": "rad/s",
                   "MAG": "uT", "MAG_UNCAL": "uT"}

    for sensor in common:
        d1 = imu1[imu1["sensor"] == sensor].dropna(subset=["time_ms"]).sort_values("time_ms")
        d2 = imu2[imu2["sensor"] == sensor].dropna(subset=["time_ms"]).sort_values("time_ms")
        if ov_lo is not None:
            d1 = d1[(d1["time_ms"] >= ov_lo) & (d1["time_ms"] <= ov_hi)]
            d2 = d2[(d2["time_ms"] >= ov_lo) & (d2["time_ms"] <= ov_hi)]
        if len(d1) == 0 or len(d2) == 0:
            continue
        unit = axis_labels.get(sensor, "")
        # --- overlay plot: both apps on the same time axis, one subplot per axis
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for k, axname in enumerate(("x", "y", "z")):
            t1, v1 = downsample(d1["time_ms"].to_numpy(), d1[axname].to_numpy())
            axes[k].plot(to_dt64(t1), v1, lw=0.6, label=name1)
            axes[k].plot(to_dt64(d2["time_ms"].to_numpy()), d2[axname].to_numpy(), "o", ms=4, label=name2)
            axes[k].set_ylabel(f"{axname} ({unit})")
            axes[k].grid(alpha=0.3)
        axes[0].set_title(f"{sensor} - both apps on the same time axis (overlap window)")
        axes[0].legend(fontsize=8)
        plot_time_axis(axes[-1])
        fig.tight_layout()
        p = os.path.join(args.out, f"04_imu_overlay_{sensor}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        say(f"  saved plot: {p}")

        # --- sample-by-sample difference: for every sample of app2 find the
        #     nearest-in-time sample of app1 and subtract the values
        t1 = d1["time_ms"].to_numpy()
        t2 = d2["time_ms"].to_numpy()
        idx = np.searchsorted(t1, t2)
        idx = np.clip(idx, 1, len(t1) - 1)
        nearest = np.where(np.abs(t1[idx - 1] - t2) <= np.abs(t1[idx] - t2), idx - 1, idx)
        dt_match = np.abs(t1[nearest] - t2)
        ok = dt_match <= 1000  # only compare samples at most 1 s apart
        pair_rows = []
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        diff_stats = {}
        ok_idx = np.where(ok)[0]
        for k, axname in enumerate(("x", "y", "z")):
            v1 = d1[axname].to_numpy()
            v2 = d2[axname].to_numpy()
            diff = v2[ok] - v1[nearest[ok]]
            axes[k].plot(to_dt64(t2[ok]), diff, ".", ms=3)
            axes[k].axhline(0, color="k", lw=0.5)
            axes[k].set_ylabel(f"d{axname} ({unit})")
            axes[k].grid(alpha=0.3)
            if len(diff):
                diff_stats[axname] = (float(np.mean(np.abs(diff))), float(np.max(np.abs(diff))))
                for row_i, i in enumerate(ok_idx):
                    pair_rows.append({
                        "sensor": sensor, "time_utc": fmt_ts(int(t2[i])), "time_ms": int(t2[i]),
                        "dt_to_other_app_ms": int(dt_match[i]),
                        f"{axname}_{name2}": float(v2[i]), f"{axname}_{name1}": float(v1[nearest[i]]),
                        f"{axname}_diff": float(diff[row_i]),
                    })
        axes[0].set_title(f"{sensor}: value difference ({name2} minus {name1}), samples matched within 1 s")
        plot_time_axis(axes[-1])
        fig.tight_layout()
        p = os.path.join(args.out, f"06_imu_diff_{sensor}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        say(f"  saved plot: {p}")
        stats_txt = ", ".join(f"{a}: mean|d|={m:.3f} max|d|={x:.3f}" for a, (m, x) in diff_stats.items())
        say(f"  {sensor}: matched {int(ok.sum())}/{len(t2)} samples of {name2}; |diff| stats -> {stats_txt}")
        if pair_rows:
            pd.DataFrame(pair_rows).to_csv(os.path.join(args.out, f"imu_pairs_{sensor}.csv"), index=False)

    # --- IMU clock health plot (dt between consecutive IMU samples, log scale)
    imu_sensors_for_plot = [s for s in ("ACC_UNCAL", "ACC", "GYR_UNCAL", "GYR", "MAG_UNCAL", "MAG") if s in common]
    if imu_sensors_for_plot:
        ncols = len(imu_sensors_for_plot)
        fig, axes = plt.subplots(2, ncols, figsize=(5.2 * ncols, 8), sharex="col")
        if ncols == 1:
            axes = np.array(axes).reshape(2, 1)
        for col_i, sensor in enumerate(imu_sensors_for_plot):
            for row_i, (name, app) in enumerate(apps):
                d = app["imu"]
                d = d[d["sensor"] == sensor].dropna(subset=["time_ms"]).sort_values("time_ms")
                if ov_lo is not None:
                    d = d[(d["time_ms"] >= ov_lo) & (d["time_ms"] <= ov_hi)]
                ax = axes[row_i][col_i]
                if len(d) > 1:
                    t = d["time_ms"].to_numpy()
                    dts = np.diff(t)
                    ax.semilogy(to_dt64(t[1:]), dts, ".", ms=2)
                    h = health.get((name, f"IMU:{sensor}"))
                    if h and h["gap_thr_ms"]:
                        ax.axhline(h["gap_thr_ms"], color="orange", ls="--", lw=1)
                    for idx, tp, tt, kind, dd in (h["events"] if h else []):
                        if kind == "gap":
                            ax.plot(to_dt64([tt]), [dd], "r^", ms=6)
                        elif kind == "backward":
                            ax.plot(to_dt64([tt]), [abs(dd)], "rv", ms=6)
                        elif kind == "epoch":
                            ax.plot(to_dt64([tt]), [1], "ks", ms=6)
                ax.set_title(f"{name} - {sensor}", fontsize=9)
                ax.grid(alpha=0.3, which="both")
                if col_i == 0:
                    ax.set_ylabel("dt between samples (ms, log)")
                if row_i == 1:
                    plot_time_axis(ax)
        fig.suptitle("IMU clock health: time step between consecutive IMU samples "
                     "(red triangle = gap, red v = backward jump, black square = 1970 jump; "
                     "orange line = gap threshold)")
        fig.tight_layout()
        p = os.path.join(args.out, "05_imu_clock_health.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        say(f"  saved plot: {p}")

    # ------------------------------------- clock comparison: Fix vs IMU accelerometer
    # The user wants to see the Fix clock (~1 Hz) and the IMU accelerometer clock
    # (~125/150 Hz) side by side for EACH app. Left column: the clock values
    # themselves (y = clock value as UTC time, x = sample number in log order),
    # so a clock jump drops down and a logging gap steps up. Right column: the
    # time step between consecutive samples (log scale) plotted against the real
    # UTC time, with Fix and IMU on the SAME time axis, so you can see at which
    # moments each clock glitched.
    say("\n" + "=" * 78)
    say("CLOCK COMPARISON - Fix clock vs IMU accelerometer clock, per app")
    say("=" * 78)
    acc_sensor = None
    for cand in ("ACC_UNCAL", "ACC"):  # prefer the uncalibrated accelerometer
        if cand in sensors1 and cand in sensors2:
            acc_sensor = cand
            break
    if acc_sensor is None and sensors1 and sensors2:
        acc_sensor = sorted(sensors1 & sensors2)[0]
    if acc_sensor:
        say(f"  accelerometer sensor used for the IMU clock: {acc_sensor}")
        for name, app in apps:
            fix_t = app["fixes"].dropna(subset=["time_ms"])["time_ms"].to_numpy()      # log order
            imu_t = app["imu"][app["imu"]["sensor"] == acc_sensor].dropna(subset=["time_ms"])["time_ms"].to_numpy()
            if len(fix_t) > 1 and len(imu_t) > 1:
                fix_span_s = (fix_t.max() - fix_t.min()) / 1000.0
                imu_span_s = (imu_t.max() - imu_t.min()) / 1000.0
                say(f"  {name}: Fix clock rate ~{len(fix_t) / fix_span_s:.2f} Hz overall (n={len(fix_t)}, "
                    f"per-provider rates listed above); IMU {acc_sensor} clock rate "
                    f"~{len(imu_t) / imu_span_s:.2f} Hz (n={len(imu_t)})")
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for row, (name, app) in enumerate(apps):
            fix_t = app["fixes"].dropna(subset=["time_ms"])["time_ms"].to_numpy()
            imu_t = app["imu"][app["imu"]["sensor"] == acc_sensor].dropna(subset=["time_ms"])["time_ms"].to_numpy()
            if len(fix_t) == 0 or len(imu_t) == 0:
                continue
            axc, axd = axes[row][0], axes[row][1]
            # left: clock values vs sample number (IMU downsampled for display only)
            axc.plot(np.arange(len(fix_t)), to_dt64(fix_t), lw=0.9,
                     label=f"Fix clock (n={len(fix_t)})")
            k = max(1, int(math.ceil(len(imu_t) / 20000)))
            axc.plot(np.arange(0, len(imu_t), k), to_dt64(imu_t[::k]), ".", ms=1,
                     label=f"IMU {acc_sensor} clock (n={len(imu_t)})")
            axc.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            axc.set_xlabel("sample number (log order)")
            axc.set_ylabel("clock value (UTC)")
            axc.set_title(f"{name}: clock values - Fix vs IMU accelerometer")
            axc.legend(fontsize=8)
            axc.grid(alpha=0.3)
            for lbl in ("Fix", f"IMU:{acc_sensor}"):
                h = health.get((name, lbl))
                if h:
                    for idx, tp, tt, kind, d in h["events"]:
                        axc.axvline(idx, color="red", alpha=0.25)
            # right: time step dt vs real UTC time, Fix and IMU on the same axis
            for lbl, t in (("Fix", fix_t), (f"IMU {acc_sensor}", imu_t)):
                if len(t) > 1:
                    d = np.diff(t)
                    axd.semilogy(to_dt64(t[1:]), np.abs(d) + 1, ".", ms=1.5, label=f"{lbl} dt")
                    h = health.get((name, "Fix" if lbl == "Fix" else f"IMU:{acc_sensor}"))
                    if h and h["gap_thr_ms"]:
                        axd.axhline(h["gap_thr_ms"], ls="--", alpha=0.7,
                                    label=f"{lbl} gap threshold {h['gap_thr_ms']:.0f} ms")
            axd.set_ylabel("dt between consecutive samples (ms, log)")
            axd.set_title(f"{name}: Fix and IMU clock steps on the same UTC time axis")
            axd.legend(fontsize=7)
            axd.grid(alpha=0.3, which="both")
            plot_time_axis(axd)
        fig.suptitle("Clock comparison per app - left: clock value vs sample number "
                     "(jumps drop down, gaps step up); right: time step vs UTC time "
                     "(a glitch in either clock spikes up at the same moment)")
        fig.tight_layout()
        p = os.path.join(args.out, "08_clock_fix_vs_imu.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        say(f"  saved plot: {p}")
    else:
        say("  no common accelerometer sensor found in both apps; skipped")

    # ------------------------------------------------------------- raw GNSS
    say("\n" + "=" * 78)
    say("RAW GNSS MEASUREMENTS (brief comparison)")
    say("=" * 78)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, app in apps:
        df = app["raw"].dropna(subset=["time_ms"])
        if len(df) == 0:
            continue
        if ov_lo is not None:
            df = df[(df["time_ms"] >= ov_lo) & (df["time_ms"] <= ov_hi)]
        if len(df) == 0:
            continue
        t = df["time_ms"].to_numpy()
        bins = np.arange(t.min(), t.max() + 5000, 5000)
        counts, _ = np.histogram(t, bins=bins)
        cn0 = df.dropna(subset=["cn0"])
        cn0_bins = np.arange(t.min(), t.max() + 5000, 5000)
        cn0_mean = [cn0[(cn0["time_ms"] >= b) & (cn0["time_ms"] < b + 5000)]["cn0"].mean() for b in cn0_bins[:-1]]
        centers = to_dt64(bins[:-1] + 2500)
        axes[0].plot(centers, counts, lw=1, label=f"{name} ({len(df)} raw measurements, "
                                            f"{df['svid'].nunique()} satellites, mean Cn0 "
                                            f"{cn0['cn0'].mean() if len(cn0) else float('nan'):.1f} dB-Hz)")
        axes[1].plot(centers, cn0_mean, lw=1, label=name)
    axes[0].set_ylabel("raw measurements per 5 s")
    axes[1].set_ylabel("mean Cn0 (dB-Hz) per 5 s")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[0].set_title("Raw GNSS measurements of both apps")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    plot_time_axis(axes[1])
    fig.tight_layout()
    p = os.path.join(args.out, "07_raw_overview.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    say(f"  saved plot: {p}")

    # ------------------------------------------------------------- summary
    say("\n" + "=" * 78)
    say("SUMMARY - answers to the two questions")
    say("=" * 78)
    for name, app in apps:
        h_fix = health.get((name, "Fix"), {})
        say(f"  {name}: Fix 1970-jumps={h_fix.get('n_epoch', 0)}, backward jumps={h_fix.get('n_backward', 0)}, gaps={h_fix.get('n_gap', 0)}")
        for sensor in sorted(set(app["imu"]["sensor"].unique()) if len(app["imu"]) else []):
            h = health.get((name, f"IMU:{sensor}"), {})
            say(f"  {name}: IMU {sensor}: gaps={h.get('n_gap', 0)} (threshold {h.get('gap_thr_ms', 0):.0f} ms), "
                f"max dt={h.get('max_dt', 0):.0f} ms, 1970-jumps={h.get('n_epoch', 0)}, backward={h.get('n_backward', 0)}")
    say("\nAll plots and tables are in: " + os.path.abspath(args.out))
    say("(report.txt contains this same text)")

    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
