#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download and cache the large datasets as float32 .npy files in DATA_DIR (about 3 GB of .npy in all; the UCI
downloads are 4 GB compressed and are deleted after conversion unless KEEP_RAW=1).

    python large/get_large_data.py                    # all seven
    python large/get_large_data.py SUSY HIGGS         # a subset
"""
import os, sys, gzip, io, time, zipfile, urllib.request
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from large_data import DATA_DIR, ORDER

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
URLS = {"MiniBooNE": [UCI + "00199/MiniBooNE_PID.txt"],
        "YearPredictionMSD": [UCI + "00203/YearPredictionMSD.txt.zip"],
        "SUSY": [UCI + "00279/SUSY.csv.gz"],
        "HEPMASS": [UCI + "00347/all_train.csv.gz", UCI + "00347/all_test.csv.gz"],
        "HIGGS": [UCI + "00280/HIGGS.csv.gz"]}


def download(url, path, tries=3):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    for attempt in range(tries):
        try:
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, path + ".part"); os.replace(path + ".part", path)
            print(f"  {os.path.getsize(path) / 1e6:.0f} MB", flush=True); return path
        except Exception as e:
            print(f"  retry {attempt + 1}/{tries}: {type(e).__name__}: {e}", flush=True); time.sleep(10)
    raise SystemExit(f"could not download {url}")


def read_csv_float32(path, skiprows=0, usecols=None, chunksize=1_000_000, sep=",", comment=None):
    import pandas as pd
    parts = []
    for chunk in pd.read_csv(path, header=None, skiprows=skiprows, usecols=usecols, dtype=np.float32, chunksize=chunksize, sep=sep, comment=comment):
        parts.append(chunk.to_numpy(np.float32)); print(f"  read {sum(len(p) for p in parts):,} rows", flush=True)
    return np.vstack(parts)


def save(name, X, y):
    os.makedirs(DATA_DIR, exist_ok=True)
    X = np.ascontiguousarray(X, dtype=np.float32)
    keep = X.std(0) > 0                                       # constant columns carry no information for imputation
    if (~keep).any(): print(f"  dropping {(~keep).sum()} constant column(s)")
    X = X[:, keep]
    np.save(os.path.join(DATA_DIR, name + "_X.npy"), X)
    if y is not None: np.save(os.path.join(DATA_DIR, name + "_y.npy"), np.asarray(y))
    print(f"{name}: {X.shape[0]:,} x {X.shape[1]} saved to {DATA_DIR}", flush=True)


def fetch(name):
    os.makedirs(DATA_DIR, exist_ok=True); raw = os.path.join(DATA_DIR, "raw"); os.makedirs(raw, exist_ok=True)
    if name == "MiniBooNE":
        path = download(URLS[name][0], os.path.join(raw, "MiniBooNE_PID.txt"))
        with open(path) as f: n_sig, n_bkg = map(int, f.readline().split())
        X = np.loadtxt(path, skiprows=1, dtype=np.float32)
        y = np.r_[np.ones(n_sig, int), np.zeros(n_bkg, int)]
        ok = ~(X <= -998).any(1)                                # 471 rows filled with the -999 placeholder
        save(name, X[ok], y[ok])
    elif name == "YearPredictionMSD":
        path = download(URLS[name][0], os.path.join(raw, "YearPredictionMSD.txt.zip"))
        with zipfile.ZipFile(path) as z:
            with z.open("YearPredictionMSD.txt") as f: A = read_csv_float32(io.BytesIO(f.read()))
        year = A[:, 0].astype(int); save(name, A[:, 1:], (year // 10) * 10)   # decade bins as the classification label
    elif name == "covertype":
        from sklearn.datasets import fetch_covtype
        d = fetch_covtype(); save(name, d.data[:, :10], d.target.astype(int))
    elif name == "KDDCup99":
        from sklearn.datasets import fetch_kddcup99
        d = fetch_kddcup99(percent10=False, as_frame=True)
        df = d.data.drop(columns=["protocol_type", "service", "flag"]).apply(lambda c: c.astype(np.float32))
        y = d.target.astype(str).str.strip("b'").to_numpy()
        save(name, df.to_numpy(np.float32), y)
    elif name == "SUSY":
        path = download(URLS[name][0], os.path.join(raw, "SUSY.csv.gz"))
        A = read_csv_float32(path); save(name, A[:, 1:], A[:, 0].astype(int))
    elif name == "HIGGS":
        path = download(URLS[name][0], os.path.join(raw, "HIGGS.csv.gz"))
        A = read_csv_float32(path); save(name, A[:, 1:], A[:, 0].astype(int))
    elif name == "HEPMASS":
        parts = []
        for url, fn in zip(URLS[name], ["all_train.csv.gz", "all_test.csv.gz"]):
            path = download(url, os.path.join(raw, fn)); parts.append(read_csv_float32(path, skiprows=1))   # header line "# label,f0,..."
        A = np.vstack(parts); save(name, A[:, 1:], A[:, 0].astype(int))
    else:
        raise ValueError(name)
    if os.environ.get("KEEP_RAW", "0") != "1":
        for f in os.listdir(raw):
            try: os.remove(os.path.join(raw, f))
            except OSError: pass


if __name__ == "__main__":
    for name in (sys.argv[1:] or ORDER):
        if os.path.exists(os.path.join(DATA_DIR, name + "_X.npy")): print(f"have {name}"); continue
        fetch(name)
