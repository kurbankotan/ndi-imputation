# -*- coding: utf-8 -*-
"""Dataset loaders for the additional experiments (version 4.1).

Benchmark datasets of the paper (the 15 OpenML tables of Table 2), in this order of preference:
  1. a CSV exported from OpenML with sklearn's fetch_openml in extra/openml_csv/<name>.csv (columns = features,
     plus "__target__"); the export is exact and needs no network at run time:
         from sklearn.datasets import fetch_openml; d = fetch_openml(data_id=ID, as_frame=True, parser="auto")
         df = d.data.copy(); df["__target__"] = d.target; df.to_csv("extra/openml_csv/<name>.csv", index=False)
  2. OpenML itself through ndi_benchmark.load_openml (needs network; identical to the main benchmark);
  3. the PMLB copy of the same UCI table (extra/data/pmlb/*.tsv.gz, downloaded by extra/get_data.py).  The PMLB
     copies of wine, ecoli, ionosphere, yeast, wine-quality-red, spambase and MagicTelescope reproduce the paper's
     NDI-F results exactly (extra/verify_datasets.py); those of iris, glass, wdbc, diabetes, vehicle and segment
     have a different row order, so they are NOT used unless PMLB_FALLBACK_ALL=1 is set.  banknote is read from
     the UCI file downloaded by get_data.py, qsar-biodeg has no PMLB copy.
The preprocessing (finish) is that of ndi_benchmark.load_openml: numerical columns, constant columns dropped,
classes with fewer than 10 instances removed, stratified sub-sample of 2,000 rows with random_state=0.

Larger datasets (Section 4.7, experiment (vi)): nine PMLB tables used without sub-sampling."""
import gzip, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("EXTRA_DATA_DIR", os.path.join(HERE, "data"))
PMLB_DIR = os.environ.get("PMLB_DIR", os.path.join(DATA_DIR, "pmlb"))
OPENML_DIR = os.environ.get("OPENML_DIR", os.path.join(HERE, "openml_csv"))
N_CAP = 2000

# paper name -> PMLB file (same UCI sources; the paper took them from OpenML)
PMLB_MAP = {"iris": "iris", "wine": "wine_recognition", "glass": "_deprecated_glass", "ecoli": "ecoli",
            "ionosphere": "ionosphere", "wdbc": "_deprecated_wdbc", "diabetes": "_deprecated_diabetes",
            "vehicle": "vehicle", "yeast": "yeast", "wine-quality-red": "wine_quality_red",
            "segment": "segmentation", "spambase": "spambase", "MagicTelescope": "magic"}
PMLB_VERIFIED = {"wine", "ecoli", "ionosphere", "yeast", "wine-quality-red", "spambase", "MagicTelescope"}
LARGE = [("magic-full", "magic"), ("letter", "letter"), ("pendigits", "pendigits"), ("satimage", "satimage"),
         ("optdigits", "optdigits"), ("texture", "texture"), ("musk2", "clean2"), ("mfeat-factors", "mfeat_factors"),
         ("mnist", "mnist")]
PAPER = ["iris", "wine", "glass", "ecoli", "ionosphere", "wdbc", "diabetes", "vehicle", "yeast", "banknote",
         "qsar-biodeg", "wine-quality-red", "segment", "spambase", "MagicTelescope"]


def read_pmlb(fname):
    path = os.path.join(PMLB_DIR, fname + ".tsv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run  python extra/get_data.py")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t")
    y = df["target"]; X = df.drop(columns=["target"])
    return X, y


def finish(name, X, y, source, cap=N_CAP, keep_constant=()):
    """the preprocessing of load_openml() in ndi_benchmark.py (constant columns are dropped on the full OpenML
    table, before the rare classes are removed; `keep_constant` restores columns that are constant only in the
    class-filtered PMLB copy, e.g. chg in ecoli)"""
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.loc[:, (X.nunique() > 1) | X.columns.isin(keep_constant)]
    y = pd.Series(y).astype(str)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    counts = y.value_counts()
    keep = y.isin(counts[counts >= 10].index)
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    if cap and len(X) > cap:
        X, _, y, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=0)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    y = pd.factorize(y)[0]
    return dict(name=name, natural=False, X_num=X.astype(float), X_cat=None, y=y, source=source)


def _openml_ids():
    import ndi_benchmark as nb
    return dict(nb.OPENML)


def load_paper_dataset(name):
    csv = os.path.join(OPENML_DIR, f"{name}.csv")
    if os.path.exists(csv):                                   # 1. exact OpenML export
        df = pd.read_csv(csv)
        return finish(name, df.drop(columns=["__target__"]), df["__target__"], "OpenML (fetch_openml export)")
    if os.environ.get("NO_OPENML", "0") != "1":              # 2. OpenML through the benchmark's own loader
        try:
            import ndi_benchmark as nb
            return nb.load_openml(name, _openml_ids()[name])
        except Exception as e:
            print(f"!! OpenML not reachable for {name} ({type(e).__name__}); trying the local copy", flush=True)
    if name == "banknote":                                    # 3. local copies
        path = os.path.join(DATA_DIR, "data_banknote_authentication.txt")
        if not os.path.exists(path): raise FileNotFoundError(f"{path} not found: run  python extra/get_data.py")
        df = pd.read_csv(path, header=None); return finish(name, df.iloc[:, :4], df.iloc[:, 4], "UCI 267 (banknote file)")
    if name == "qsar-biodeg":
        raise FileNotFoundError("qsar-biodeg: no local copy (export OpenML 1494 to extra/openml_csv/qsar-biodeg.csv)")
    if name not in PMLB_VERIFIED and os.environ.get("PMLB_FALLBACK_ALL", "0") != "1":
        raise FileNotFoundError(f"{name}: the PMLB copy does not reproduce the paper's table (row order); export OpenML "
                                f"{_openml_ids().get(name, '?')} to extra/openml_csv/{name}.csv or set PMLB_FALLBACK_ALL=1")
    X, y = read_pmlb(PMLB_MAP[name])
    return finish(name, X, y, f"PMLB {PMLB_MAP[name]}", keep_constant=("chg",) if name == "ecoli" else ())


def load_paper_datasets(names=PAPER, strict=False):
    out = []
    for n in names:
        try: out.append(load_paper_dataset(n))
        except FileNotFoundError as e:
            if strict: raise
            print("!!", e, flush=True)
    return out


def load_large_dataset(name, cap=None):
    fname = dict(LARGE)[name]
    X, y = read_pmlb(fname)
    return finish(name, X, y, f"PMLB {fname}", cap=cap)
