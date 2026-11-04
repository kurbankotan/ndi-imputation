#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download the local data used by the additional experiments into extra/data/ (about 40 MB):
the PMLB copies of the benchmark datasets and of the nine larger datasets (EpistasisLab/pmlb on GitHub) and the UCI
banknote-authentication file (or its mirror in jbrownlee/Datasets, byte-identical).  Files already present are skipped.  Usage: python extra/get_data.py"""
import os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from data_local import DATA_DIR, PMLB_DIR, PMLB_MAP, LARGE

PMLB_RAW = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{name}/{name}.tsv.gz"   # Git LFS media URL
BANKNOTE = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv"]            # same UCI file (mirror)


def fetch(urls, path, tries=2):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print("have ", os.path.basename(path)); return
    for url in ([urls] if isinstance(urls, str) else urls):
        for attempt in range(tries):
            try:
                urllib.request.urlretrieve(url, path + ".part"); os.replace(path + ".part", path)
                print("got  ", os.path.basename(path), f"{os.path.getsize(path)/1e6:.1f} MB"); return
            except Exception as e:
                print(f"retry {attempt + 1}/{tries} for {url}: {type(e).__name__}: {e}")
                time.sleep(5)
    raise SystemExit(f"could not download {path}")


if __name__ == "__main__":
    os.makedirs(PMLB_DIR, exist_ok=True)
    names = sorted(set(PMLB_MAP.values()) | {f for _, f in LARGE})
    for name in names:
        fetch(PMLB_RAW.format(name=name), os.path.join(PMLB_DIR, name + ".tsv.gz"))
    fetch(BANKNOTE, os.path.join(DATA_DIR, "data_banknote_authentication.txt"))
    print("done:", DATA_DIR)
