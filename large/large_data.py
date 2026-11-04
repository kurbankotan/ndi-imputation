# -*- coding: utf-8 -*-
"""Loaders for the million-row experiments.  Every dataset is cached once as float32 .npy (features) and .npy (labels)
in DATA_DIR (default large/data; on Colab, a folder on Google Drive), see get_large_data.py for the downloads.

    MiniBooNE          130,064 x 50   UCI 199   particle identification (rows with the -999 placeholder dropped)
    YearPredictionMSD  515,345 x 90   UCI 203   audio features; label = release year (decade bins for the downstream task)
    covertype          581,012 x 10   sklearn   the ten quantitative columns (the 44 binary indicator columns are one-hot
                                                 encodings of two categorical variables and are left out)
    KDDCup99         4,898,431 x 34   sklearn   numeric, non-constant columns (the 3 categorical columns are left out)
    SUSY             5,000,000 x 18   UCI 279
    HEPMASS         10,500,000 x 28   UCI 347   train + test files; the 'mass' column is a feature
    HIGGS           11,000,000 x 28   UCI 280
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
ORDER = ["MiniBooNE", "YearPredictionMSD", "covertype", "KDDCup99", "SUSY", "HEPMASS", "HIGGS"]


def synthetic(n=20_000, p=20, seed=1):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, 3)); L = rng.standard_normal((3, p))
    X = (F @ L + 0.5 * rng.standard_normal((n, p))).astype(np.float32)
    y = (F[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    return X, y


def load(name):
    if name == "synthetic":
        return synthetic()
    xf, yf = os.path.join(DATA_DIR, name + "_X.npy"), os.path.join(DATA_DIR, name + "_y.npy")
    if not os.path.exists(xf):
        import get_large_data
        get_large_data.fetch(name)
    X = np.load(xf)
    y = np.load(yf, allow_pickle=True) if os.path.exists(yf) else None
    return X, y


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        return "none"
