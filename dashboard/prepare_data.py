# makes the cleaned data files (data/processed) from the raw data, same steps as notebook 02.
# the processed files are not in git, so on a new server the app calls make_processed() once at start.
# can also be run by hand: python prepare_data.py

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data/raw"
OUT = BASE / "data/processed"


def add_features(df, country):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["weekend"] = (df.index.dayofweek >= 5).astype(int)
    hol = holidays.country_holidays(country, years=range(df.index.year.min(), df.index.year.max() + 1))
    df["holiday"] = df.index.normalize().isin(pd.to_datetime(list(hol.keys()))).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def find_spikes(s, train_end, k=10):
    d = s - (s.shift(1) + s.shift(-1)) / 2
    med = d[:train_end].median()
    mad = (d[:train_end] - med).abs().median()
    z = (d - med) / (1.4826 * mad)
    return z.abs() > k


def clean_pjm():
    pjm = pd.read_csv(RAW / "pjm/PJME_hourly.csv", parse_dates=["Datetime"])
    pjm = pjm.groupby("Datetime")["PJME_MW"].mean().asfreq("h").interpolate()
    train_end = pjm.index[int(len(pjm) * 0.7) - 1]
    pjm[find_spikes(pjm, train_end)] = np.nan
    return pjm.interpolate().to_frame("load")


def clean_uci():
    txt = RAW / "uci_household/household_power_consumption.txt"
    src = txt if txt.exists() else RAW / "uci_household/household_power.zip"   # pandas can read the zip directly
    uci = pd.read_csv(src, sep=";", na_values="?", low_memory=False,
                      usecols=["Date", "Time", "Global_active_power"])
    uci.index = pd.to_datetime(uci["Date"] + " " + uci["Time"], format="%d/%m/%Y %H:%M:%S")
    uci = uci["Global_active_power"].resample("h").mean()
    filled = uci.interpolate(limit=3, limit_area="inside")
    train_end = uci.index[int(len(uci) * 0.7) - 1]
    tr = uci[:train_end]
    profile = tr.groupby([tr.index.dayofweek, tr.index.hour]).mean()
    miss = filled.isna()
    idx = filled[miss].index
    filled[miss] = profile.loc[list(zip(idx.dayofweek, idx.hour))].values
    return filled.to_frame("load")


def split(df, train=0.7, val=0.15):
    a, b = int(len(df) * train), int(len(df) * (train + val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def make_processed():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, clean, country in [("pjme", clean_pjm, "US"), ("uci", clean_uci, "FR")]:
        if all((OUT / f"{name}_{p}.parquet").exists() for p in ["train", "val", "test"]):
            continue
        df = add_features(clean(), country)
        for part, d in zip(["train", "val", "test"], split(df)):
            d.to_parquet(OUT / f"{name}_{part}.parquet")
        print("made", name)


if __name__ == "__main__":
    make_processed()
