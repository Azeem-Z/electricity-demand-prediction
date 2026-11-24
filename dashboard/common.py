# shared helpers for the dashboard pages

from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

from eia_data import load_eia
from prepare_data import add_features, make_processed

BASE = Path(__file__).resolve().parent.parent      # the code/ folder

MODELS = ["naive", "rf", "xgb", "lstm", "gru", "cnn_lstm", "attention"]
NAMES = {"naive": "Seasonal naive", "rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM",
         "gru": "GRU", "cnn_lstm": "CNN-LSTM", "attention": "Attention (TFT)"}
UNIT = {"pjme": "MW", "uci": "kW"}
LABEL = {"pjme": "PJM East (regional)", "uci": "UCI household"}
HORIZONS = [1, 24, 168]


@st.cache_resource(show_spinner="Preparing the data (only the first time, about 1 minute)...")
def ensure_data():
    # the cleaned data is not in git, so on a new server it is made from the raw data once
    make_processed()
    return True


def dataset_picker(key="dataset"):
    return st.sidebar.selectbox("Dataset", ["pjme", "uci"], format_func=lambda x: LABEL[x], key=key)


@st.cache_data
def load_scores():
    return pd.concat([pd.read_csv(BASE / "results/baseline_scores.csv"),
                      pd.read_csv(BASE / "results/deep_scores.csv")])


@st.cache_data
def load_csv(path):
    return pd.read_csv(BASE / path)


@st.cache_data
def load_split(name, part):
    return pd.read_parquet(BASE / f"data/processed/{name}_{part}.parquet")


@st.cache_data
def load_full(name):
    return pd.concat([load_split(name, p) for p in ["train", "val", "test"]])


@st.cache_data
def load_pred(name, model, h):
    return pd.read_parquet(BASE / f"results/predictions/{name}_{model}_h{h}.parquet")


@st.cache_resource
def load_model(name, h):
    m = xgb.XGBRegressor()
    m.load_model(BASE / f"models/xgb_{name}_h{h}.json")
    return m


def make_features(df, h):
    # same features as notebook 04 - only values known h hours before the target hour
    X = pd.DataFrame(index=df.index)
    lags = [h + k for k in range(6)] + [m for m in [24, 48, 168, 336] if m >= h]
    for s in sorted(set(lags)):
        X[f"lag_{s}"] = df["load"].shift(s)
    X["roll_mean_24"] = df["load"].shift(h).rolling(24).mean()
    X["roll_mean_168"] = df["load"].shift(h).rolling(168).mean()
    for c in ["hour", "dayofweek", "month", "dayofyear", "weekend", "holiday"]:
        X[c] = df[c]
    return X.dropna()


@st.cache_data
def features(name, h):
    return make_features(load_full(name), h)


@st.cache_data(ttl=6 * 3600, show_spinner="Getting the newest PJM data from the EIA...")
def load_extended(name):
    # study data + (for PJM) the EIA data from Sep 2018 to today + one empty week at the end.
    # the empty week has no actual demand yet, but the lags are known, so the model can forecast it.
    # downloaded again at most every 6 hours (ttl)
    study = load_full(name)["load"]
    info = {"study_end": study.index.max(), "last_actual": study.index.max(), "updated": False}
    if name == "pjme":
        eia, info["updated"] = load_eia()
        s = pd.concat([study, eia[eia.index > study.index.max()]]).asfreq("h")    # gap Aug-Sep 2018 stays empty
        info["eia_start"] = eia.index.min()
        info["last_actual"] = s.last_valid_index()
        future = pd.date_range(info["last_actual"] + pd.Timedelta(hours=1), periods=168, freq="h")
        s = pd.concat([s, pd.Series(index=future, dtype=float)])
        df = add_features(s.to_frame("load"), "US")
    else:
        df = load_full(name)
    return df, info


def extended_features(name, h):
    df, info = load_extended(name)
    X = make_features(df, h)
    # keep hours that have an actual value, plus the real future after the last actual hour
    keep = df["load"].reindex(X.index).notna() | (X.index > info["last_actual"])
    return X[keep], df["load"], info
