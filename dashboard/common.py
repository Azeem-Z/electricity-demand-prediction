# shared helpers for the dashboard pages

from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent      # the code/ folder

MODELS = ["naive", "rf", "xgb", "lstm", "gru", "cnn_lstm", "attention"]
NAMES = {"naive": "Seasonal naive", "rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM",
         "gru": "GRU", "cnn_lstm": "CNN-LSTM", "attention": "Attention (TFT)"}
UNIT = {"pjme": "MW", "uci": "kW"}
LABEL = {"pjme": "PJM East (regional)", "uci": "UCI household"}
HORIZONS = [1, 24, 168]


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
