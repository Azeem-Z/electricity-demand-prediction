# shared helpers for the dashboard pages

# why: every page needs the same data files and names, so the loading code is kept here in one place
from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

# all files are found from this folder, so the app works no matter where it is started from
BASE = Path(__file__).resolve().parent.parent      # the code/ folder

# short model names used in the result files, and the long names shown on the screen
MODELS = ["naive", "rf", "xgb", "lstm", "gru", "cnn_lstm", "attention"]
NAMES = {"naive": "Seasonal naive", "rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM",
         "gru": "GRU", "cnn_lstm": "CNN-LSTM", "attention": "Attention (TFT)"}
# unit and full name of each dataset
UNIT = {"pjme": "MW", "uci": "kW"}
LABEL = {"pjme": "PJM East (regional)", "uci": "UCI household"}
# forecast horizons in hours: 1 hour, 1 day, 1 week
HORIZONS = [1, 24, 168]


# dataset_picker: shows the dataset dropdown in the left sidebar and gives back "pjme" or "uci".
# format_func = show the long name on the screen but keep the short name in the code
def dataset_picker(key="dataset"):
    return st.sidebar.selectbox("Dataset", ["pjme", "uci"], format_func=lambda x: LABEL[x], key=key)


# st.cache_data = run the function once and remember the result, so the file is not read again on every click
@st.cache_data
# load_scores: test scores of all models (tree models from notebook 04 + deep models from notebook 05) in one table
def load_scores():
    return pd.concat([pd.read_csv(BASE / "results/baseline_scores.csv"),
                      pd.read_csv(BASE / "results/deep_scores.csv")])


@st.cache_data
# load_csv: reads any csv file from the code/ folder, e.g. load_csv("results/dm_tests.csv")
def load_csv(path):
    return pd.read_csv(BASE / path)


@st.cache_data
# load_split: reads one part of a dataset, e.g. load_split("pjme", "test")
def load_split(name, part):
    return pd.read_parquet(BASE / f"data/processed/{name}_{part}.parquet")


@st.cache_data
# load_full: train + val + test joined into one table in time order
def load_full(name):
    return pd.concat([load_split(name, p) for p in ["train", "val", "test"]])


@st.cache_data
# load_pred: saved test predictions of one model, e.g. load_pred("uci", "xgb", 24)
def load_pred(name, model, h):
    return pd.read_parquet(BASE / f"results/predictions/{name}_{model}_h{h}.parquet")


# st.cache_resource is like cache_data but for objects such as a trained model
@st.cache_resource
# load_model: loads the XGBoost model saved in notebook 07 for one dataset and horizon
def load_model(name, h):
    m = xgb.XGBRegressor()
    m.load_model(BASE / f"models/xgb_{name}_h{h}.json")
    return m


# make_features: builds the input columns for the model (lags, rolling means, calendar).
# it must be the same as in notebooks 04 and 07, or the saved model would get wrong inputs
def make_features(df, h):
    # same features as notebook 04 - only values known h hours before the target hour
    X = pd.DataFrame(index=df.index)
    # e.g. h = 24 -> lags 24, 25, ..., 29, 48, 168, 336
    lags = [h + k for k in range(6)] + [m for m in [24, 48, 168, 336] if m >= h]
    for s in sorted(set(lags)):
        # shift(s) = the load from s hours before
        X[f"lag_{s}"] = df["load"].shift(s)
    # average of the last 24 (and 168) known hours
    X["roll_mean_24"] = df["load"].shift(h).rolling(24).mean()
    X["roll_mean_168"] = df["load"].shift(h).rolling(168).mean()
    for c in ["hour", "dayofweek", "month", "dayofyear", "weekend", "holiday"]:
        X[c] = df[c]
    # remove the first rows that have no past values
    return X.dropna()


@st.cache_data
# features: the feature table for the whole dataset, cached so it is only built once
def features(name, h):
    return make_features(load_full(name), h)
