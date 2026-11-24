import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import dataset_picker, extended_features, load_model, load_pred, load_split, UNIT, LABEL

name = dataset_picker()
unit = UNIT[name]

st.title("⚡ Predict electricity demand")
st.markdown(f"""
This page uses **XGBoost**, the model that was best or equally best in most tests of the study, to predict the electricity demand of **{LABEL[name]}**
and compares the prediction with the demand that really happened.

**How to use it**
1. Choose **how far ahead** the model should predict.
2. Choose the **first day** you want to see and **how many days** to show.
3. Click **Predict**. The model predicts the demand for every hour of those days.
""" + ("""
For PJM East you can choose any day up to today, and the next days as a forecast. After 2018 the actual demand
comes from the U.S. Energy Information Administration (EIA) and is downloaded automatically.
""" if name == "pjme" else ""))

h = st.radio("How far ahead to predict", [1, 24, 168], index=1, horizontal=True,
             format_func=lambda x: {1: "1 hour ahead", 24: "1 day ahead", 168: "1 week ahead"}[x])
st.caption({1: "Each hour is predicted using the data known one hour before it.",
            24: "Each hour is predicted using the data known one day (24 hours) before it.",
            168: "Each hour is predicted using the data known one week (168 hours) before it."}[h])

model = load_model(name, h)
X, y, info = extended_features(name, h)
test = load_split(name, "test")
X_show = X[X.index >= test.index.min()]
first_day = X_show.index.min().normalize() + pd.Timedelta(days=1)
last_day = X_show.index.max().normalize()
last_actual = info["last_actual"]

if name == "pjme":
    default = max(first_day, last_actual.normalize() - pd.Timedelta(days=5))   # last days + the forecast
    date_help = ("Test period of the study: Feb 2016 - Aug 2018. From 18 Sep 2018: actual demand from the EIA. "
                 "After the newest actual hour: forecast only. There is no data between 3 Aug and 18 Sep 2018.")
else:
    default = first_day
    date_help = "Only dates in the test period can be chosen. The model never saw these dates in training."

c1, c2 = st.columns(2)
start = c1.date_input("First day to show", default.date(), min_value=first_day.date(), max_value=last_day.date(),
                      help=date_help)
days = c2.slider("How many days to show", 1, 14, 7, help="7 = one week of hourly predictions")

if name == "pjme":
    st.caption(f"Available dates: {first_day:%d %b %Y} – {last_day:%d %b %Y}. "
               f"Newest actual demand: {last_actual:%d %b %Y %H:%M}. "
               f"Forecast up to: {X_show.index.max():%d %b %Y %H:%M} ({h} hours after the newest actual hour).")
    if not info["updated"]:
        st.caption("⚠️ The newest days could not be downloaded from the EIA, so the saved copy of the data is used.")
else:
    st.caption(f"Available dates: {first_day:%d %b %Y} – {last_day:%d %b %Y} (test period). "
               "The household data ends in Nov 2010, there is no newer data for this house.")

if st.button("Predict", type="primary"):
    rows = X_show.loc[pd.Timestamp(start): pd.Timestamp(start) + pd.Timedelta(days=days) - pd.Timedelta(hours=1)]
    if rows.empty:
        st.warning("There is no data for these days (gap between the study data and the EIA data, Aug - Sep 2018). "
                   "Please choose another day.")
        st.stop()
    pred = model.predict(rows)
    actual = y.reindex(rows.index).values
    known = ~np.isnan(actual)          # False = future hour, no actual demand yet
    err = pred - actual

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows.index, y=actual, name="Actual demand", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=rows.index, y=np.where(known, pred, np.nan), name="Predicted demand",
                             line=dict(color="#2563eb")))
    if (~known).any():
        fig.add_trace(go.Scatter(x=rows.index, y=np.where(known, np.nan, pred), name="Forecast (no actual yet)",
                                 line=dict(color="#ea8a0a", dash="dash")))
    fig.update_layout(yaxis_title=f"Demand ({unit})", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if (~known).any():
        st.info(f"The orange dashed line is a real forecast: these {(~known).sum()} hours are after the newest "
                f"actual demand ({last_actual:%d %b %Y %H:%M}), so there is nothing to compare with yet.")

    if known.any():
        e, a = err[known], actual[known]
        # error in % = average error / average demand
        pct = 100 * np.mean(np.abs(e)) / np.mean(a)
        p = load_pred(name, "xgb", h)
        pct_all = 100 * (p["seed_0"] - p["actual"]).abs().mean() / p["actual"].mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Error in % (WAPE)", f"{pct:.1f} %",
                  help="WAPE = average error (MAE) / average demand of the chosen days x 100. "
                       "5 % means the prediction is on average 5 % away from the actual demand. "
                       "Normal MAPE is not used, because household demand is close to zero at night.")
        m2.metric("Average error (MAE)", f"{np.mean(np.abs(e)):,.2f} {unit}",
                  help="On average, the prediction was this far away from the actual demand.")
        m3.metric("RMSE", f"{np.sqrt(np.mean(e ** 2)):,.2f} {unit}",
                  help="Like the average error, but large mistakes count more.")
        st.caption(f"For comparison: over the whole test period the error is {pct_all:.1f} % for this dataset and horizon.")

    if name == "pjme" and rows.index.max() > info["study_end"]:
        st.caption("After Aug 2018 the actual demand is the sum of the 11 Mid-Atlantic zones of PJM (= PJM East), "
                   "from the U.S. EIA Hourly Electric Grid Monitor. The model was trained on Aug 2009 - Aug 2013, "
                   "so for recent years the error can be a bit higher than in the study.")

    if name == "uci":
        st.info("Household demand is very spiky (kettle, oven, washing machine). No model can know when these happen, "
                "so the prediction shows the usual level for each hour, not every spike. That is why the error in % is high.")
    elif h == 1:
        st.info("1 hour ahead the model knows the demand of the last hours, so the prediction is very close to the actual demand.")
    elif h == 24:
        st.info("1 day ahead the prediction follows the daily shape well, but can miss some peaks, because the model does not know the weather.")
    else:
        st.info("1 week ahead the model only knows the demand from one week before, so changes during the week "
                "(e.g. a heatwave) are not seen and the error is larger.")

    # future hours have no actual, so actual / error stay empty in the table
    out = pd.DataFrame({"predicted": pred.round(3), "actual": actual, "error": err.round(3),
                        "error_pct": (100 * err / actual).round(1)}, index=rows.index)
    out.index.name = "hour"
    with st.expander("Show the numbers"):
        st.dataframe(out, use_container_width=True, column_config={
            "hour": st.column_config.Column("Hour", help="Date and hour of the prediction"),
            "predicted": st.column_config.NumberColumn(f"Predicted ({unit})", help="Demand predicted by the model"),
            "actual": st.column_config.NumberColumn(f"Actual ({unit})", help="Demand that really happened"),
            "error": st.column_config.NumberColumn(f"Error ({unit})",
                help="Predicted minus actual. Positive = prediction too high, negative = too low."),
            "error_pct": st.column_config.NumberColumn("Error (%)", format="%.1f %%",
                help="Error of this hour as a percentage of the actual demand of this hour: error / actual x 100. "
                     "Example: actual 100, predicted 105 -> +5 %. When the actual demand is very small "
                     "(household at night) this percentage can become very large."),
        })
        st.download_button("Download as CSV", out.to_csv().encode(), f"prediction_{name}_{h}h.csv", "text/csv")
