import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import dataset_picker, features, load_full, load_model, load_pred, load_split, UNIT, LABEL

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
""")

h = st.radio("How far ahead to predict", [1, 24, 168], index=1, horizontal=True,
             format_func=lambda x: {1: "1 hour ahead", 24: "1 day ahead", 168: "1 week ahead"}[x])
st.caption({1: "Each hour is predicted using the data known one hour before it.",
            24: "Each hour is predicted using the data known one day (24 hours) before it.",
            168: "Each hour is predicted using the data known one week (168 hours) before it."}[h])

model = load_model(name, h)
X = features(name, h)
y = load_full(name)["load"]
test = load_split(name, "test")
X_test = X[X.index >= test.index.min()]
first_day = X_test.index.min().normalize() + pd.Timedelta(days=1)
last_day = X_test.index.max().normalize()

c1, c2 = st.columns(2)
start = c1.date_input("First day to show", first_day.date(), min_value=first_day.date(), max_value=last_day.date(),
                      help="Only dates in the test period can be chosen. The model never saw these dates in training.")
days = c2.slider("How many days to show", 1, 14, 7, help="7 = one week of hourly predictions")
st.caption(f"Available dates: {first_day:%d %b %Y} – {last_day:%d %b %Y} (test period).")

if st.button("Predict", type="primary"):
    rows = X_test.loc[pd.Timestamp(start): pd.Timestamp(start) + pd.Timedelta(days=days) - pd.Timedelta(hours=1)]
    pred = model.predict(rows)
    actual = y.loc[rows.index].values
    err = pred - actual

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows.index, y=actual, name="Actual demand", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=rows.index, y=pred, name="Predicted demand", line=dict(color="#2563eb")))
    fig.update_layout(yaxis_title=f"Demand ({unit})", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # error in % = average error / average demand
    pct = 100 * np.mean(np.abs(err)) / np.mean(actual)
    p = load_pred(name, "xgb", h)
    pct_all = 100 * (p["seed_0"] - p["actual"]).abs().mean() / p["actual"].mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Error in % (WAPE)", f"{pct:.1f} %",
              help="WAPE = average error (MAE) / average demand of the chosen days x 100. "
                   "5 % means the prediction is on average 5 % away from the actual demand. "
                   "Normal MAPE is not used, because household demand is close to zero at night.")
    m2.metric("Average error (MAE)", f"{np.mean(np.abs(err)):,.2f} {unit}",
              help="On average, the prediction was this far away from the actual demand.")
    m3.metric("RMSE", f"{np.sqrt(np.mean(err ** 2)):,.2f} {unit}",
              help="Like the average error, but large mistakes count more.")
    st.caption(f"For comparison: over the whole test period the error is {pct_all:.1f} % for this dataset and horizon.")

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
