import json

import pandas as pd
import plotly.express as px
import streamlit as st

from common import dataset_picker, load_scores, load_csv, BASE, MODELS, NAMES, UNIT, LABEL

name = dataset_picker()
unit = UNIT[name]
st.title(f"Results – {LABEL[name]}")
st.markdown("All seven models were tested on the **test period** (data they never saw in training). "
            "Each model predicted the demand **1 hour**, **1 day (24h)** and **1 week (168h)** ahead.")

with st.expander("ℹ️ The models compared"):
    st.markdown("""
| Model | In simple words |
|---|---|
| Seasonal naive | simplest guess: "demand will be the same as at this hour one week ago" – used as a minimum benchmark |
| Random Forest | many decision trees that vote together |
| XGBoost | decision trees built one after another, each fixing the mistakes of the previous ones |
| LSTM | neural network with memory, reads the last hours one by one |
| GRU | a simpler and faster version of the LSTM |
| CNN-LSTM | first finds short local patterns (CNN), then reads them in order (LSTM) |
| Attention (TFT) | neural network that learns which past hours to pay most attention to |
""")

t1, t2, t3, t4 = st.tabs(["Model accuracy", "Errors by period", "Significance tests", "Attention weights"])

# ---------------- accuracy ----------------
METRIC_HELP = {
    "rmse": f"Root Mean Squared Error, in {unit}. Typical size of the prediction error, where big mistakes count more. Lower is better.",
    "mae": f"Mean Absolute Error, in {unit}. On average, how far the prediction is from the actual demand. Lower is better.",
    "r2": "R² (R-squared), from 0 to 1. How much of the ups and downs in demand the model explains. 1 = perfect, 0 = no better than always guessing the average. Higher is better.",
    "mape": "Mean Absolute Percentage Error, in %. Average error as a percentage of the actual demand. Lower is better.",
}
with t1:
    st.markdown("How close were the predictions to the real demand?")
    metric = st.selectbox("Measure", ["rmse", "mae", "r2", "mape"],
                          format_func={"rmse": "RMSE – typical error", "mae": "MAE – average error",
                                       "r2": "R² – how much is explained", "mape": "MAPE – average error in %"}.get,
                          help="Pick how to measure accuracy. Hover over the ? icons for an explanation.")
    st.info(METRIC_HELP[metric])

    s = load_scores()
    s = s[s["dataset"] == name]
    t = s.groupby(["model", "horizon"])[metric].mean().unstack().loc[MODELS].rename(index=NAMES)
    t.columns = ["1 hour ahead", "1 day ahead", "1 week ahead"]
    best = t.idxmax() if metric == "r2" else t.idxmin()
    st.dataframe(
        t.style.format("{:.3f}").highlight_max(axis=0, color="#bbf7d0") if metric == "r2"
        else t.style.format("{:.3f}").highlight_min(axis=0, color="#bbf7d0"),
        use_container_width=True,
        column_config={
            "_index": st.column_config.Column("Model", help="Name of the forecasting model"),
            "1 hour ahead": st.column_config.Column(help="Each hour predicted using data known 1 hour before"),
            "1 day ahead": st.column_config.Column(help="Each hour predicted using data known 24 hours before"),
            "1 week ahead": st.column_config.Column(help="Each hour predicted using data known 168 hours before"),
        })
    st.caption("Green = best model for that horizon. Values are the average of 3 training runs (seeds).")
    if metric == "mape" and name == "uci":
        st.warning("For the household, MAPE looks very large because night demand is close to zero, "
                   "so a small error becomes a big percentage. RMSE and MAE are more useful here.")

    plot_df = s.groupby(["model", "horizon"])[metric].mean().reset_index()
    plot_df["model"] = plot_df["model"].map(NAMES)
    plot_df["horizon"] = plot_df["horizon"].map({1: "1 hour ahead", 24: "1 day ahead", 168: "1 week ahead"})
    fig = px.bar(plot_df, x="horizon", y=metric, color="model", barmode="group",
                 labels={metric: metric.upper().replace("R2", "R²"), "horizon": ""})
    st.plotly_chart(fig, use_container_width=True)
    if name == "pjme":
        st.markdown("**In short:** for the region, the tree models (Random Forest, XGBoost) are clearly best **1 hour ahead**. "
                    "**1 day ahead** GRU, Random Forest and XGBoost are equally good, and **1 week ahead** the neural networks "
                    "and the tree models are equally good (see the significance tests). "
                    "All models get worse the further ahead they predict.")
    else:
        st.markdown("**In short:** for the household, XGBoost was the most accurate model at every horizon, and the "
                    "difference to the other models is real (see the significance tests). "
                    "All models get worse the further ahead they predict.")

# ---------------- errors by period ----------------
GROUPS = {"hour": "Hour of the day", "day_type": "Type of day (weekday / weekend / holiday)",
          "level": "Demand level (low / normal / peak)", "season": "Season"}
with t2:
    st.markdown("One average error can hide when a model struggles. Here the error (RMSE) is split by **when** the demand happened.")
    c1, c2 = st.columns(2)
    h = c1.selectbox("How far ahead", [1, 24, 168], index=1, key="h_period",
                     format_func={1: "1 hour ahead", 24: "1 day ahead", 168: "1 week ahead"}.get)
    by = c2.selectbox("Split the error by", list(GROUPS), format_func=GROUPS.get,
                      help="Peak = the 10% of hours with the highest demand, low = the 10% with the lowest demand.")
    chosen = st.multiselect("Models", MODELS[1:], default=["xgb", "gru"], format_func=lambda m: NAMES[m])

    b = load_csv("results/error_by_period.csv")
    t = b[(b["dataset"] == name) & (b["horizon"] == h) & (b["by"] == by) & (b["model"].isin(chosen))].copy()
    t["model"] = t["model"].map(NAMES)
    if by == "hour":
        t["group"] = t["group"].astype(int)
        fig = px.line(t.sort_values("group"), x="group", y="rmse", color="model", markers=True)
    else:
        fig = px.bar(t, x="group", y="rmse", color="model", barmode="group")
    fig.update_layout(xaxis_title=GROUPS[by], yaxis_title=f"Error – RMSE ({unit})")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher bar / line = bigger errors in that period. "
               + ("The household test period has no winter." if name == "uci" else ""))
    st.markdown("**In short:** errors are largest in the **evening peak hours** and in **peak-demand periods** – "
                "the hours where demand changes the most.")
    if name == "pjme":
        st.markdown("For the region, the neural networks (e.g. GRU) are better than XGBoost in the **peak-demand hours** "
                    "1 day and 1 week ahead, while XGBoost is better on **holidays** and in normal hours.")

# ---------------- significance ----------------
with t3:
    st.markdown("""
Two models can have slightly different errors just by chance. The **Diebold–Mariano test** checks whether the
difference between two models is **real**.
- **p-value below 0.05** → the difference is real (statistically significant).
- **p-value 0.05 or higher** → the two models are not really different.
""")
    h = st.selectbox("How far ahead", [1, 24, 168], key="h_dm",
                     format_func={1: "1 hour ahead", 24: "1 day ahead", 168: "1 week ahead"}.get)
    dm = load_csv("results/dm_tests.csv")
    t = dm[(dm["dataset"] == name) & (dm["horizon"] == h)].copy()
    t = pd.DataFrame({
        "Model A": t["model_a"].map(NAMES),
        "Model B": t["model_b"].map(NAMES),
        "p-value": t["p_value"],
        "Better model": t["better"].map(NAMES),
        "Real difference?": t["significant"].map({True: "Yes", False: "No"}),
    })
    st.dataframe(t.style.format({"p-value": "{:.4f}"}), hide_index=True, use_container_width=True,
                 column_config={
                     "p-value": st.column_config.Column(help="Chance that the difference is only luck. Below 0.05 = real difference."),
                     "Better model": st.column_config.Column(help="The model with the smaller errors"),
                     "Real difference?": st.column_config.Column(help="Yes if p-value < 0.05"),
                 })

# ---------------- attention ----------------
with t4:
    st.markdown("""
The **attention model** looks at the last hours before it makes a prediction and gives each hour a **weight**
(how much attention it pays to that hour). If one hour mattered much more than the others, its bar would be much taller.
""")
    h = st.selectbox("How far ahead", [1, 24, 168], key="h_att",
                     format_func={1: "1 hour ahead", 24: "1 day ahead", 168: "1 week ahead"}.get)
    a = load_csv(f"results/attention_{name}_h{h}.csv")
    fig = px.bar(a, x="hours_before_origin", y="weight", labels={"weight": "attention weight"})
    fig.add_hline(y=1 / len(a), line_dash="dash", annotation_text="equal attention to every hour")
    fig.update_xaxes(autorange="reversed", title="hours before the prediction was made")
    st.plotly_chart(fig, use_container_width=True)
    with open(BASE / "results/deep_params.json") as f:
        L = json.load(f)[name]["attention"]["L"]
    st.markdown(f"**In short:** the model looks at the last {L} hours. All bars are close to the dashed line, "
                "so the model spreads its attention almost equally – the weights do not clearly show which hours matter most.")
