# About page: explains the project (overview, problem statement, datasets).
# it is mostly text. only the table of the train / val / test periods at the end is made from the data
import pandas as pd
import streamlit as st

from common import load_split, LABEL

st.title("About the project")
st.markdown("**Deep Learning Based Electricity Demand Prediction with Temporal Pattern Analysis** – COM748 Masters Research Project")

# three tabs
tab1, tab2, tab3 = st.tabs(["🏠 Overview", "❓ Problem statement", "🗂️ Datasets"])

# ---------------- overview ----------------
with tab1:
    st.markdown("""
Electricity providers need accurate demand forecasts to plan generation, keep the grid stable and reduce costs.
This project compares tree-based and deep learning models for predicting electricity demand at **two scales**,
a regional grid and a single household, and uses **temporal pattern analysis** to explain where and why the models make errors.
""")

    # four small boxes with the key facts. delta_color="off" keeps the small text grey
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", "2", "PJM East + UCI household", delta_color="off")
    c2.metric("Models compared", "7", "incl. seasonal naive", delta_color="off")
    c3.metric("Predicts", "1 hour, 1 day, 1 week ahead", delta_color="off")
    c4.metric("Best overall", "Tree models", "deep models equal on 1 day / 1 week (region)", delta_color="off",
              help="Random Forest / XGBoost were best 1 hour ahead and for the household. For the region, "
                   "1 day and 1 week ahead, the best neural networks were equally good.")

    st.subheader("Aim")
    st.markdown("""
Design, build and evaluate a framework for electricity demand prediction that compares recurrent, hybrid,
attention-based and tree-based models on the PJM Hourly Energy Consumption data and the UCI Individual Household
Electric Power Consumption data, and that uses temporal pattern analysis to explain forecasting performance.
""")

    st.subheader("Research questions")
    st.markdown("""
- **RQ1** – Do deep learning models outperform tree-based models when all models use the same data split and evaluation protocol?
- **RQ2** – How does model performance change between regional and household data, and across one-hour, one-day and one-week horizons?
- **RQ3** – Which temporal patterns are linked to larger forecast errors, and can attention weights help to explain them?
""")

    st.subheader("What is in this app")
    st.markdown("""
| Page | What you find there |
|---|---|
| **Predict demand** | use XGBoost (best or equally best in most tests) to predict demand and compare it with the actual demand |
| Results | how accurate each model was, and when the models make bigger errors |
| Temporal patterns | how demand repeats every day, week and year, and why that matters |
| About the project | this page – overview, problem statement and datasets |
""")

# ---------------- problem statement ----------------
with tab2:
    st.markdown("""
Electricity demand forecasting is a core task in power system planning. Forecasts are used for generation
scheduling, reserve planning and energy trading, and they help to keep the grid stable. With more low-carbon
technology on the grid, reliable forecasts are needed at several network levels – from whole regions down to
single households.

Demand is hard to predict because many patterns act at the same time: it repeats over the **day**, the **week**
and the **year**, and it reacts to weather, public holidays and human behaviour. Household demand is much more
volatile than regional demand.

Deep learning models (LSTM, GRU, CNN-LSTM, Transformers) are now widely used for this task. But most studies test
**one new model on one dataset** and report **one overall error value**. It is therefore unclear which models work
best at different scales and horizons, and when and why they fail.
""")

    st.subheader("Research gaps")
    # two columns with two boxes each for the four research gaps
    c1, c2 = st.columns(2)
    with c1:
        st.info("**1. Single-scale evaluation**\n\nMost studies use one dataset at one network level, so it is unknown "
                "whether findings transfer between regional and household demand.")
        st.info("**3. Weak link between pattern analysis and modelling**\n\nDecomposition is used to build models, "
                "but seldom to explain the errors of different models.")
    with c2:
        st.info("**2. Overall error values only**\n\nOne average error hides how models behave at peak hours, "
                "at weekends, on holidays and as the horizon grows.")
        st.info("**4. Few controlled comparisons**\n\nTree-based, recurrent, hybrid and attention-based models are "
                "rarely compared with the same data split and statistical tests, even though tree models can be strong.")

    st.subheader("How this project addresses them")
    st.markdown("""
- two datasets at two scales: **PJM East** (regional grid) and **UCI** (one household)
- seven models tested on the **same data split** and in the same way
- errors reported **by horizon, hour, day type, demand level and season**, with a significance test
- **temporal pattern analysis** used to explain where the errors happen
""")

# ---------------- datasets ----------------
with tab3:
    st.markdown("Two public datasets are used, so the models can be tested on **two scales**: a whole region and a single house.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("PJM East – a whole region")
        st.markdown("""
- **What:** electricity demand of the PJM East region in the eastern USA, every hour, in **MW** (megawatts)
- **Period:** Jan 2002 – Aug 2018 (about 145,000 hours)
- **Source:** [Kaggle – PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
""")
    with c2:
        st.subheader("UCI – a single household")
        st.markdown("""
- **What:** electricity use of one house near Paris, every minute, in **kW** (kilowatts)
- **Period:** Dec 2006 – Nov 2010 (about 2 million readings)
- **Source:** [UCI Machine Learning Repository](https://doi.org/10.24432/C58K54)
""")

    st.subheader("How the data was cleaned")
    st.markdown("""
| Problem found | What was done |
|---|---|
| Household data is per minute | averaged to **one value per hour**, so both datasets have the same time step |
| Repeated hours in PJM (clocks going back in autumn) | the two values were **averaged** |
| Missing hours in PJM (clocks going forward in spring) | filled with the value **between the hour before and the hour after** |
| Missing data in the household (421 hours, some gaps of several days) | short gaps filled from the neighbouring hours, long gaps with the **usual demand for that weekday and hour** |
| 2 impossible one-hour jumps in PJM | replaced like a missing hour; real events (e.g. Hurricane Sandy 2012) were kept |
""")
    st.markdown("Calendar information was added for every hour: hour of day, weekday, month, weekend and public holiday "
                "(US holidays for PJM, French holidays for the household).")

    st.subheader("Training, validation and test periods")
    st.markdown("""
Each dataset is split **in time order** – the model learns from the past and is tested on the future, just like in real use:
- **Training (70%)** – the model learns from this data
- **Validation (15%)** – used to choose the model settings
- **Test (15%)** – used only at the end to measure accuracy; the model never sees it during training
""")
    # one row for each dataset and part, with the first date, the last date and the number of hours
    rows = []
    for name in ["pjme", "uci"]:
        for part, label in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
            d = load_split(name, part)
            # strftime("%d %b %Y") writes the date as text, e.g. 01 Jan 2015
            rows.append({"Dataset": LABEL[name], "Part": label, "From": d.index.min().strftime("%d %b %Y"),
                         "To": d.index.max().strftime("%d %b %Y"), "Hours": len(d)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
