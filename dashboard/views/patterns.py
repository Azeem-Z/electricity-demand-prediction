import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import dataset_picker, load_split, load_csv, UNIT, LABEL

name = dataset_picker()
unit = UNIT[name]
st.title(f"Temporal patterns – {LABEL[name]}")

st.markdown("""
**What is temporal pattern analysis?**
Electricity demand repeats itself over time: it goes up and down **every day**, changes between **weekdays and weekends**,
and changes with the **seasons**. These repeating shapes are called *temporal patterns*.

**Why is it in this project?**
Before training any model, we look at these patterns to understand the data. Later, they help to explain the results:
the hours where demand changes the most (for example the evening peak) are also the hours where the models make the biggest errors.
""")
st.caption("All charts on this page use the training data only.")

df = load_split(name, "train")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ---------------- daily pattern ----------------
st.subheader("1. Daily pattern – demand over the day")
day_type = np.where(df["holiday"] == 1, "holiday", np.where(df["weekend"] == 1, "weekend", "weekday"))
by_hour = df.groupby([df["hour"], day_type])["load"].mean().reset_index()
by_hour.columns = ["hour", "day type", "load"]
fig = px.line(by_hour, x="hour", y="load", color="day type", markers=True,
              labels={"load": f"average demand ({unit})", "hour": "hour of the day (0 = midnight)"})
st.plotly_chart(fig, use_container_width=True)
st.markdown("**How to read it:** each line is the average demand at each hour of the day, separately for weekdays, weekends and public holidays. "
            + ("**What it shows:** demand is lowest at night (around 3–4 am) and highest in the late afternoon and evening. "
               "Weekdays are clearly higher than weekends, and holidays look like weekends."
               if name == "pjme" else
               "**What it shows:** the house uses little electricity at night, has a small morning peak and a big evening peak "
               "(around 8–9 pm). Weekends are higher than weekdays, because people are at home."))

# ---------------- yearly pattern ----------------
st.subheader("2. Yearly pattern – demand by month")
by_month = df.groupby("month")["load"].mean().reset_index()
fig = px.bar(by_month, x="month", y="load", labels={"load": f"average demand ({unit})", "month": "month (1 = January)"})
st.plotly_chart(fig, use_container_width=True)
st.markdown("**How to read it:** each bar is the average demand in that month. "
            + ("**What it shows:** demand is highest in July and August (air conditioning in summer), with a second smaller peak in winter (heating). "
               "Spring and autumn are the lowest."
               if name == "pjme" else
               "**What it shows:** the opposite of the region – the house uses most electricity in winter and least in summer, "
               "lowest in August (probably the family is on holiday)."))

# ---------------- weekly pattern ----------------
st.subheader("3. Weekly pattern – weekday × hour")
heat = df.pivot_table(index="dayofweek", columns="hour", values="load")
heat.index = DAYS
fig = px.imshow(heat, aspect="auto", color_continuous_scale="Viridis",
                labels={"x": "hour of the day", "y": "", "color": unit})
st.plotly_chart(fig, use_container_width=True)
st.markdown("**How to read it:** each square is the average demand on that weekday at that hour. "
            "Bright (yellow) = high demand, dark (purple) = low demand. "
            + ("**What it shows:** the bright band in the afternoon/evening is much stronger Monday to Friday than at the weekend."
               if name == "pjme" else
               "**What it shows:** the brightest squares are in the evening and at weekends."))

# ---------------- key facts ----------------
st.subheader("4. The two datasets compared")
# copy of data/processed/pattern_summary.csv (notebook 03), kept in results/ so it is in git for the server
p = load_csv("results/pattern_summary.csv").set_index("dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Daily pattern strength – region / house", f"{p.loc['PJME', 'daily_strength']:.2f} / {p.loc['UCI', 'daily_strength']:.2f}",
          help="From 0 to 1: how strongly the demand repeats the same shape every day. Close to 1 = very regular.")
c2.metric("Weekly pattern strength – region / house", f"{p.loc['PJME', 'weekly_strength']:.2f} / {p.loc['UCI', 'weekly_strength']:.2f}",
          help="From 0 to 1: how strongly the demand repeats the same shape every week.")
c3.metric("Variability – region / house", f"{p.loc['PJME', 'cv']:.2f} / {p.loc['UCI', 'cv']:.2f}",
          help="How much the demand jumps around compared with its average (coefficient of variation). Higher = more irregular and harder to predict.")
st.markdown("""
**In short:** the region's demand is **very regular** – it follows the same daily and weekly shape again and again, so it is easier to predict.
A single household is **much more irregular** (about 4 times more variable), because it depends on what the people in the house are doing.
This is why the household errors on the Results page are much larger.
""")
