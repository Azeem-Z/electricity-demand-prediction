import streamlit as st

st.title("📖 Terms explained")
st.markdown("Short explanations of the terms used in this app, each with a small example.")

st.subheader("Small example used below")
st.markdown("""
Imagine the model predicted the demand of a house for 3 hours:

| Hour | Actual demand | Predicted | Error (predicted − actual) |
|---|---|---|---|
| 03:00 (night) | 0.1 kW | 0.3 kW | +0.2 kW |
| 18:00 | 2.0 kW | 1.8 kW | −0.2 kW |
| 20:00 | 3.0 kW | 2.8 kW | −0.2 kW |

A **positive** error means the prediction was **too high**, a **negative** error means it was **too low**.
""")

st.subheader("Error measures")

with st.expander("**MAE** – Mean Absolute Error", expanded=True):
    st.markdown("""
**What it is:** the average size of the error, ignoring if it was too high or too low. Same unit as the demand (MW or kW).

**Example:** errors 0.2, 0.2 and 0.2 kW → MAE = (0.2 + 0.2 + 0.2) / 3 = **0.2 kW**.
"On average the prediction was 0.2 kW away from the actual demand."

**Lower is better.**
""")

with st.expander("**RMSE** – Root Mean Squared Error", expanded=True):
    st.markdown("""
**What it is:** like MAE, but every error is squared first, so **big mistakes count more**. Same unit as the demand.

**How:** square the errors, take the average, then the square root.

**Example:** errors 1, 1 and 4 MW → MAE = 2 MW, but RMSE = √((1 + 1 + 16) / 3) = √6 = **2.45 MW**.
The one big error (4 MW) pushes RMSE up more than MAE.

**Lower is better.** This is the main measure used to compare the models.
""")

with st.expander("**R²** – R-squared", expanded=True):
    st.markdown("""
**What it is:** how much of the ups and downs in demand the model explains. Usually between 0 and 1.

- **1** = perfect prediction
- **0** = no better than always guessing the average demand
- **below 0** = worse than always guessing the average

**Example:** R² = 0.85 → the model explains **85 %** of the changes in demand.
For PJM 1 hour ahead XGBoost has R² = 0.99, the simple seasonal naive model on the household has R² = −0.26.

**Higher is better.**
""")

with st.expander("**MAPE** – Mean Absolute Percentage Error", expanded=True):
    st.markdown("""
**What it is:** the error of **each hour** as a percentage of the actual demand of that hour, then averaged.

**Example:**
- 03:00: 0.2 / 0.1 = **200 %**
- 18:00: 0.2 / 2.0 = **10 %**
- 20:00: 0.2 / 3.0 = **6.7 %**

MAPE = (200 + 10 + 6.7) / 3 = **72 %**

**Problem:** the night hour has almost no demand, so a small error of 0.2 kW becomes 200 % and makes the
whole value very large. That is why MAPE looks very bad for the household. **Lower is better.**
""")

with st.expander("**WAPE** – Weighted Absolute Percentage Error (the \"Error in %\" in this app)", expanded=True):
    st.markdown("""
**What it is:** the **total** error divided by the **total** demand. It is the same as MAE / average demand.

**Example:** total error = 0.2 + 0.2 + 0.2 = 0.6 kW, total demand = 0.1 + 2.0 + 3.0 = 5.1 kW
→ WAPE = 0.6 / 5.1 = **11.8 %**

Same predictions as the MAPE example, but the night hour no longer blows up the result.
That is why this app shows WAPE as the "Error in %". For PJM, WAPE and MAPE are almost the same
(e.g. 5.9 % and 5.8 % one day ahead), because regional demand is never close to zero.

**Lower is better.**
""")

st.subheader("Other terms")

with st.expander("**Horizon** – how far ahead"):
    st.markdown("""
How far in the future the model predicts. **1 hour ahead** = the model knows the demand up to one hour before.
**1 day ahead (24h)** = it only knows the demand up to 24 hours before. **1 week ahead (168h)** = up to one week before.
The further ahead, the harder it is.
""")

with st.expander("**Training, validation and test period**"):
    st.markdown("""
The data is split in time order. The model **learns** from the training period, its **settings are chosen** on the
validation period, and its accuracy is **measured** on the test period, which it never saw before.
""")

with st.expander("**Seasonal naive**"):
    st.markdown("""
The simplest possible prediction: "the demand will be the same as at the same hour one week ago".
It is used as a minimum benchmark – a useful model must be better than this.
""")

with st.expander("**Seed** – training run"):
    st.markdown("""
Training a model involves some randomness. Each model was trained **3 times** with a different random start (seed)
and the results were averaged, so one lucky or unlucky run does not decide the result.
""")

with st.expander("**Peak, normal and low hours**"):
    st.markdown("""
**Peak** = the 10 % of hours with the highest demand, **low** = the 10 % with the lowest demand,
**normal** = all the others. The limits come from the training data.
""")

with st.expander("**Diebold–Mariano test and p-value**"):
    st.markdown("""
A statistical test that checks if the difference between two models is **real** or could be **just luck**.

- **p-value below 0.05** → the difference is real (statistically significant)
- **p-value 0.05 or higher** → the two models are not really different

**Example:** PJM 1 day ahead, GRU vs XGBoost: p = 0.93 → no real difference, they are equally good.
""")

with st.expander("**Attention weight**"):
    st.markdown("""
The attention model gives each of the past hours it looks at a weight (how much it "pays attention" to that hour).
All weights add up to 1. If the model looks at 48 hours and treats them all the same, each weight is 1/48 ≈ 0.021.
""")

with st.expander("**MW and kW**"):
    st.markdown("""
Units of power. **kW** (kilowatt) is used for the household – a kettle uses about 2 kW.
**MW** (megawatt) = 1,000 kW, used for the whole PJM East region (around 30,000 MW, millions of homes).
""")
