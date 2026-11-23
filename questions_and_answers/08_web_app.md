# 8. Web app (dashboard folder)

#### Q: What is the web app and which tool did you use?
A small website made with Streamlit (a Python library for data apps). Start it with `streamlit run app.py` inside the
`dashboard` folder. Charts are made with Plotly, so you can hover to see values.

#### Q: What pages does it have?
| Page | What it does |
|---|---|
| Predict demand | choose dataset, horizon (1/24/168 h), a start date in the test period and 1-14 days; shows predicted vs actual, error boxes and an hourly table |
| Results | tabs: model accuracy (RMSE / MAE / R2 / MAPE, best in green), errors by period, significance tests, attention weights |
| Temporal patterns | daily, yearly and weekly patterns and the two datasets compared (training data only) |
| About the project | aim, research questions, gaps, datasets, cleaning, split |
| Terms explained | glossary of the measures and terms |

#### Q: Which model does the app use for predictions? Why only one?
XGBoost. It is best or statistically tied with the best in 5 of 6 cases (DM test), it is fast and it does not need
TensorFlow, so the app is light. The saved models are in `models/xgb_<dataset>_h<horizon>.json` (notebook 07).
Notebook 07 checks that the saved model gives the same predictions as notebook 04 (seed 0).

#### Q: Why can the user only pick dates in the test period?
The model has never seen those dates in training, so the result is an honest test. If we allowed training dates, the
error would look too good.

#### Q: What do the three error boxes mean?
- Error in % (WAPE): Weighted Absolute Percentage Error = MAE / average demand of the chosen days x 100. We use WAPE
  and not MAPE here because it does not explode when demand is close to zero.
- Average error (MAE) and the large-error measure (RMSE) for the chosen days, in MW or kW.
Under the chart the app also shows the error over the whole test period, so the user can see if the chosen week was
easier or harder than normal.

#### Q: What does the error column in the table mean?
Predicted minus actual. Positive = prediction too high, negative = too low.

#### Q: The results page shows a different "best" model per horizon. Is the app wrong?
No. It shows the real results. Green only means "lowest number". The Significance tests tab shows if that difference is
real. Many green cells on PJM are "no real difference" in the DM test.

#### Q: Does the app retrain the model?
No. It loads the saved model, builds the same lag and calendar features as notebook 04 and predicts. This takes a
second.
