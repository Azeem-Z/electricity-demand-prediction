# 9. Hard questions

#### Q: Your title says "deep learning", but trees often won. Is the project a failure?
No. The question was whether deep learning is better under fair conditions, and the honest answer is "only sometimes".
On regional data at 24 and 168 hours the GRU equals the trees and is better at peaks. Showing when deep learning helps
and when it does not is a useful result, and it agrees with other studies (e.g. Shwartz-Ziv and Armon, "Tabular data: Deep learning is not all you need", 2022).

#### Q: Is the comparison really fair?
We tried to make it fair:
- same split, same training years (last 4 years of PJM training), same calendar information for all models,
- same tuning budget (8 Optuna trials at 24 h) and same 3 seeds,
- test data used only once, at the end,
- DM test to check if differences are real.
One thing is different on purpose: trees get hand-made lag columns, neural networks get the raw window. This is how
each model type is normally used.

#### Q: Why only 8 tuning trials and 10 epochs? Maybe the neural networks were undertrained.
Possible, and we say this in the limitations. With 8 models x 2 datasets x 3 horizons x 3 seeds, a bigger budget was
not practical. The budget is the same for every model, so no model was favoured. More tuning and longer training are
in future work.

#### Q: Why did you tune only at 24 hours?
To keep the budget small. 24 hours is the middle horizon and the most common in practice (day-ahead). The same rule was
used for all models.

#### Q: Why no weather data? Temperature is the biggest driver of demand.
True. We wanted to study what can be done with demand history and calendar only, for both datasets in the same way. The
UCI data has no weather, so adding it to PJM only would break the comparison. It is our first limitation and first
future work item. It would especially help the summer peaks on PJM.

#### Q: Why only one household? It does not represent all houses.
Right, one house cannot represent all homes. It is a well-known public dataset and shows how hard single-house
forecasting is. Using many houses (e.g. smart meter data) is future work.

#### Q: The UCI test period has no winter. Is that a problem?
Yes, a little. The test runs from April to November 2010, so we cannot say how the models do in winter for the house.
This comes from the 70/15/15 time split and is listed as a limitation.

#### Q: Why is the attention model "simplified"?
The full TFT has many parts (variable selection, static inputs, quantile outputs). We kept the main ideas (LSTM encoder,
gates, multi-head attention) and used the same output part as the other neural models, so the comparison is fair and
training stays affordable.

#### Q: Why didn't the attention weights show anything clear?
They were almost equal (0.016-0.030 vs 0.021 uniform). Possible reasons: short training, small window (48 h), and the
target-hour calendar input already gives the model the time information. So attention weights are not always a good
explanation tool. The pattern analysis was more useful.

#### Q: Why do all models under-predict the peaks?
Peaks are rare (top 10%), and models trained with squared error learn to stay close to the average. Peaks also depend a
lot on weather (hot days), which the models do not see. On PJM the mean error in peak hours was -1800 to -2600 MW at 24 hours.

#### Q: Why does the household error grow so little with the horizon?
Because even the last hour does not tell much about the next one (ACF 0.72 at lag 1). So losing the recent hours at
longer horizons loses little information. The model mostly relies on the daily and weekly habits, which are the same at
all horizons.

#### Q: Why is the seasonal naive the same at all horizons?
It always uses the value one week earlier (lag_168), which is known at every horizon up to 168. So the prediction, and the
score, does not change.

#### Q: At the 1-hour horizon, is RF really better than XGBoost?
Yes, but only a little: 532 vs 549 MW, about 3%. The DM test says it is significant (p < 0.001), but in practice both are
very good (R2 = 0.993).

#### Q: What would you do differently?
Add weather data, use more tuning trials and epochs, train on all PJM years, use more households, and try the full TFT
and newer models (e.g. N-BEATS). Also give prediction intervals (a range, not only one number).

#### Q: What is your practical advice to a grid company?
Start with a tree model with horizon-aware lag features: cheap, fast and strong, especially for short horizons and
household data. For regional day-ahead or week-ahead forecasts a GRU is an equal option and can be better at peaks.
Always report errors for peak and holiday hours separately, not only one overall number.

#### Q: How can someone reproduce your results?
Follow the README: install Python 3.11 and `requirements.txt`, run notebooks 01 to 08 in order. All seeds are fixed, so
the results should be the same or very close (small differences can come from library versions).
