# 6. Metrics and results (notebooks 04, 05, 06)

## Error measures

Example used below: actual demand [100, 200, 300], prediction [110, 190, 330] -> errors [10, -10, 30].

#### Q: What is MAE?
Mean Absolute Error = average size of the errors, ignoring the sign.
(10 + 10 + 30) / 3 = 16.7. "On average the forecast is 16.7 MW away." Easy to explain.

#### Q: What is RMSE?
Root Mean Squared Error = square the errors, average, take the square root.
sqrt((100 + 100 + 900) / 3) = sqrt(366.7) = 19.1.
RMSE is always >= MAE. The bigger the gap, the more there are a few big mistakes. Big mistakes matter a lot for the grid
(e.g. missing a peak), which is why RMSE is the main measure.

#### Q: What is R2?
Coefficient of determination. How much of the ups and downs of the demand the model explains.
- 1 = perfect.
- 0 = no better than always predicting the average.
- below 0 = worse than the average (the seasonal naive on UCI has -0.26).
Example: PJM 1 hour ahead R2 = 0.993 -> the model explains 99.3% of the variation.

#### Q: What is MAPE and why is it not in the paper?
Mean Absolute Percentage Error = average of |error| / actual x 100.
It is in the notebooks and the web app, but not in the paper, because the household demand at night is close to zero
(minimum 0.12 kW). Dividing by such a small number makes MAPE huge (44% to 76%) and misleading.
Example: actual 0.2 kW, prediction 0.4 kW -> only 0.2 kW wrong but 100% error.

#### Q: Why can't you compare RMSE of PJM and UCI directly?
They are in different units (MW vs kW) and very different sizes. To compare we divide RMSE by the mean demand
(normalised RMSE). XGBoost: PJM 0.018 / 0.081 / 0.122 and UCI 0.475 / 0.583 / 0.600 at 1 / 24 / 168 hours.

## Results table (mean over 3 seeds, RMSE)

| Model | PJM 1h | PJM 24h | PJM 168h | UCI 1h | UCI 24h | UCI 168h |
|---|---|---|---|---|---|---|
| Seasonal naive | 4733 | 4733 | 4733 | 0.791 | 0.791 | 0.791 |
| Random Forest | **532** | **2488** | 3817 | 0.462 | 0.574 | 0.589 |
| XGBoost | 549 | 2512 | 3780 | **0.457** | **0.562** | **0.578** |
| LSTM | 1125 | 2581 | 3747 | 0.491 | 0.593 | 0.599 |
| GRU | 829 | 2509 | 3748 | 0.493 | 0.591 | 0.598 |
| CNN-LSTM | 1066 | 2606 | **3740** | 0.508 | 0.601 | 0.599 |
| Attention | 1143 | 2671 | 3802 | 0.514 | 0.607 | 0.613 |

PJM in MW, UCI in kW. Bold = lowest value.

#### Q: How do I read this table?
1. Pick a column (one dataset and one horizon).
2. Lower number = better.
3. First check every model beats the seasonal naive. Yes, all do.
4. Then look at the gaps. PJM 1h: trees about 530-550, neural 830-1140 -> clear win for trees.
   PJM 168h: everything between 3740 and 3817 -> about 2% apart, basically a tie.
5. Then check the DM test (below) before saying "best": a lower number is not always a real difference.

#### Q: Why does the error grow with the horizon?
The further ahead, the less we know. At 1 hour the last hour (correlation 0.97 for PJM) is known. At 24 hours we only
know yesterday, at 168 hours only last week. PJM error grows about 7 times from 1 to 168 hours, UCI only 1.25 times,
because for the house even the last hour does not tell much.

#### Q: Why are the neural networks bad at 1 hour?
The trees get the last 6 hours as separate columns (lag_1 ... lag_6) and can use lag_1 directly in a split. The neural
networks must learn this from the 0-1 scaled window with a short training budget (max 10 epochs). At longer horizons
the trees lose those lags while the neural networks still see the whole window, so they catch up on PJM.

#### Q: The best model is different for each column. How do you explain that?
That is the main finding, not a problem. The best model depends on the scale (region vs house) and the horizon.
The DM test shows that in many PJM cases the "winner" is not really better. XGBoost is best or statistically tied in
5 of 6 cases, so it is a safe default. That is why the web app uses XGBoost.

## Diebold-Mariano test

#### Q: What is the Diebold-Mariano (DM) test?
A statistical test that checks if two models really have different accuracy on the test data, or if the difference
could just be luck. It compares the squared errors of the two models hour by hour.

#### Q: How do I read the DM table?
`results/dm_tests.csv` has: dataset, horizon, model_a, model_b, dm_stat, p_value, better, significant.
- dm_stat negative -> model_a is better. Positive -> model_b is better.
- p_value < 0.05 -> the difference is real (significant at 5%). Otherwise "no real difference".
Examples:
| Comparison | dm_stat | p-value | Meaning |
|---|---|---|---|
| PJM 1h, RF vs XGB | -6.88 | < 0.001 | RF really better |
| PJM 24h, RF vs XGB | -1.23 | 0.22 | no real difference |
| PJM 24h, GRU vs XGB | -0.08 | 0.93 | no real difference |
| PJM 168h, all vs XGB | | > 0.05 | nobody is really different |
| UCI all horizons, all vs XGB | positive | < 0.01 | XGBoost really better than all |

#### Q: What is a p-value?
The chance of seeing a difference this big if the two models were in truth equally good.
p = 0.22 -> 22% chance, so it could easily be luck. p = 0.001 -> 0.1% chance, so it is very likely real.

#### Q: What is the HLN correction?
Harvey, Leybourne and Newbold correction. It makes the DM test more reliable for long multi-step forecasts.
We also correct for the fact that errors of neighbouring hours are linked (we use h-1 lags in the variance).
At h = 1 there is no such correction, which is listed as a limitation.

## Error by period

#### Q: How did you break the errors down?
Each test hour gets labels: hour of day, day type (weekday / weekend / holiday), demand level (peak / normal / low) and
season. Then RMSE is computed per group (`results/error_by_period.csv`).

#### Q: What did you find? (24 hours ahead)
| Period | PJM XGB | PJM GRU | UCI XGB | UCI GRU |
|---|---|---|---|---|
| Weekday | 2447 | 2433 | 0.515 | 0.544 |
| Holiday | 2618 | 2982 | 0.679 | 0.670 |
| Peak (top 10%) | 4221 | 3855 | 1.480 | 1.510 |
| Normal | 2361 | 2432 | 0.480 | 0.510 |
- Peak hours are the hardest: 1.5-1.8 times the normal error on PJM, about 3 times on UCI. All models predict peaks too low.
- On PJM the GRU is better in peak hours (3855 vs 4221), XGBoost is better in normal hours, so the totals end up equal.
- Holidays are a weakness of the neural networks on PJM (2982 vs 2618). Holidays are only about 3% of training hours, so
  there are few examples to learn from. A tree can use the holiday flag in one split.
- Summer is worst on PJM (air conditioning peaks), best on UCI (low, steady demand). The UCI test period has no winter.

#### Q: What did the attention weights show?
With a 48-hour window, equal weights would be 1/48 = 0.021 each. The mean weights were between 0.016 and 0.030, so
almost equal. The model did not clearly focus on special hours like "same hour yesterday". The temporal pattern analysis
explained the errors better than the attention weights.
