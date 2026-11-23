# 3. Temporal pattern analysis (notebook 03)

#### Q: What is temporal pattern analysis and why did you do it?
"Temporal" means "about time". We look for patterns that repeat in time: every day, every week, every year.
We did it on the training data only, before any model, for two reasons:
1. To understand the two datasets (how regular they are, which earlier hours matter).
2. To explain later where and why the models make bigger errors.

#### Q: What is MSTL decomposition?
MSTL = Multiple Seasonal-Trend decomposition using Loess. It splits the demand into parts that add up to the original:
demand = trend + daily pattern (24 h) + weekly pattern (168 h) + residual (the rest).
- Trend: slow long-term change.
- Daily part: the shape of a normal day (low at night, high in the evening).
- Weekly part: weekdays vs weekend.
- Residual: what the patterns cannot explain (noise, weather, special events).
"Loess" is a smoothing method that fits small local curves. "Multiple" means it handles more than one season (24 and 168).

#### Q: What is "strength" of a pattern?
strength = 1 - variance(residual) / variance(pattern + residual). It is between 0 and 1.
Close to 1 = the pattern is strong and clear. Close to 0 = it is lost in the noise.
| | Daily strength | Weekly strength |
|---|---|---|
| PJM | 0.87 | 0.55 |
| UCI | 0.53 | 0.30 |
So the region repeats much more regularly than the house.

#### Q: What is autocorrelation (ACF)?
How similar the demand is to the demand k hours earlier. 1 = exactly the same shape, 0 = no link.
| Lag | PJM | UCI |
|---|---|---|
| 1 hour | 0.97 | 0.72 |
| 24 hours (yesterday) | 0.89 | 0.43 |
| 168 hours (last week) | 0.80 | 0.45 |
Example: PJM 0.97 at lag 1 means the next hour is almost the same as this hour. That is why 1-hour forecasts are easy
for PJM and why the error grows so much when the last hours are not available (24 and 168 hours ahead).

#### Q: What is PACF?
Partial autocorrelation. The direct link to the hour k hours back, after removing the effect of the hours in between.
Example: lag 2 looks strong in ACF only because lag 1 is strong. PACF removes that, so it shows which lags really add new
information.

#### Q: What is a periodogram?
It shows how strong each repeating cycle is. High point = strong cycle. We marked 12 h, 24 h, 1 week and 1 year.
- PJM: strongest are 24 hours and half a year (summer and winter peaks), then 12 hours and 1 week.
- UCI: strongest are 12 and 24 hours (morning and evening peaks) and the yearly cycle.

#### Q: Did you use any logs (logarithms)?
Yes, but only for display and for searching settings, never on the demand itself:
- The periodogram plot uses log-log axes (`loglog`), because the cycles go from 2 hours to 1 year and the strength
  values differ by thousands of times. On a normal axis everything except the biggest bar would be flat.
  On a log axis each step is x10 (1, 10, 100, 1000), so small and big values both fit.
- In tuning, the learning rate is searched on a log scale (`log=True`), so 0.0001 to 0.001 gets as many tries as
  0.001 to 0.01. See file 05.
- We did not take the log of the demand. The models predict MW / kW directly (neural networks on the 0-1 scaled values).

#### Q: What is stationarity and why test it?
A series is stationary if its average and spread do not change over time. Many classic methods need it.
We used two tests on the daily averages:
- ADF (Augmented Dickey-Fuller): small p-value (< 0.05) = stationary.
- KPSS: small p-value = NOT stationary (the opposite question).
Using both is safer. Result: PJM is stationary, the household is borderline.

#### Q: How did you define peak and low hours?
Peak = the 10% of training hours with the highest demand. Low = the 10% with the lowest.
These limits are learned from training data and then used to label the test hours for the error analysis.

#### Q: What did the pattern analysis find?
- PJM: smooth, strong daily and weekly cycles, highest in July and August (air conditioning), peaks 16:00-20:00 in summer,
  weekdays higher than weekends.
- UCI: noisy (coefficient of variation 0.85 vs 0.20 for PJM), highest in winter, peaks 19:00-21:00,
  weekends higher than weekdays (people are at home).

#### Q: What is the coefficient of variation (CV)?
CV = standard deviation / mean. It shows how much the data moves around compared to its average.
PJM 0.20 = moves about 20% around the mean. UCI 0.85 = moves about 85%, so much more jumpy.

#### Q: How did the pattern analysis link to the errors?
The hours with the most variable demand (largest standard deviation in training) are the hours with the largest errors.
The Spearman correlation between "error per hour of day" and "spread of demand per hour of day" was 0.91 to 0.98 at
24 and 168 hours. So the patterns explain the errors. Spearman correlation = correlation of the ranks (1 = same order).
