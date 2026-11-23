# 7. Charts: what each one shows and how to read it

All charts are in `figures/`. A one-line description of each is also in `figures/figure_descriptions.csv`.

#### Q: General tips to read any chart in this project?
1. Read the title and the axis labels first: what is on x (often time or hour) and what is on y (demand or error).
2. Check the unit: MW for PJM, kW for UCI. Charts with "divided by the mean" have no unit, so both datasets fit.
3. For error charts: lower = better.
4. Top row / top panel is usually PJM, bottom is UCI.

## Notebook 01 and 02 (data)

#### 01_daily_mean_load.png - daily average demand
x = date, y = average demand of each day. Top PJM (2002-2018), bottom UCI (2006-2010).
How to read: PJM goes up every summer and a bit every winter, very regular. UCI is high in winter, low in August
(holiday), and very jumpy from day to day.

#### 02_example_spike.png - a wrong value
24 hours of PJM around one spike. One single point jumps far away from both neighbours and comes straight back.
This cannot happen in a whole region, so it is a data error and was replaced.

#### 02_uci_gap_fill.png - how the longest gap was filled
Household demand in August 2010. Blue = original with a gap of about 5 days, orange = after filling. The orange part
follows the usual daily shape (weekday x hour average from training).

#### 02_train_val_test_split.png - the three periods
Daily demand coloured by part: train (first 70%), validation (next 15%), test (last 15%). The colours come one after
the other in time, which shows the split is in time order and the test is always the future.

## Notebook 03 (temporal patterns, training data only)

#### 03_profiles.png - average by hour, weekday and month
Left: average demand for each hour (0-23), with separate lines for weekday, weekend and holiday.
Middle: average for each weekday (0 = Monday). Right: average for each month.
How to read: PJM peaks in the afternoon/evening on weekdays and in summer. UCI peaks in the evening, at weekends and in winter.

#### 03_heatmap_weekday_hour.png - heat map
Rows = weekday (0 = Monday), columns = hour of day, colour = average demand. Brighter = higher.
How to read: find the brightest area. PJM = weekday daytime. UCI = evenings, also weekend middays.

#### 03_mstl_pjme.png and 03_mstl_uci.png - decomposition
Five panels from top: original demand, trend, daily part (seasonal_24), weekly part (seasonal_168), residual.
How to read: compare the size of the daily part with the residual. For PJM the daily part is big and the residual
small (regular). For UCI the residual is big (noisy, hard to predict).

#### 03_seasonal_3_weeks.png - zoom into the patterns
The daily and weekly parts for the last 3 weeks of training. You can see the shape of one day (repeats 21 times) and
the weekly part lower at the weekend for PJM.

#### 03_acf_pacf.png - autocorrelation
x = lag in hours (1 to 200), y = correlation (-1 to 1). The blue band = values inside it are not significant.
How to read ACF (left): look at the bars at 24 and 168. High bars = demand is similar to yesterday and last week.
PJM bars stay very high (0.8-0.97), UCI bars are lower (about 0.4-0.7).
How to read PACF (right): only the first few lags and lags near 24 and 168 stick out, so these are the useful ones.

#### 03_periodogram.png - strength of cycles
x = cycle length in hours, y = strength. Both axes are on a log scale (each step is x10).
Red lines mark 12 h, 24 h, 1 week, 1 year. A high peak on a red line = that cycle is strong.

#### 03_peak_low_hours.png - when peaks and lows happen
Peak = top 10% hours, low = bottom 10%. Left: share of them in each hour of the day. Right: share in each month.
How to read: PJM peaks are summer afternoons, lows are spring/autumn nights. UCI peaks are winter evenings.

## Notebook 04 (tree models)

#### 04_test_mape.png - percentage error of trees and benchmark
Grouped bars: x = horizon (1, 24, 168), bars = naive, RF, XGB, y = MAPE in %.
How to read: trees are far below the naive bar. Bars grow with the horizon. UCI bars are high because night demand is
close to zero (see file 06).

#### 04_feature_importance_24h.png - which inputs matter
Horizontal bars, longer = more important, for RF and XGB at 24 hours.
How to read: the top bar is the input the model uses most. PJM: lag_24. UCI: hour of day and lag_168 / lag_336.

#### 04_forecast_week_24h.png - prediction vs actual for one week
Black = actual, coloured = predictions (naive, RF, XGB), first week of the test period, 24 hours ahead.
How to read: the closer a coloured line is to the black line, the better. PJM lines are close, UCI lines get the daily
shape but miss the sudden spikes.

## Notebook 05 (neural networks)

#### 05_test_rmse.png - error of all 7 models
Grouped bars: x = horizon, one bar per model, y = RMSE (MW left, kW right).
How to read: within each horizon group compare bar heights. PJM 1h: tree bars clearly lowest. PJM 24h and 168h: bars
almost the same height. UCI: XGBoost lowest everywhere. Naive is the tall bar.

#### 05_attention_weights.png - where the attention model looks
x = hours before the forecast (48 to 1), y = weight. Red dashed lines at 24 and 48 hours.
How to read: if the model focused on "same hour yesterday" there would be a tall bar at 24. The bars are almost equal
(about 0.021 each = 1/48), so the model does not focus on special hours.

#### 05_forecast_week_24h.png - all models for one week
Like 04_forecast_week_24h but with XGBoost and the 4 neural networks. Differences are mostly at the peaks.

## Notebook 08 (paper charts)

#### 08_paper_patterns.png (paper Fig. 3) - daily shape and autocorrelation
Top: average demand by hour for weekdays and weekends, divided by the mean (1.0 = average). So 1.3 = 30% above average.
Bottom: ACF from 1 to 200 hours, dotted lines at 24 and 168.
How to read: PJM has one smooth daily hill and high ACF. UCI has a morning and a bigger evening peak and low ACF.

#### 08_paper_rmse_horizon.png (paper Fig. 4) - error vs horizon
x = horizon (1, 24, 168), y = RMSE divided by the mean demand, one line per model, grey dashed = seasonal naive.
How to read: the slope shows how fast the error grows. PJM lines rise steeply and come together at 168 h (tie).
UCI lines start much higher but are flatter, XGBoost stays at the bottom.

#### 08_paper_error_by_hour.png (paper Fig. 5) - error for each hour of the day
x = hour of day, solid lines = RMSE of XGBoost and GRU (24 hours ahead), dashed = how much demand normally varies in that
hour (standard deviation in training). All lines divided by their own mean so they share one axis.
How to read: the lines go up and down together. Highest error = evening (18:00 PJM, 21:00 UCI), lowest = 04:00.
This shows errors are biggest where demand is most variable.

#### 08_paper_forecast_week.png (not used in the final paper)
Actual vs XGBoost and GRU for the highest-demand PJM week (summer 2018) and one UCI week. Both models stay below the
highest peaks, which matches "all models under-predict peaks".

## Charts in the paper that are not from the notebooks

#### Fig. 1 - framework
The five stages of the work from left to right: data collection -> cleaning -> pre-processing -> model training ->
evaluation, with the temporal pattern analysis box below, which uses the cleaned training data and helps explain the errors.

#### Fig. 2 - model structure
Top: same input window for all neural models. Middle: the different layers of each model. Bottom: the same output part
(join with target-hour calendar -> Dense 32 ReLU -> Dense 1 -> forecast). See file 04 for every layer name.
