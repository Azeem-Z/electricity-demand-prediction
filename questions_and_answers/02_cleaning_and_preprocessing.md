# 2. Cleaning and pre-processing (notebook 02)

## Cleaning

#### Q: What problems did the raw data have?
- PJM: 4 duplicate hours and 30 missing hours. They come from daylight saving time (clocks go back -> one hour
  appears twice, clocks go forward -> one hour is missing). There were also a few impossible one-hour spikes.
- UCI: about 26,000 missing minutes ("?" in the file). Some gaps are short, the longest is about 5 days (August 2010).

#### Q: How did you fix duplicate and missing hours in PJM?
- Duplicate hours: we took the mean of the two values (`groupby("Datetime").mean()`).
  Example: 01:00 appears twice with 30,000 and 30,200 -> we keep one 01:00 with 30,100.
- Missing hours: linear interpolation, i.e. a straight line between the hour before and the hour after.
  Example: 02:00 = 30,000, 03:00 missing, 04:00 = 31,000 -> 03:00 becomes 30,500.

#### Q: What is a spike and how did you find it?
A spike is one hour that jumps far away from both neighbours and comes back, which is impossible for a whole region.
For each hour we compute: difference = value - (hour before + hour after) / 2.
Then a robust z-score: z = (difference - median) / (1.4826 x MAD).
- MAD = Median Absolute Deviation, a measure of spread that is not pulled by the spikes themselves.
- 1.4826 makes the MAD comparable to a normal standard deviation.
If |z| > 10 the hour is a spike. It is set to empty and filled by interpolation.

#### Q: Why such a high threshold (10)? Normally people use 3.
Because we only want to remove clear data errors. With a threshold of 3.5 about 2.6% of PJM hours and 3% of
household hours would be marked, and most of those are real peaks and holiday dips that the models must learn.
Real events like Hurricane Sandy (October 2012) were kept on purpose.

#### Q: Did you remove spikes in the household data?
No. In one house short sudden peaks are normal (oven, kettle, washing machine). They are real demand.

#### Q: How did you fill the household gaps?
Two steps:
1. Short gaps: up to 3 missing hours are filled by linear interpolation (`interpolate(limit=3)`).
2. Longer gaps: the rest is filled with the average demand of the same weekday and hour, learned from the training
   part only. Example: a missing Tuesday 19:00 gets the average of all Tuesdays at 19:00 in the training data.

#### Q: Why did you change the household data from minutes to hours?
So both datasets have the same time step and the same horizons (1, 24, 168 hours) mean the same thing.
`resample("h").mean()` takes the 60 minute values of each hour and averages them.

## Features

#### Q: What features (inputs) did you create?
Calendar features from the time stamp (`add_features`):
| Feature | Meaning | Example for Saturday 25 Dec 2010, 19:00 |
|---|---|---|
| hour | hour of day 0-23 | 19 |
| dayofweek | 0 = Monday ... 6 = Sunday | 5 |
| month | 1-12 | 12 |
| dayofyear | 1-366 | 359 |
| weekend | 1 if Saturday or Sunday | 1 |
| holiday | 1 if public holiday (US for PJM, France for UCI) | 1 |
| hour_sin, hour_cos | hour as a point on a circle | see below |
| month_sin, month_cos | month as a point on a circle | see below |

#### Q: Why sin and cos of the hour?
With plain numbers, hour 23 and hour 0 look far apart (23 vs 0), but they are only 1 hour apart.
Putting the hour on a circle with sin(2 x pi x hour / 24) and cos(2 x pi x hour / 24) makes 23:00 and 00:00 close
again. The same is done for the month (December and January are neighbours).

#### Q: Why US holidays for PJM and French holidays for UCI?
PJM is in the USA and the UCI house is in France. On holidays people behave like on a Sunday, so demand changes.

#### Q: What are lag features? (tree models only)
A lag is the demand some hours earlier. lag_24 = the demand at the same hour yesterday.
Trees cannot read a sequence, so we give them these past values as separate columns (`make_features`).

#### Q: Why are the lags different for each horizon?
Because we can only use values that are already known when we make the forecast.
If we predict 24 hours ahead, the last 23 hours are still in the future, so the smallest lag we can use is 24.
| Horizon | Lags used |
|---|---|
| 1 hour | 1, 2, 3, 4, 5, 6, 24, 48, 168, 336 |
| 24 hours | 24 to 29, 48, 168, 336 |
| 168 hours | 168 to 173, 336 |

Plus two rolling means: the average of the last 24 hours and of the last 168 hours (both starting h hours back),
and the calendar columns. Using a lag smaller than h would be cheating (data leakage).

#### Q: What inputs do the neural networks get?
A window of the last L hours (L = 24, 48 or 72, chosen by tuning) with 8 values per hour:
load, hour_sin, hour_cos, month_sin, month_cos, dayofweek, weekend, holiday.
Plus the 7 calendar values of the target hour (the hour we predict), so the model knows e.g. "the target is a holiday".

## Split

#### Q: How did you split the data?
In time order, 70% / 15% / 15%:
| Part | Used for | PJM | UCI |
|---|---|---|---|
| Train (70%) | the model learns from it | Jan 2002 - Aug 2013 (101,774 h) | Dec 2006 - Sep 2009 (24,212 h) |
| Validation (15%) | choosing settings, early stopping | Aug 2013 - Feb 2016 (21,809 h) | Sep 2009 - Apr 2010 (5,188 h) |
| Test (15%) | final score, used only once at the end | Feb 2016 - Aug 2018 (21,809 h) | Apr 2010 - Nov 2010 (5,189 h) |

#### Q: Why not a random split?
In real life we always predict the future from the past. A random split would put hours from 2018 into training
and hours from 2017 into test, so the model would "see the future". That gives scores that are too good. A split
in time order is the honest way for time series.

#### Q: Did all models use the full PJM training data?
No. All models (trees and neural networks) use only the last 4 years of the PJM training period
(4 x 365 x 24 = 35,040 hours). This keeps training time the same for everybody and keeps the comparison fair.
The data is close in time to the test period, so it is also the most relevant part.

## Scaling

#### Q: Which scaler did you use and why?
`MinMaxScaler` from scikit-learn. It changes every column to the range 0 to 1:
scaled = (x - min) / (max - min).
Example: PJM min 14,500 MW and max 62,000 MW -> 38,250 MW becomes (38,250 - 14,500) / 47,500 = 0.5.
Neural networks train better and faster when all inputs are small numbers of similar size.

#### Q: Do the tree models also use scaled data?
No. A tree only asks questions like "is lag_24 > 30,000?", and the answer is the same whether the data is scaled
or not. So trees use the original MW / kW values.

#### Q: Why did you fit the scaler on the training part only?
`scaler.fit(train)` learns the min and max from the training data only. If we used the test data, the model would
get a small hint about the future (for example the highest value in 2018). This is called data leakage.
The same scaler is then used to change validation and test (`transform`).

#### Q: How do you get MW / kW back from the scaled prediction?
The opposite formula: x = scaled x (max - min) + min. In notebook 05: `pred * d["rng"] + d["lo"]`.
The scaler is saved with `joblib.dump` (to `data/processed/pjme_scaler.pkl`) and loaded back with `joblib.load`.
All errors are reported in MW / kW, never in scaled units.

#### Q: What is a parquet file?
A file format for tables (`to_parquet`, `read_parquet`). It is smaller and faster than CSV, and it keeps the date
index and the column types, so we do not have to parse dates again.

## Data leakage

#### Q: What did you do to avoid data leakage?
- Split in time order.
- Scaler min and max from training only.
- Gap-filling profile (weekday x hour) from training only.
- Spike threshold (median, MAD) from training only.
- Peak / low thresholds (top / bottom 10%) from training only.
- Lags never smaller than the horizon.
- The test set was used only for the final scores, never for tuning.
