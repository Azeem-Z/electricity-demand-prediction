# 1. Project and data

#### Q: What is your project about, in two sentences?
We predict how much electricity will be used 1 hour, 1 day (24 hours) and 1 week (168 hours) ahead. We compare 7 models
(simple benchmark, 2 tree models, 4 neural networks) on a big region and on one house, all under the same rules,
and we use the patterns in the data (daily, weekly, yearly) to explain where the models make mistakes.

#### Q: What are your research questions?
1. Are deep learning models better than tree models when all models get the same data and the same rules?
2. How does the result change between a region and a household, and between 1 hour, 1 day and 1 week ahead?
3. Which time patterns (peak hours, weekends, holidays, seasons) give bigger errors, and can attention weights explain them?

#### Q: Why is electricity demand prediction important?
The grid must produce the same amount of electricity that people use, at every moment. If the forecast is too low
there can be shortages, if it is too high fuel and money are wasted. Forecasts are used for planning power plants,
reserves and energy trading.

#### Q: Which datasets did you use?
| | PJM East (PJME) | UCI household |
|---|---|---|
| What | total demand of a large US region | one house near Paris, France |
| Unit | MW (megawatt) | kW (kilowatt) |
| Time step | every hour | every minute (we change it to hourly) |
| Period | 2002 to 2018 | Dec 2006 to Nov 2010 |
| Size | 145,366 hourly rows | 2,075,259 minute rows -> 34,589 hourly rows |
| Source | Kaggle (Hourly Energy Consumption) | UCI Machine Learning Repository |

#### Q: Why two datasets and not one?
To see if the best model is the same at two very different scales. A region is the sum of millions of homes, so
random ups and downs cancel out and the curve is smooth. One house is very jumpy (e.g. the kettle goes on and the
demand jumps for 3 minutes). Our results show the answer really is different for the two, so this was worth doing.

#### Q: What does MW and kW mean? Give an example.
Watt is the unit of power. 1 kW = 1,000 W, 1 MW = 1,000 kW.
Example: a kettle uses about 2 kW. The average PJM East demand is about 32,400 MW, which is like 16 million
kettles on at the same time. The average house demand in the UCI data is about 1.09 kW.

#### Q: The PJM folder has many files (AEP, COMED, DAYTON ...). Why only PJME?
Notebook 01 checks all of them (`data/processed/data_inventory.csv`). PJME is the longest and largest region
(2002 to 2018), so it gives the most data. Using one region keeps the study clear.

#### Q: What does "horizon" mean?
How far into the future we predict.
- h = 1: at 9:00 we predict 10:00.
- h = 24: at 9:00 today we predict 9:00 tomorrow.
- h = 168: at 9:00 today we predict 9:00 next week (168 = 24 x 7).

#### Q: What is the main finding?
There is no single winner.
- 1 hour ahead: the tree models (Random Forest, XGBoost) are clearly best on both datasets.
- Household: XGBoost is significantly best at every horizon.
- Region, 1 day and 1 week ahead: the best neural model (GRU) is as good as the trees, and it is better in the peak hours.
- Errors are biggest in the hours where demand changes most, mainly the evening peak.
