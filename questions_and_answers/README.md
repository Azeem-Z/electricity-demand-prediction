# Questions and answers

Possible questions about this project (viva, presentation, supervisor meeting) with short answers in easy English.
All numbers come from the notebooks and the files in `results/`.

| File | What is inside |
|---|---|
| [01_project_and_data.md](01_project_and_data.md) | what the project is about, the two datasets, MW and kW |
| [02_cleaning_and_preprocessing.md](02_cleaning_and_preprocessing.md) | cleaning, missing values, spikes, features, lags, train / validation / test split, scaler, data leakage |
| [03_temporal_pattern_analysis.md](03_temporal_pattern_analysis.md) | MSTL, autocorrelation (ACF, PACF), periodogram, ADF and KPSS tests, log scale |
| [04_models.md](04_models.md) | seasonal naive, Random Forest, XGBoost, LSTM, GRU, CNN-LSTM, attention model, every layer name |
| [05_training_tuning_and_seeds.md](05_training_tuning_and_seeds.md) | how training was done, epochs, batch, early stopping, Optuna tuning, seeds, logs |
| [06_metrics_and_results.md](06_metrics_and_results.md) | RMSE, MAE, R2, MAPE, how to read the results table, Diebold-Mariano test, p-value |
| [07_charts.md](07_charts.md) | every chart in `figures/`, what it shows and how to read it |
| [08_web_app.md](08_web_app.md) | the Streamlit web app |
| [09_hard_questions.md](09_hard_questions.md) | difficult questions about the choices and the limits of the work |

## Short forms used in this folder

| Short form | Full term | Meaning in one line |
|---|---|---|
| PJM | Pennsylvania - New Jersey - Maryland Interconnection | US grid company; we use its East region (PJME) |
| UCI | University of California, Irvine (Machine Learning Repository) | where the household dataset comes from |
| MW | megawatt = 1,000,000 watts | unit of the regional demand |
| kW | kilowatt = 1,000 watts | unit of the household demand |
| h | horizon | how many hours ahead we predict (1, 24 or 168) |
| RF | Random Forest | many decision trees, average of their answers |
| XGB | XGBoost (Extreme Gradient Boosting) | trees built one after another, each fixes the last one's errors |
| RNN | Recurrent Neural Network | network that reads data step by step in time order |
| LSTM | Long Short-Term Memory | RNN with gates, can remember long patterns |
| GRU | Gated Recurrent Unit | simpler and faster version of LSTM |
| CNN | Convolutional Neural Network | network that slides small filters over the data |
| TFT | Temporal Fusion Transformer | attention-based forecasting model; we use a simplified version |
| MSTL | Multiple Seasonal-Trend decomposition using Loess | splits a series into trend + daily + weekly + rest |
| ACF | Autocorrelation Function | how similar the series is to itself some hours earlier |
| PACF | Partial Autocorrelation Function | the direct link to one earlier hour only |
| ADF | Augmented Dickey-Fuller test | test for stationarity |
| KPSS | Kwiatkowski-Phillips-Schmidt-Shin test | second test for stationarity |
| RMSE | Root Mean Squared Error | average error, big mistakes count more |
| MAE | Mean Absolute Error | plain average error |
| MAPE | Mean Absolute Percentage Error | average error in % |
| R2 | coefficient of determination | share of the ups and downs the model explains (1 = perfect) |
| DM test | Diebold-Mariano test | checks if two models are really different or just by luck |
| TPE | Tree-structured Parzen Estimator | the search method Optuna uses to pick settings |
| MAD | Median Absolute Deviation | robust measure of spread, used to find spikes |
| ReLU | Rectified Linear Unit | negative number -> 0, positive stays the same |
