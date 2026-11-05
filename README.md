# Electricity Demand Prediction

Code for my MSc project "Deep Learning Based Electricity Demand Prediction with Temporal Pattern Analysis".

Two datasets are used:
- PJM Hourly Energy Consumption (regional, hourly, MW) - https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- UCI Individual Household Electric Power Consumption (one house, every minute, kW) - https://doi.org/10.24432/C58K54

Notebooks in this repository:

1. `notebooks/01_download_data.ipynb` - gets the data and checks it (start, end, missing hours, duplicates)
2. `notebooks/02_preprocessing.ipynb` - cleaning, hourly resampling, calendar features, train / validation / test split, scaling
3. `notebooks/03_temporal_pattern_analysis.ipynb` - daily, weekly and yearly patterns, STL decomposition, autocorrelation, periodogram, stationarity tests, peak hours
4. `notebooks/04_baselines.ipynb` - seasonal naive, Random Forest and XGBoost, tuned with Optuna, 1h / 24h / 168h ahead
5. `notebooks/05_deep_models.ipynb` - LSTM, GRU, CNN-LSTM and an attention model (simplified Temporal Fusion Transformer)


## Folders

```
data/raw/pjm/              PJM csv files
data/raw/uci_household/    UCI data (zip file, unzipped by notebook 01)
data/processed/            cleaned data, made by notebook 02 (not in the repo)
figures/                   all charts from the notebooks as png files
notebooks/                 the notebooks
results/                   scores, tuned settings, attention weights (notebooks 04 and 05)
results/predictions/       test predictions of every model, dataset and horizon
requirements.txt           python packages
```


## How to run on Windows (step by step)

### Step 1 - Install Python 3.11

1. Go to https://www.python.org/downloads/windows/ and download **Python 3.11** (Windows installer 64-bit).
2. Run the installer and **tick "Add python.exe to PATH"** at the bottom of the first screen, then click "Install Now".
3. Check it works: open **Command Prompt** (press the Windows key, type `cmd`, press Enter) and type:

```
py -3.11 --version
```

It should print `Python 3.11.x`.

### Step 2 - Get the code

Option A - with Git (https://git-scm.com/download/win):

```
git clone https://github.com/Azeem-Z/electricity-demand-prediction.git
```

Option B - without Git: on the GitHub page click the green **Code** button, then **Download ZIP**, and unzip it (right click, "Extract All").

### Step 3 - Open Command Prompt in the project folder

```
cd C:\path\to\electricity-demand-prediction
```

(Replace `C:\path\to` with where you saved the folder. Tip: in File Explorer open the folder, click the address bar, type `cmd` and press Enter.)

### Step 4 - Create a virtual environment

```
py -3.11 -m venv .venv
```

### Step 5 - Activate it

In Command Prompt:

```
.venv\Scripts\activate
```

In PowerShell use `.venv\Scripts\Activate.ps1` instead. If PowerShell says running scripts is disabled, run this once and try again:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

When it works you see `(.venv)` at the start of the line.

### Step 6 - Install the packages

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes (TensorFlow is a big download, about 500 MB).

### Step 7 - Start Jupyter

```
jupyter lab
```

Jupyter opens in the browser. Open the `notebooks` folder.

### Step 8 - Run the notebooks in order

Open each notebook and click **Run > Run All Cells**:

1. `01_download_data.ipynb` (about 1 min) - unzips the UCI data, checks both datasets
2. `02_preprocessing.ipynb` (about 1 min) - writes the cleaned data to `data/processed/`
3. `03_temporal_pattern_analysis.ipynb` (about 1 min) - pattern analysis
4. `04_baselines.ipynb` (about 30 min) - naive, Random Forest and XGBoost
5. `05_deep_models.ipynb` (about 3 hours on a normal laptop CPU) - LSTM, GRU, CNN-LSTM and attention model

The charts are shown in the notebooks and also saved in `figures/`. Notebooks 04 and 05 save their scores and
test predictions in `results/`.

You don't have to run 04 and 05 to see the results: their outputs are already saved in the notebooks and in
`results/`. If you run them again, the tree models give the same numbers; the neural networks can differ
very slightly because TensorFlow training is not fully repeatable on every computer.

Next time you only need Step 3, Step 5 and Step 7.


## Using VS Code instead of Jupyter Lab

1. Install VS Code and the **Python** and **Jupyter** extensions.
2. Do Steps 1 to 6 above.
3. Open the project folder in VS Code (File > Open Folder).
4. Open a notebook, click **Select Kernel** (top right) and pick the `.venv` environment.
5. Click **Run All**.


## About the data

The raw data is already in the repository, so no account is needed. The UCI file is stored as a zip
(the unzipped file is 127 MB, too big for GitHub) and notebook 01 unzips it.

If the data folder is empty, notebook 01 downloads everything again. The PJM data comes from Kaggle,
which may ask for a login: create a free Kaggle account, go to Settings > API > Create New Token, and save the
downloaded `kaggle.json` in `C:\Users\<your user name>\.kaggle\`.


## Problems

- `py is not recognized` - Python is not installed or "Add python.exe to PATH" was not ticked. Run the installer again and choose "Modify".
- `jupyter is not recognized` - the virtual environment is not active. Do Step 5 again.
- Wrong kernel in Jupyter / VS Code - run `python -m ipykernel install --user --name electricity-demand` with the environment active, then pick "electricity-demand" as kernel.
