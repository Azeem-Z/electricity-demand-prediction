# PJM East demand after the study data (Sep 2018 -> today).
# source: U.S. Energy Information Administration (EIA), Hourly Electric Grid Monitor (form EIA-930).
# free files, no login: https://www.eia.gov/electricity/gridmonitor/
# PJM East = the 11 Mid-Atlantic zones of PJM, so we add up these zones.
# a saved copy is in data/raw/eia/pjm_east_eia.csv. the app only downloads the newest days on top of it.
# to rebuild the saved copy from zero: python eia_data.py

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
SAVED = BASE / "data/raw/eia/pjm_east_eia.csv"
URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_SUBREGION_{}_{}.csv"
ZONES = ["AE", "BC", "DPL", "JC", "ME", "PE", "PEP", "PL", "PN", "PS", "RECO"]
# the first weeks of the EIA zone data do not match PJM East and then 71 days are missing,
# so we only use the data from this date on
START = "2018-09-01"


def half_year(t):
    return (t.year, "Jan_Jun" if t.month <= 6 else "Jul_Dec")


def download_half(year, half):
    # one file = all US regions for 6 months (about 25 MB). we keep only the PJM East zones
    r = requests.get(URL.format(year, half), timeout=180)
    r.raise_for_status()
    d = pd.read_csv(io.BytesIO(r.content), thousands=",",
                    usecols=["Balancing Authority", "Sub-Region", "Demand (MW)", "UTC Time at End of Hour"])
    d = d[(d["Balancing Authority"] == "PJM") & d["Sub-Region"].isin(ZONES)]
    d["utc"] = pd.to_datetime(d["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p")
    w = d.pivot_table(index="utc", columns="Sub-Region", values="Demand (MW)", aggfunc="mean")
    # only hours where all 11 zones have a value, otherwise the sum would be too small
    return w.sum(axis=1, min_count=len(ZONES)).dropna().rename("load")


def read_saved():
    s = pd.read_csv(SAVED, parse_dates=["utc"], index_col="utc")["load"]
    return s


def build_saved():
    # downloads every half year from Jul 2018 to now and saves the sum of the zones
    now = pd.Timestamp.now()
    parts = []
    for year in range(2018, now.year + 1):
        for half in ["Jan_Jun", "Jul_Dec"]:
            if (year, half) < (2018, "Jul_Dec") or (year, half) > half_year(now):
                continue
            print("download", year, half)
            parts.append(download_half(year, half))
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    SAVED.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(SAVED, index_label="utc")
    print("saved", len(s), "hours up to", s.index.max())


def update(s):
    # adds the newest days. only the half year files after the last saved hour are downloaded
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if s.index.max() > now - pd.Timedelta(hours=36):
        return s
    halves = sorted({half_year(s.index.max()), half_year(now)})
    new = pd.concat([download_half(y, h) for y, h in halves])
    s = pd.concat([s, new])
    return s[~s.index.duplicated(keep="last")].sort_index()


def fill_gaps(s):
    # same idea as for the household data in notebook 02:
    # up to 3 missing hours -> straight line, up to 2 days -> same hour one week before
    s = s.interpolate(limit=3, limit_area="inside")
    gap = s.isna()
    run = gap.groupby((gap != gap.shift()).cumsum()).transform("size")
    short = gap & (run <= 48)
    for _ in range(3):      # if last week is missing too, take the week before
        s[short & s.isna()] = s.shift(168)[short & s.isna()]
    return s


def to_local(s):
    # EIA uses UTC "end of hour". the PJM data of the study uses US Eastern local time (like 18:00 = peak).
    # after changing to local time the average daily shape of both matches (correlation 0.99),
    # with the same peak hour (19:00) and lowest hour (04:00)
    s = s.copy()
    s.index = s.index.tz_localize("UTC").tz_convert("US/Eastern").tz_localize(None)
    s = s.groupby(level=0).mean()       # clock change in autumn gives one hour twice -> mean, like notebook 02
    s = s.asfreq("h")
    s = s[START:]
    s = s[s.first_valid_index():]       # the EIA data really starts on 18 Sep 2018
    return fill_gaps(s)


def load_eia(try_update=True):
    # returns (hourly PJM East demand in local time, True if the newest days were downloaded)
    s = read_saved()
    updated = False
    if try_update:
        try:
            s = update(s)
            updated = True
        except Exception:
            pass    # no internet -> the saved copy is used
    return to_local(s), updated


if __name__ == "__main__":
    build_saved()
