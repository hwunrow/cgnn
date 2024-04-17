import pandas as pd
from codebook import BOROUGH_FIPS_MAP


def get_date_range(start, end):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return dates


def get_fips_list():
    fips_list = list(BOROUGH_FIPS_MAP.values())
    return fips_list
