import pandas as pd
import pathlib

BOROUGH_FIPS_MAP = {
    "BX": 36005,
    "BK": 36047,
    "MN": 36061,
    "QN": 36081,
    "SI": 36085,
}


def getDateRange(start, end):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return dates


def writeListToCSV(filepath, src_list):
    pathlib.Path(filepath).parent.mkdir(exist_ok=True)
    save_df = pd.DataFrame(src_list)
    save_df.to_csv(filepath, header=False, index=False)
