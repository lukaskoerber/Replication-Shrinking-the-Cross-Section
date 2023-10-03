import pandas as pd
from datetime import datetime

def load_managed_portfolios(filename, daily=True, drop_perc=1, omit_prefixes=None, keeponly=None):
    
    if omit_prefixes is None:
        omit_prefixes = []
    if keeponly is None:
        keeponly = []

    # Decide on the date format based on whether data is daily or not
    if daily:
        date_format = '%m/%d/%Y'
    else:
        date_format = '%m/%Y'
    
    # Read DATA from CSV and parse dates
    DATA = pd.read_csv(filename, parse_dates=["date"], dayfirst=False, date_parser=lambda x: datetime.strptime(x, date_format))
    
    # If keeponly is specified, filter columns based on it
    if keeponly:
        columns_to_keep = ['date', 'rme'] + keeponly
        DATA = DATA[columns_to_keep]
    else:
        # drop columns with certain prefixes
        for prefix in omit_prefixes:
            DATA.drop(columns=[col for col in DATA.columns if col.startswith(prefix)], inplace=True)
    
    # Drop columns with more than drop_perc percentage of missing values
    thresh = len(DATA) * drop_perc
    DATA.dropna(axis=1, thresh=thresh, inplace=True)

    # Drop rows with missing values (after the first two columns)
    DATA.dropna(subset=DATA.columns[2:], inplace=True)
    assert len(DATA) > 0.75 * len(DATA), 'More than 25% of obs. need to be dropped!'

    # Extract required values
    dates = DATA['date']
    mkt = DATA['rme']
    re = DATA.iloc[:, 3:]
    names = re.columns.tolist()

    # Return the values
    return dates, re, mkt, names#, DATA

