import pandas as pd
from datetime import datetime

def load_ff25(datapath, daily, t0=None, tN=None):
    # Set default values for t0 and tN if they are not provided
    if t0 is None:
        t0 = datetime.min
    if tN is None:
        tN = datetime.max

    # Decide on the file names based on whether data is daily or not
    if daily:
        ffact5 = 'F-F_Research_Data_Factors_daily.csv'
        ff25 = '25_Portfolios_5x5_Daily_average_value_weighted_returns_daily.csv'
    else:
        ffact5 = 'F-F_Research_Data_Factors.csv'
        ff25 = '25_Portfolios_5x5_average_value_weighted_returns_monthly.csv'

    # Read DATA from CSV
    DATA = pd.read_csv(datapath + ffact5, parse_dates=["Date"], dayfirst=False)
    
    # Filter rows based on date range
    DATA = DATA[(DATA['Date'] >= t0) & (DATA['Date'] <= tN)]
    
    # Read RET from CSV
    RET = pd.read_csv(datapath + ff25, parse_dates=["Date"], dayfirst=False)
    
    # Inner join of DATA and RET on 'Date' column
    DATA = pd.merge(DATA, RET, on='Date', how='inner')

    # Extract required columns and perform operations
    dates = DATA['Date']
    mkt = DATA['Mkt_RF'] / 100
    ret = DATA.iloc[:, 5:30].divide(100) - DATA['RF'] / 100
    labels = RET.columns[1:].tolist()

    return dates, ret, mkt, DATA, labels
