import pandas as pd
from datetime import datetime

def load_ff_anomalies(datapath, daily, t0=None, tN=None):
    # Setting default values for t0 and tN if they are not provided
    if t0 is None:
        t0 = datetime.min
    if tN is None:
        tN = datetime.max

    # Decide on the file names based on whether data is daily or not
    if daily:
        ffact5 = 'F-F_Research_Data_5_Factors_2x3_daily.csv'
        fmom = 'F-F_Momentum_Factor_daily.csv'
    else:
        ffact5 = 'F-F_Research_Data_5_Factors_2x3.csv'
        fmom = 'F-F_Momentum_Factor.csv'

    # Read DATA from CSV
    DATA = pd.read_csv(datapath + ffact5, parse_dates=["Date"], dayfirst=False)
    
    # Filter rows based on date range
    DATA = DATA[(DATA['Date'] >= t0) & (DATA['Date'] <= tN)]
    
    # Read MOM from CSV
    MOM = pd.read_csv(datapath + fmom, delimiter=",", parse_dates=["Date"], dayfirst=False)
    
    # Inner join of DATA and MOM on 'Date' column
    DATA = pd.merge(DATA, MOM, on='Date', how='inner')
    
    # Extract required columns and perform operations
    dates = DATA['Date']
    ret = DATA[['SMB', 'HML', 'Mom', 'RMW', 'CMA']] / 100
    mkt = DATA['Mkt_RF'] / 100
    
    # No de-market operation is done in the provided MATLAB function so omitting that

    return dates, ret, mkt, DATA

