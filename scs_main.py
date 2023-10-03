import numpy as np
import pandas as pd
from datetime import datetime
import os
from load_managed_portfolios import load_managed_portfolios
from SCS_L2est import SCS_L2est

# Options
daily = True
interactions = False
rotate_PC = False
withhold_test_sample = False
dataprovider = 'anom'

# Sample dates
t0 = datetime.strptime('01-Jul-1963', '%d-%b-%Y')
tN = datetime.strptime('31-Dec-2017', '%d-%b-%Y')
oos_test_date = datetime.strptime('01JAN2005', '%d%b%Y')

# Current run folder
run_folder = datetime.today().strftime('%d%b%Y').upper() + "/"

# Paths
projpath = ''
datapath = os.path.join(projpath, 'Data')
instrpath = os.path.join(datapath, 'instruments')

# Initialize
if daily:
    freq = 252
    suffix = '_d'
    date_fmt = '%m/%d/%Y'
else:
    freq = 12
    suffix = ''
    date_fmt = '%m/%Y'

# Set random seed (equivalent of MATLAB's rng default)
np.random.seed(0)

# Default estimation parameters
default_params = {
    'gridsize': 100,
    'contour_levelstep': 0.01,
    'objective': 'CSR2',
    'rotate_PC': False,
    'devol_unconditionally': False,
    'kfold': 3,
    'plot_dof': True,
    'plot_coefpaths': True,
    'plot_objective': True,
    'fig_options': {'fig_sizes': ['width=half'], 'close_after_print': True}
}

# Load FF factors (assuming function has been translated)
# dd, re, _ = load_ff_anomalies(datapath, daily, t0, tN)

# Parameters setup
p = default_params

if interactions:
    p['kfold'] = 2
else:
    p['gridsize'] = 100

if withhold_test_sample:
    p['oos_test_date'] = oos_test_date

# Process original ff25 portfolios if requested
if dataprovider == 'ff25':
    if not interactions:
        pass
        # Assuming functions have been translated
        # dd, re, mkt, DATA, labels = load_ff25(datapath, daily, 0, tN)
        # Followed by processing and estimation logic as in MATLAB
else:
    # Managed portfolios
    fmask = os.path.join(instrpath, f"managed_portfolios_{dataprovider}{suffix}_*.csv")
    # Instead of ls in MATLAB, we can list files in directory using os
    flist = [f for f in os.listdir(instrpath) if f.startswith(f"managed_portfolios_{dataprovider}{suffix}")]
    filename = os.path.join(instrpath, flist[0].strip())
    # Followed by data loading and estimation as in MATLAB

    p['L1_truncPath'] = True

    if interactions:  # use interactions
        dd, re, mkt, anomalies = load_managed_portfolios(filename, daily, 0.2, [])
        p = SCS_L2est(dd, re, mkt, freq, anomalies, p)
    else:  # use only raw characteristics (no derived instruments)
        # load data
        dd, re, mkt, anomalies = load_managed_portfolios(filename, daily, 0.2, ['rX_', 'r2_', 'r3_'])
        
        # estimate
        p = SCS_L2est(dd, re, mkt, freq, anomalies, p)

    