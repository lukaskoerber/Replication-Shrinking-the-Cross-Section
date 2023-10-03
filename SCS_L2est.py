import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
try:
    from python_code.utils import demarket, regcov, l2est
    from python_code.cross_validate import cross_validate
except ModuleNotFoundError:
    from utils import demarket, regcov, l2est
    from cross_validate import cross_validate

def SCS_L2est(dates, re, market, freq, anomalies, parameters):
    """
    Computes the L2 shrinkage estimator of the SDF parameters
    based on the method in Kozak, Nagel, and Santosh (2019).
    
    Parameters:
    - dates (pd.Series): time series of dates
    - re (pd.DataFrame): matrix of excess returns time series
    - market (pd.Series): matrix of market's excess returns time series
    - freq (int): number of observations per year
    - anomalies (list): list of anomaly names
    - kwargs: other optional keyword arguments
        * gridsize (int): default=20
        * cvmethod (str): default='CV'
        * kfold (int): default=4
        * objective (str): default='C-S R^2'
        * plot_dof (bool): default=False
        * plot_coefpaths (bool): default=False
        * plot_objective (bool): default=False
        * ... more parameters...
    
    Returns:
    - estimates (dict): structure of estimates
    
    Note: Always cite the paper when using this function.
    """

    # Assign default values
    parameters = {
        'gridsize': 100,
        'method': 'CV',
        'objective': 'CSR2',
        'ignore_scale': False,
        'kfold': 3,
        'oos_test_date': dates.iloc[-1],
        'freq': freq,
        'rotate_PC': False,
        'demarket_conditionally': False,
        'demarket_unconditionally': True,
        'devol_conditionally': False,
        'devol_unconditionally': True,
        'plot_dof': True,
        'plot_coefpaths': True,
        'plot_objective': True,
        'line_width': 1.5,
        'font_size': 10,
        'L2_max_legends': 20,
        'L2_sort_loc': 'opt',
        'L1_log_scale': True,
        'L2_log_scale': True,
        'legend_loc': 'best',
        'results_export': True,
        'show_plot': False
    }

    # Parse config and assign default values
    # parameters = parse_config(parameters, default_config)

    # We usually maximize an objective (e.g., R^2), except for HJ-distance (GLS) and SSE
    if parameters["objective"] in ['GLS', 'SSE']:
        optfunc = min
    else:
        optfunc = max

    # User-friendly names for objectives to use in plots
    mapObj = {
        'CSR2': 'Cross-sectional $R^2$',
        'GLSR2': 'Cross-sectional GLS $R^2$',
        'GLS': 'Residual $SR^2$',
        'SRexpl': 'Explained SR',
        'SSE': 'SDF RMSE',
        'SR': 'Sharpe Ratio',
        'MVU': 'Mean-variance utility'
    }
    parameters["sObjective"] = mapObj[parameters["objective"]]

    # Initialize; compute means, cov, SVD decomposition
    # Testing sample start date
    tT0 = datetime.strptime(parameters["oos_test_date"].strftime("%Y-%m-%d"), "%Y-%m-%d")
    re.index = dates.values
    market.index = dates.values

    mkt0 = market.copy()

    # De-market all excess returns 
    if parameters['demarket_conditionally']:  # conditionally
        demarket_ma_window = 3 * parameters['freq']  # use past 3 years to estimate betas

        # Placeholder for custom function `demarkcond`
        r0 = demarketcond(re.iloc[idx_train, :], market.iloc[idx_train], demarket_ma_window)
        idx_train = idx_train[demarket_ma_window:]  # drop NaNs

    elif parameters['demarket_unconditionally']:  # unconditionally

        # Placeholder for custom function `demarket`
        r_train, b_train = demarket(re.loc[:tT0, :], market.loc[:tT0])
        r_test = demarket(re.loc[tT0:, :], market.loc[tT0:], b_train)  # use betas estimated in the training sample
        # check if r_test is a dataframe
        if isinstance(r_test, pd.DataFrame):
            r0 = pd.concat([r_train, r_test], axis=0)
        else:
            r0 = r_train.copy()

    else:
        r0 = re.copy()

    # De-vol all excess returns conditionally if requested
    if parameters['devol_conditionally']:
        devol_ma_window = 22  # use past 22 days to estimate volatilities

        # Placeholder for custom function `devolcond`
        r0, mkt0 = devolcond(r0, market, devol_ma_window)
        idx_train = idx_train[devol_ma_window:]  # drop NaNs

    elif parameters['devol_unconditionally']:  # de-vol unconditionally

        # Normalize so that all returns have the standard deviation of the VW market
        r0 = r0.divide(r0.std(axis=0), axis=1).multiply(market.std())

    # Construct dates, mkt, and returns for train and test sets
    mkt = mkt0.loc[:tT0]
    mkt_test = mkt0.loc[tT0:]

    r_train = r0.loc[:tT0, :]
    r_test = r0.loc[tT0:, :]

    # Length of the training sample
    T, n = r_train.shape
    parameters['T'] = T
    parameters['n'] = n

    # Rotate into PC space if requested and change file suffix
    if parameters.get('rotate_PC', False):
        # Use training sample to form eigenvectors
        _, _, Q = np.linalg.svd(np.cov(r_train, rowvar=False)) 
        r_train = np.dot(r_train, Q)
        r_test = np.dot(r_test, Q)
        anomalies = ['PC' + str(i) for i in range(1, n + 1)]

    # Compute first and second moments
    X = regcov(r_train)
    y = np.mean(r_train, axis=0)#.reshape(-1, 1)  # making y a column vector
    #X_test = regcov(r_test)  # TODO: test does not have data in it!!
    #y_test = np.mean(r_test, axis=0)#.reshape(-1, 1)  # making y_test a column vector

    # Maximum in-sample SR
    w = np.dot(np.dot(y.T, np.linalg.pinv(X)), y)
    maxSR2 = freq * w
    # Note: the line for maxSR2_test is commented out in the original MATLAB code. 

    # Precompute E-V decomposition
    U, D, Q = np.linalg.svd(X)
    X2 = np.dot(np.dot(Q, np.sqrt(np.diag(D))), Q.T)
    d = np.sum(np.diag(D), axis=1)

    # Pre-compute pseudo inverses
    tol = max(X.shape) * np.finfo(float).eps * np.linalg.norm(d, np.inf)
    r1 = np.sum(d > tol) + 1
    Q1 = Q[:, :r1]
    s = d[:r1]
    s2 = 1 / np.sqrt(s)
    s = 1 / s
    Q1 = Q1.T
    Xinv = np.dot(Q1 * s.reshape(1, -1), Q1.T)
    X2inv = np.dot(Q1 * s2.reshape(1, -1), Q1.T)

    # Options
    parameters['xlbl'] = 'Root Expected SR$^2$ (prior), $\\kappa$'
    parameters['Q'] = Q
    parameters['d'] = d
    parameters['Xinv'] = Xinv


    # Functions to map L2pen <-> kappa
    kappa2pen = lambda kappa, T, X, p: p['freq'] * np.trace(X) / T / (kappa ** 2)

    # Find left and right limits
    lr = np.arange(1, 22)  # equivalent of 1:21 in MATLAB
    lm = 1

    z = np.empty((n, len(lr)))
    z.fill(np.nan)

    for i in lr:
        params = parameters.copy()  # Make a copy of p to avoid modifying the original
        params['L2pen'] = kappa2pen(2 ** (i - lm), T, X, parameters)
        z[:, i - 1] = l2est(X, y, params)[0]

    # Coefficient stabilize condition
    mean_val = np.mean(np.abs((z[:, 1:] - z[:, :-1])) / (1 + np.abs(z[:, :-1])), axis=0) > 0.01
    x_rlim = np.nonzero(mean_val)[0]

    # Use the left and right points to define the support and create a finer grid on this support
    x = np.logspace(np.log10(2**x_rlim[-1]), np.log10(0.01), parameters['gridsize'])
    l = [kappa2pen(val, T, X, parameters) for val in x]
    lCV = [val / (1 - 1 / parameters['kfold']) for val in l]  
    nl = len(l)

    # Estimate the L2 model
    params = parameters.copy() # Make a copy of p to avoid modifying the original

    # Create placeholders for outputs
    phi = np.full((n, nl), np.nan)
    se = np.full_like(phi, np.nan)
    objL2 = np.full((nl, 4), np.nan)
    objL2_folds = np.full((nl, params['kfold']), np.nan)  # Assuming params['kfold'] is the number of folds
    MVE = [None] * nl

    for i in range(nl):
        print(i)
        # Estimate parameters at each grid point
        params['L2pen'] = l[i]
        # Note: You need to define the l2est function in Python or provide its MATLAB code for translation
        phi[:, i], _, se[:, i] = l2est(X, y, params, True)

        # Cross validate estimated parameters
        params['L2pen'] = lCV[i]
        # Note: You need to define the cross_validate function in Python or provide its MATLAB code for translation
        objL2[i, :], params, objL2_folds_ = cross_validate(l2est, dates.values, r_train, params)
        objL2_folds[i, :] = objL2_folds_[:, 1] # Python is 0-indexed

        # Store OOS MVE portfolios for each CV run
        MVE[i] = params['cv_MVE']

    cv_idx_test = params['cv_idx_test'] 

    # Effective degrees of freedom
    df = np.sum(d.reshape(50,1) / (d.reshape(50,1) + np.array(l).reshape(1,100)), axis=0)

    # Optimal L2 model
    # Note: You need to define the optfunc function in Python or provide its MATLAB code for translation
    objL2opt = optfunc(objL2[:, 1])
    if optfunc == max:
        iL2opt = objL2[:, 1].argmax()
    if optfunc == min:
        iL2opt = objL2[:, 1].argmin()
    bL2 = phi[:, iL2opt]
    parameters['bL2'] = bL2
    parameters['R2oos'] = objL2opt
    L2optKappa = x[iL2opt]

    # MVE portfolios for each fold at the optimal level of shrinkage [flatten into single time-series]
    MVEopt = MVE[iL2opt]

    # Return coefficients paths, degrees of freedom, and objective's value
    parameters['coeffsPaths'] = phi
    parameters['objL2_IS'] = objL2[:, 0]
    parameters['objL2_OOS'] = objL2[:, 1]
    z = np.concatenate([MVEopt[key] for key in MVEopt], axis=0)
    parameters['optimal_model_L2'] = {
        'coefficients': bL2,
        'objective': objL2opt,
        'kappa': L2optKappa,
        'SR': np.mean(z) / np.std(z) * np.sqrt(parameters['freq'])
    }
    estimates = parameters

    # df <-> kappa plot
    if parameters['plot_dof']:  # plot degrees of freedom
        plot_dof(df, x, parameters)

    # SDF 2nd moment constraint (L2) coefficients 
    if parameters['plot_coefpaths']:
        # plot coefficients
        plot_L2coefpaths(x, phi, iL2opt, anomalies, 'SDF Coefficient, $b$', parameters)
        # plot t-stats
        plot_L2coefpaths(x, phi/se, iL2opt, anomalies, 'SDF Coefficient $t$-statistic', parameters)

    # L2 Cross-Validation/BIC plot
    if parameters['plot_objective']:
        plot_L2cv(x, objL2, parameters)

    # output table with coefficient & tstats estimates
    table_L2coefs(phi[:, iL2opt], se[:, iL2opt], anomalies, parameters)

    return estimates

def plot_dof(df, x, p):
    """
    degrees of freedom <-> kappa plot
    
    Parameters:
    - df: Degrees of freedom data to be plotted on the y-axis.
    - x: Data to be plotted on the x-axis.
    - p: Dictionary containing various plot parameters.
    """
    
    # Open a new figure
    plt.figure()
    
    # Plot
    plt.plot(x, df, linewidth=p['line_width'])
    
    # Log-scale adjustments
    if p['L1_log_scale']:
        plt.yscale('log')
        plt.yticks([tick + 1e-12 for tick in plt.yticks()[0]])  # Adding a small constant
    
    if p['L2_log_scale']:
        plt.xscale('log')
        plt.xticks([tick + 1e-12 for tick in plt.xticks()[0]])  # Adding a small constant
    
    # Labels and grid
    plt.xlabel(p['xlbl'], fontsize=12, labelpad=10, fontweight='bold')
    plt.ylabel('Effective degrees of freedom', fontsize=12, labelpad=10, fontweight='bold')
    plt.grid(True)
    
    # Setting x-axis limits
    plt.xlim([min(x), max(x)])
    
    # Show plot
    if p['show_plot']:
        plt.show()

    if p['results_export']:
        plt.savefig('python_code/results_export/degrees_of_freedom.png', dpi=300, bbox_inches='tight')

def plot_L2coefpaths(x, phi, iL2opt, anomalies, ylbl, p):
    """
    L2 coefficients paths plot
    
    Parameters:
    - x: Data for the x-axis.
    - phi: Coefficient path data.
    - iL2opt: Optimal index for regularization.
    - anomalies: Names for legends.
    - ylbl: Label for the y-axis.
    - p: Dictionary containing various plot parameters.
    """
    
    # Decide sorting location
    if p['L2_sort_loc'] == 'opt':
        iSortLoc = iL2opt
    elif p['L2_sort_loc'] == 'OLS':
        iSortLoc = 0
    else:
        raise ValueError('Unknown option')
    
    # Sorting mechanism
    if p['n'] > p['L2_max_legends']:
        I = np.argsort(-np.abs(phi[:, iSortLoc]))  # Descending sort by absolute value
    else:
        I = np.argsort(-phi[:, iSortLoc])  # Descending sort
    
    # Open a new figure
    plt.figure()
    
    # Plot
    for i in I:
        plt.plot(x, phi[i, :], linewidth=p['line_width'])
    
    # Log-scale adjustment
    if p['L2_log_scale']:
        plt.xscale('log')
        plt.xticks([tick + 1e-16 for tick in plt.xticks()[0]])
    
    # Labels and grid
    plt.xlabel(p['xlbl'], fontsize=12, labelpad=10, fontweight='bold')
    plt.ylabel(ylbl, fontsize=12, labelpad=10, fontweight='bold')
    plt.grid(True)
    
    # Legend
    idx = I[:min(p['L2_max_legends'], len(I))]
    plt.legend([anomalies[i] for i in idx], loc=p['legend_loc'], fontsize=p['font_size'], bbox_to_anchor=(1.05, 1))
    
    # Dashed line at optimal regularization
    plt.plot([x[iL2opt], x[iL2opt]], [np.min(phi), np.max(phi)], '--k')
    
    # x-axis limits
    plt.xlim([min(x), max(x)])
    
    # Show plot
    if p['show_plot']:
        plt.show()

    if p['results_export']:
        plt.savefig('python_code/results_export/coefficients_paths.png', dpi=300, bbox_inches='tight')


def plot_L2cv(x, objL2, p):
    """
    Plot SSE/objective & BIC as a function of degrees of freedom.
    
    Parameters:
    - x: Data for the x-axis.
    - objL2: Data for plotting objectives and possible other values.
    - p: Dictionary containing various plot parameters.
    """
    
    # Open a new figure
    plt.figure()
    
    # Plot In-sample (IS) and Out-of-Sample (OOS)
    plt.plot(x, objL2[:, 0], '--', linewidth=p['line_width'])  # IS
    plt.plot(x, objL2[:, 1], '-', linewidth=p['line_width'])  # OOS
    
    # Log-scale adjustment
    if p['L2_log_scale']:
        plt.xscale('log')
        plt.xticks([tick + 1e-16 for tick in plt.xticks()[0]])
    
    # Labels
    plt.xlabel(p['xlbl'], fontsize=12, labelpad=10, fontweight='bold')
    plt.ylabel(f"IS/OOS {p['sObjective']}", fontsize=12, labelpad=10, fontweight='bold')
    
    # Legends and plot +1, -1 standard error
    co = plt.gca().lines[-1].get_color()  # Getting color of last line plotted (OOS line)
    plt.plot(x, objL2[:, 1] + objL2[:, 3], ':', color=co, linewidth=1)  # +1 SE
    plt.plot(x, objL2[:, 1] - objL2[:, 3], ':', color=co, linewidth=1)  # -1 SE
    
    plt.legend(['In-sample', f"OOS {p['method']}", f"OOS {p['method']} +/- 1 s.e."],
               loc='upper right')
    
    # Grid, axis limits
    plt.grid(True)
    plt.ylim([0, max(0.1, min(10, 2*max(objL2[:, 1])))])
    plt.xlim([min(x), 2])
    
    # Show plot
    if p['show_plot']:
        plt.show()

    if p['results_export']:
        plt.savefig('python_code/results_export/cross_validation.png', dpi=300, bbox_inches='tight')



def table_L2coefs(phi, se, anomalies, p):
    """
    Function to display a table of largest coefficients and t-stats.

    Parameters:
    - phi: Coefficients.
    - se: Standard error.
    - anomalies: Anomaly descriptions.
    - p: Dictionary containing various parameters.
    """
    nrows = 10  # number of rows in the table to show
    
    # t-stats
    tstats = phi / se
    
    # by absolute tstats
    idx = np.argsort(np.abs(tstats))[::-1]
    
    # show only nrows items
    idx = idx[:nrows]
   
    # create a DataFrame
    data = {
        'Portfolio': [anomalies[i] for i in idx],
        'b': phi[idx],
        't_stat': np.abs(tstats[idx])
    }
    df = pd.DataFrame(data)
    
    # display table
    print(df)

    # export as a latex formatted table
    if p['results_export']:
        df.to_latex('python_code/results_export/coefficients_table.tex', index=False)
