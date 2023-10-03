from datetime import datetime
import numpy as np
from utils import regcov

def cross_validate(FUN, dates, r, params):
    """
    Compute IS/OOS values of the objective function based on the FUN function.
    Implements multiple objectives and validation methods.

    Parameters:
    - FUN: Handle to a function which estimates model parameters.
    - dates: (T x 1) array of dates.
    - r: (T x N) matrix of returns.
    - params: Dictionary that contains extra arguments.

    Returns:
    - obj: (1 x 2) IS and OOS values of the estimated objective function.
    - params: Returns back the params dictionary.
    - obj_folds: ...
    """
    if not callable(FUN):
        raise ValueError('Provided FUN argument is not a callable function.')

    # Select requested method
    if 'method' not in params:
        cross_validate_handler = cross_validate_cv_handler
    else:
        map_cv_method = {
            'CV': cross_validate_cv_handler,
            'ssplit': cross_validate_ssplit_handler,
            # 'bootstrap': cross_validate_bootstrap_handler
        }
        cross_validate_handler = map_cv_method.get(params['method'])

    # Execute selected method
    params['dd'] = dates
    params['ret'] = r
    params['fun'] = FUN
    obj, params, obj_folds = cross_validate_handler(params)

    return obj, params, obj_folds

def cross_validate_ssplit_handler(params):
    """
    Sample split handler for cross-validation.

    Parameters:
    - params: Dictionary with parameters, including 'splitdate', 'dd', etc.

    Returns:
    - obj, params: Results from the bootstrp_handler.
    """
    # Get split date or default to '01JAN2000'
    sd = params.get('splitdate', '01JAN2000')

    # Convert string date to datetime object
    tT0 = datetime.strptime(sd, '%d%b%Y')
    idx_test = [i for i, d in enumerate(params['dd']) if d >= tT0]

    return bootstrp_handler(idx_test, params)


def cross_validate_cv_handler(params):
    """
    Perform k-fold cross-validation.
    
    Parameters:
    - params: dictionary containing the parameters
    
    Returns:
    - obj: (k x 2) array of IS and OOS values of the estimated objective function for each fold
    - params: updated params dictionary
    - obj_folds: (k x 2) array, equal to obj
    
    Note: Requires custom function `bootstrp_handler` and `cvpartition_contiguous`.
    """
    
    # Set k (number of folds) either to provided value or default to 2
    k = params.get('kfold', 2)
    
    cv = cvpartition_contiguous(np.size(params['ret'],0), k)
    
    # Initialize obj to hold IS/OOS stats for each partition
    obj = np.nan * np.zeros((k, 2))
    
    for i in range(k):
        
        idx_test = cv[i]
        if 'cv_idx_test' not in params:
            params['cv_idx_test'] = {}
        params['cv_idx_test'][i] = idx_test
        params['cv_iteration'] = i
        obj[i, :], params = bootstrp_handler(idx_test, params)
        
    # Store estimates for each fold
    obj_folds = obj
    
    # Compute average and standard error of IS/OOS stats across folds
    obj = np.hstack([np.mean(obj, axis=0), np.std(obj, axis=0) / np.sqrt(k)])
    
    # Uncomment and modify the following code if 'SRexpl' objective function is used
    # if params['objective'] == 'SRexpl':
    #     obj = np.sqrt(np.maximum(0, obj))
    
    return obj, params, obj_folds

def bootstrp_handler(idx_test, params):
    
    if 'objective' in params:
        map_bootstrp_obj = {
            #'SSE': bootstrp_obj_SSE,
            #'GLS': bootstrp_obj_HJdist,
            'CSR2': bootstrp_obj_CSR2,
            #'GLSR2': bootstrp_obj_GLSR2,
            #'SRexpl': bootstrp_obj_SRexpl,
            #'SR': bootstrp_obj_SR,
            #'MVU': bootstrp_obj_MVutil
        }
        
        def_bootstrp_obj = map_bootstrp_obj[params['objective']]
    else:
        def_bootstrp_obj = bootstrp_obj_CSR2

    ret = params['ret']
    FUN = params['fun']

    n = ret.shape[0]
    idx = np.setdiff1d(np.arange(n), idx_test)  # difference between two arrays, providing training indices
    n_test = len(idx_test)

    invX = np.nan
    invX_test = np.nan
    res = [np.nan, np.nan]

    if n_test > 0:
        r = ret.iloc[idx, :]
        r_test = ret.iloc[idx_test, :]

        if 'cv_cache' not in params or len(params['cv_cache']) <= params['cv_iteration']:
            if 'cv_cache' not in params:
                params['cv_cache'] = {}
            cvdata = {}
            cvdata['X'] = regcov(r)
            cvdata['y'] = np.mean(r, axis=0)
            cvdata['X_test'] = regcov(r_test)
            cvdata['y_test'] = np.mean(r_test, axis=0)
            
            if params['objective'] in {'GLS', 'GLSR2', 'SRexpl'}:
                cvdata['invX'] = np.linalg.pinv(cvdata['X'])
                cvdata['invX_test'] = np.linalg.pinv(cvdata['X_test'])

            params['cv_cache'][params['cv_iteration']] = cvdata

        cvdata = params['cv_cache'][params['cv_iteration']]
        X = cvdata['X']
        y = cvdata['y']
        X_test = cvdata['X_test']
        y_test = cvdata['y_test']
        
        if params['objective'] in {'GLS', 'GLSR2', 'SRexpl'}:
            invX = cvdata['invX']
            invX_test = cvdata['invX_test']

        phi, params = FUN(X, y, params)[0:2]

        if 'cache_run' not in params or not params['cache_run']:
            if 'cv_phi' not in params:
                params['cv_phi'] = {}
            params['cv_phi'][params['cv_iteration']] = phi
            if 'cv_MVE' not in params:
                params['cv_MVE'] = {}
            params['cv_MVE'][params['cv_iteration']] = np.dot(r_test, phi)
            
            fact = np.dot(X, phi)
            fact_test = np.dot(X_test, phi)

            if params['ignore_scale']:
                b = np.linalg.lstsq(fact, y, rcond=None)[0]
                b_test = np.linalg.lstsq(fact_test, y_test, rcond=None)[0]
            else:
                b = 1
                b_test = 1

            res = [
                def_bootstrp_obj(np.dot(fact, b), y, invX, phi, r, params),
                def_bootstrp_obj(np.dot(fact_test, b_test), y_test, invX_test, phi, r_test, params)
            ]

    return np.hstack(res), params

def cvpartition_contiguous(n, k):
    """
    Create contiguous partitions for cross-validation.
    
    Parameters:
    - n: int, total number of data points
    - k: int, number of folds/partitions
    
    Returns:
    - indices: list of lists, containing indices for each fold
    """
    s = n // k  # using floor division to ensure integer result
    indices = [None] * k  # Pre-allocating list with k None elements
    
    for i in range(k - 1):
        # Using range indexing to create contiguous partitions
        indices[i] = list(range(s * i, s * (i + 1)))
    
    # Last partition takes the remaining elements
    indices[k - 1] = list(range(s * (k - 1), n))
    
    return indices

def bootstrp_obj_CSR2(y_hat, y, invX, phi, r, params):
    """
    Compute the objective based on the Coefficient of Squared Regression (CSR2).

    Parameters:
    - y_hat: Predicted values
    - y: Actual values
    - invX, phi, r, params: Other parameters that are not used in the computation
      in this function but are kept for consistency with other objective functions.
    
    Returns:
    - obj: The computed CSR2 objective value.
    """
    # Compute the CSR2 objective
    obj = 1 - (np.dot((y_hat - y).T, (y_hat - y))) / (np.dot(y.T, y))
    return obj