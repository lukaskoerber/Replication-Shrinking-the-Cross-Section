import numpy as np

def demarket(r, mkt, b=None):
    """
    Demarket function to compute market beta and de-market returns.

    Parameters:
    - r: DataFrame or 2D array of returns.
    - mkt: Series or 1D array of market returns.
    - b: Optional; market beta. If not provided, it will be computed.

    Returns:
    - rme: DataFrame or 2D array of de-marketed returns.
    - b: market beta.
    """

    # If b (beta) is not provided, compute it
    if b is None:
        # Create a design matrix with intercept (column of ones) and market returns
        rhs = np.column_stack([np.ones(mkt.shape[0]), mkt])

        # Solve for beta using least squares
        b, _ = np.linalg.lstsq(rhs, r, rcond=None)[0:2]
        b = b[1:]

    # De-market
    rme = r - np.outer(mkt, b)

    return rme, b


def regcov(r):
    """
    Compute the regularized covariance matrix of r.

    Parameters:
    - r: Input data matrix

    Returns:
    - X: Regularized covariance matrix
    """
    
    # Compute covariance matrix
    X = np.cov(r, rowvar=False)

    # Covariance regularization (with flat Wishart prior)
    T, n = r.shape
    a = n / (n + T)
    X = a * np.trace(X) / n * np.eye(n) + (1 - a) * X

    return X


def l2est(X, y, params, compute_errors=False):
    l = params['L2pen']

    if compute_errors:
        Xinv = np.linalg.inv(X + l * np.eye(X.shape[0]))
        
        b = np.dot(Xinv, y)
        se = np.sqrt(1 / params['T'] * np.diag(Xinv))
    else:
        # Solve a system of linear equations instead if errors are not needed
        b = np.linalg.solve(X + l * np.eye(X.shape[0]), y)
        se = np.full(X.shape[0], np.nan)

    return b, params, se


