import numpy as np

def portfolio_variance(weights: np.ndarray, cov_mat: np.ndarray) -> float:
    """
    Function that returns the variance of the portfolio

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix

    Returns
    -------
    float
        Variance of portfolio
    """

    return weights @ cov_mat @ weights

def portfolio_std(weights: np.ndarray, cov_mat: np.ndarray) -> float:
    """
    Function that returns the standard deviation of the portfolio

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix

    Returns
    -------
    float
        Standard deviation of portfolio
    """

    return np.sqrt(portfolio_variance(weights, cov_mat))


def portfolio_mean(weights: np.ndarray, mu: np.ndarray) -> float:
    """
    Function that returns the standard deviation of the portfolio

    Parameters
    ----------
    weights:
        Portfolio weights
    mu:
        Expected return vector.

    Returns
    -------
    float
        Expected return of portfolio
    """

    return weights @ mu