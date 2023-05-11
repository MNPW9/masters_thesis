import numpy as np
from scipy.stats import entropy


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


def calculate_marginal_risks_std(weights: np.ndarray, cov_mat: np.ndarray) -> np.ndarray:
    """
    Function that calculates marginal risk using std. as portfolio risk measure
    Parameters

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    total_risk = np.sqrt(weights @ cov_mat @ weights)
    inner_derivative = cov_mat @ weights

    return inner_derivative / total_risk


def calculate_risk_contributions_std(weights: np.ndarray, cov_mat: np.ndarray, scale: bool = False) -> np.ndarray:
    """
    Function that calculates risk contributions using std. as portfolio risk measure

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix
    scale:
        Scale risk contribution.

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr = calculate_marginal_risks_std(weights, cov_mat)
    mrc = weights * mr

    if scale:
        mrc /= np.sum(mrc)

    return mrc


def calculate_marginal_sharpe(weights: np.ndarray, cov_mat: np.ndarray, mu: np.ndarray, rf: float):

    """
    Function that calculates marginal Sharpe ratio

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix
    mu:
        Expected return vector.
    rf:
        Risk free rate.

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr = calculate_marginal_risks_std(weights, cov_mat)
    excess_mu = mu - rf

    return excess_mu / mr


def calculate_pc_var_contributions(x_p, L):

    port_var = x_p.T @ L @ x_p
    pc_variances = np.diag(L)
    return (pc_variances * np.square(x_p)) / port_var