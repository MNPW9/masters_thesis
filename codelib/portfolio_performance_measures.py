import numpy as np
from codelib.portfolio_measures import portfolio_std


def sharpe_ratio(port_mean: float, port_var: float, rf: float = 0) -> float:
    """
    Function that returns the Sharpe-Ratio of a portfolio
    
    Parameters
    ----------
    weights:
        Portfolio weights
    mu:
        Vector of asset returns
    cov_mat:
        Covariance matrix of asset returns
    rf:
        Risk-free rate of return

    Returns
    -------
    float
        Sharpe-Ratio performance measure

    """

    return (port_mean - rf) / np.sqrt(port_var)


def calculate_sharpe_ratio(weights: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray, rf: float = 0) -> float:
    """
    Function that returns the Sharpe-Ratio of a portfolio

    Parameters
    ----------
    weights:
        Portfolio weights
    mu:
        Vector of asset returns
    cov_mat:
        Covariance matrix of asset returns
    rf:
        Risk-free rate of return

    Returns
    -------
    float
        Sharpe-Ratio performance measure

    """

    mean = weights @ mu
    var = weights @ cov_mat @ weights

    return (mean - rf) / np.sqrt(var)


def calculate_cc_ratio(weights: np.ndarray, cov_mat: np.ndarray):

    """
    Calculates the diversification ratio of Chouefaty and Coignard (2008)

    .. math::

        \\begin{equation}
            \\text{GLR}(w, \\Sigma) = \\frac{\\sum_{i=1}^N w_i \\sigma_i}{\\sqrt{w^{\\top} \\Sigma w}}
        \\end{equation}

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

    Parameters
    ----------
    weights:
        Portfolio weights.
    cov_mat:
        Covariance matrix.
    Returns
    -------
    float
        Diversification ratio.
    """

    port_std = portfolio_std(weights=weights, cov_mat=cov_mat)

    vol_vec = np.sqrt(np.diag(cov_mat))
    avg_std = np.inner(weights, vol_vec)

    return avg_std / port_std


def drawdown(index: np.ndarray):
    """
    Calculates the running draw down

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

    Parameters
    ----------
    index:
        Values of e.g. an equity index

    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        Drawdown, index of running maximum

    """

    indexmax = np.maximum.accumulate(index)
    drawdowns = (index - indexmax) / indexmax

    return drawdowns, indexmax


def maxdrawdown(index: np.ndarray):
    """
    Calculates maximum draw down

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

    Parameters
    ----------
    index:
        Values of e.g. an equity index

    Returns
    -------
    float
        Maximum drawdown

    """

    return drawdown(index)[0].min()
