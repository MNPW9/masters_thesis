import numpy as np

def sharpe_ratio(weights: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray, rf: float = 0) -> float:
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

    return - (mean - rf) / np.sqrt(var)

# FUNCTIONS:
# annulized excess return
# annulized volatility
# sharpe ratio
# maximum drawdown
# # diversificaiton ratio
# # information ratio?
# # turnover

# risk factors: MRC and TRC
