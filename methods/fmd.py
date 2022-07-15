import numpy as np

def wiechert1980(mags, nums, durations, dm):
    """
    Estimation of Gutenberg-Richter (a,b) parameters by maximum
    likelihood for variable observation periods for different magnitude
    increments.
    Source: Wiechert (1980), Bull. Seismol. Soc. Am., vol. 70, num. 4, pp. 1337-1346.

    :param mags: list, Central value for each magnitude bin
    :param nums: list, Number of events for each magnitude bin
    :param durations: list, lengths of observation periods for each magnitude bin (in years)
    :param dm: float, magnitude bin width

    :returns (lambda,b), (std. dev. lambda, std. dev. b) where lambda is the annual rate of events above (mags[0]-dm/2)
    """
    beta0 = 1.5  # Initial guess
    mags = np.array(mags)
    nums = np.array(nums)
    durs = np.array(durations)
    mc = mags[0] - dm / 2.0

    def _update_beta(beta):
        """
        Update beta using Newton's method
        """
        nkount = nums.sum()
        snm = (nums * mags).sum()
        sumtex = (durs * np.exp(-beta * mags)).sum()
        stmex = (mags * durs * np.exp(-beta * mags)).sum()
        sumexp = np.exp(-beta * mags).sum()
        stm2x = (mags * mags * durs * np.exp(-beta * mags)).sum()
        # Note: stmex/sumtex  *N - sumnm = 0 for extremum
        d2ldb2 = nkount * (np.power(stmex / sumtex, 2) - stm2x / sumtex)
        dldb = (stmex / sumtex) * nkount - snm
        return beta0 - dldb / d2ldb2, d2ldb2, nkount, sumexp, sumtex

    beta, d2ldb2, nkount, sumexp, sumtex = _update_beta(beta0)
    while np.abs(beta - beta0) >= 0.0001:
        beta0 = beta
        beta, d2ldb2, nkount, sumexp, sumtex = _update_beta(beta0)

    stdv = np.sqrt(-1.0 / d2ldb2)
    b = beta / np.log(10)
    stdb = stdv / np.log(10)
    fngtm0 = nkount * sumexp / sumtex
    fn0 = fngtm0 * np.exp(beta * mc)  # lambda(m>=mc)
    a = np.log10(fn0)
    stdfn0 = fn0 / np.sqrt(nkount)  # To do: Check expression !
    return (fn0, b), (stdfn0, stdb)


def aki1965(mags, mc, dm=0.1, method_uncertainty='ShiBolt1982'):
    """
    Compute catalog b-value using maximum-likelihood formula (Aki, 1965)
    """
    m = mags[mags >= mc]
    mu = m.mean()
    N = len(m)
    beta = 1/(mu-mc)
    b = beta/np.log(10)  # Aki (1965) ML estimator
    if method_uncertainty == 'Aki1965':
        db = b/np.sqrt(N)   # Aki (1965) standard deviation estimate
    elif method_uncertainty == 'ShiBolt1982':
        db = 2.3*np.power(b,2)*np.sqrt(np.power(m-mu,2).sum()/(N*(N-1)))   # Shi and Bolt (1982)
    else:
        raise ValueError(f'Unknown method {method_uncertainty} for the standard deviation of b')
    return b, db
