
import numpy as np
from scipy.special import gammainc
from matplotlib import pyplot as plt ## TODO: To replace by pygmt!


class WaitingTimesDistribution(object):
    def __init__(self, name):
        self.name = name
        self.fit_method = getattr(self, f'_fit_{self.name}')
        self.obs = None
        self.loglikelihood = getattr(self, f'_loglikelihood_{self.name}')


    def load(self, dt):
        self.obs = np.array(dt)


    def adjust_model(self, dt: np.ndarray):
        """
        Adjusts a series of waiting-times to the distribution specified in self.name.

        :param dt: np.ndarray, series of waiting-times
        :return:
        """
        prms = self.fit_method(dt)
        return prms


    def _fit_exp(self, dt):
        """
        Performs adjustment of an exponential distribution on a series of waiting-times, using the maximum likelihood
        estimate of the rate paratemer R/
        The exponential distribution has a single parameter r, and reads:
        f(x) = r * exp(-r * dt)

        :param dt: np.array, waiting-times series
        :return: r: best-estimate rate paramter
        """
        self.obs = dt
        r = 1 / np.mean(dt)
        self.prms = np.array([r])
        return r


    def _fit_gamma(self, dt):
        """
        Performs adjustment of a gamma distribution on a series of waiting-times, according to the procedure described in
        Hainzl et al., 2006, BSSA, 96, 1, p.313.
        The gamma distribution has two parameters a and b, and reads:
        f(x) = A * exp(-a * x) * dt ** (-b)
        with A = a ** (1-b) / gamma(1-b)

        :param dt: np.array, waiting-times series
        :return: a, b: best-estimate of gamma distribution parameters
        """
        self.obs = dt
        m = np.mean(dt)
        v = np.var(dt)
        a = m / v
        b = 1 - m * m / v
        self.prms =  np.array([a, b])
        return a, b


    def normalize(self, inplace=False, verbose=True):
        """
        Normalize waiting-times by average event rate --> useful when comparing
        distributions having different average rates

        :param inplace: bool, Logical flag specifying whther normalization should occur in-place on self.dt
        """
        avgrate = len(self.obs) / self.obs.sum()
        if verbose:
            print(f'Normalization: multiply by avg. occurrence rate R = {avgrate:.4g}' +
                  'occurrence/(unit time)')
        if inplace:
            self.obs = self.obs * avgrate
        else:
            return self.obs * avgrate


    def plot(self):
        """
        Plot observed (self.dt) distribution of waiting times and model (self.prms), if available.

        :return:
        """
        if not hasattr(self, 'obs'):
            raise ValueError('Missing self.obs attribute to current instance. Use method self.adjust_model to allocate self.obs.')

        bins = np.logspace(np.log10(np.min(self.obs)), np.log10(max(self.obs))  , num=30)
        cnts = np.histogram(self.obs, bins=bins, density=True)[0]
        nb  = len(bins)
        x = bins[0:nb-1] + 0.5 * np.diff(bins)
        plt.figure()
        plt.plot(x, cnts, 'ob', label='obs')
        if hasattr(self, 'prms'):
            if self.name == 'gamma':
                a = self.prms[0]
                b = self.prms[1]
                A = (a ** (1-b)) / gamma(1-b)
                f = A * np.exp(-a * x) * (x ** (-b))
                labelmodel = f'gamma: a={a} b={b}'
            elif self.name == 'exp':
                r = self.prms[0]
                f = r * np.exp(-r * x)
                labelmodel = f'exp: $\lambda$={r:.4g}'
            plt.plot(x, f, 'r', label=labelmodel)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Waiting times')
        plt.ylabel('probability density function')
        plt.legend()
        plt.grid(True, which='both')
        plt.show()
        return plt.gcf


    def _loglikelihood_gamma(self, prms1, n1, prms2=None, n2=None):
        """
        Compute the log-likelihood function for a gamma distribution of waiting-times
        f(dt) = C.exp(-a * dt).(dt)^(-b) where C = a ** (1-b) / gamma(1-b)
        See formulas 5 and 6 in Daniel et al. (2011).
        """
        L = 0
        dt = np.sort(self.obs)
        a1 = prms1[0]
        b1 = prms1[1]
        if (prms2 is None) or (n2 is None):
            a2 = a1
            b2 = b1
            n2 = 0
        else:
            a2 = prms2[0]
            b2 = prms2[1]

        L = -n1 / (n1+n2) * gammainc(1 - b1, a1 * dt[-1]) - n2 / (n1+n2) * gammainc(1 - b2, a2 * dt[-1]) \
                + np.sum( np.log(
                     n1 / (n1 + n2) * (gammainc(1 - b1, a1 * dt[1:]) - gammainc(1 - b1, a1 * dt[0:-1])) \
                   + n2 / (n1 + n2) * (gammainc(1 - b2, a2 * dt[1:]) - gammainc(1 - b2, a2 * dt[0:-1])) \
                        ) )
        return L


    def _loglikelihood_exp(self, r1, n1, r2=None, n2=None):
        """
        Compute the log-likelihood function for the exponential model
        f(dt) = R * exp(-R * dt)
        """
        L = 0
        dt = np.sort(self.obs)
        if (r2 is None) or (n2 is None):
            r2 = r1
            n2 = 0

        L = -n1 / (n1 + n2) * (1 - np.exp(-r1 * dt[-1])) - n2 / (n1 + n2) * (1 - np.exp(-r2 * dt[-1])) \
            + np.sum(np.log(
                n1 / (n1 + n2) * (np.exp(-r1 * dt[0:-1]) - np.exp(-r1 * dt[1:])) \
              + n2 / (n1 + n2) * (np.exp(-r2 * dt[0:-1]) - np.exp(-r2 * dt[1:])) \
            ))
        return L

