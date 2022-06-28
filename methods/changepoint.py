
import numpy as np
from scipy.special import gammainc
from matplotlib import pyplot as plt ## TODO: To replace by pygmt!
from sscmm.methods.waitingtimes import WaitingTimesDistribution

rng = np.random.default_rng()


def _remove_doublons(x, verbose=False):
    """
    Sort and remove doublons from series X

    :param x: numpy.ndarray, array of floating numbers
    """
    dx = np.diff(np.sort(x))
    i0 = np.where(dx == 0.0)[0]
    if len(i0) > 0:
        if verbose: print(f'Warning! {len(i0)} doublons identified in array of length {len(x)}')
        rv = 1 - rng.random((len(i0),))  # Random-values in (0.0, 1.0]
        x[i0 + 1] = x[i0 + 1] * (1 + rv * 1E-4)  # Adds random fraction of its own value (between 0 and 1E-4)
    return x


class ChangePointFinder(object):

    def __init__(self, t: np.ndarray, wtd_name: str):
        self.t = self._parse_input_times(t)
        self.wtd_name = wtd_name
        if wtd_name == 'gamma':
            self.np1f = 2
            self.np2f = 5
        elif self.wtd_name == 'exp':
            self.np1f = 1
            self.np2f = 3


    def _parse_input_times(self, t):
        t = np.array(t)
        return _remove_doublons(np.sort(t))


    def _compute_waiting_times(self, t):
        dt = np.diff(t)
        nz = np.where(dt > 0)[0]
        dt = np.sort(dt[nz])
        return _remove_doublons(dt)

    def _find_cp(self, t: np.ndarray, nmin=30, display=False):
        """
        Find the optimal change-point in a timeseries based on the modelling
        of waiting-time distribution by a gamma function
        :param nmin: int, minimum number of occurrences permitted in
                          sub-segments of the original timeseries
        """
        dt = self._compute_waiting_times(t)
        # Single model for the whole period:
        dist0 = WaitingTimesDistribution(self.wtd_name)
        prms = dist0.adjust_model(dt)
        L1 = dist0.loglikelihood(prms, len(dt))
        # Dual model with a splitted time-window:
        nt = len(t)
        if nt < (2 * nmin - 2):
            print(f'Timeseries are too short for change-point analysis: num. samples ({nt}) < {2 * nmin -2}')
            return None
        idx = range(nmin - 1, nt - nmin + 1)
        ni = len(idx)
        L2 = np.zeros(ni)
        N = len(dt)
        for isplit in idx:  # Loop on possible change-points
            t1 = t[0:isplit + 1]
            t2 = t[isplit:]
            dt1 = self._compute_waiting_times(t1)
            n1 = len(dt1)
            dt2 = self._compute_waiting_times(t2)
            n2 = len(dt2)
            dist1 = WaitingTimesDistribution(self.wtd_name)
            prms1 = dist1.adjust_model(dt1)
            dist2 = WaitingTimesDistribution(self.wtd_name)
            prms2 = dist2.adjust_model(dt2)
            L2[idx.index(isplit)] = dist0.loglikelihood(prms1, n1, prms2, n2)
        imax= np.argmax(L2)
        L2opt = L2[imax]
        it = nmin - 1 + imax
        BIC_threshold = (self.np2f - self.np1f) /2 * np.log(N)
        if L2opt - L1 > BIC_threshold:  # If condition is verified, keep the change-point
            tcp = t[it]
            print(f'change-point identified at t={tcp}')
        else:
            tcp = None

        if display:
            plt.figure()
            plt.plot(t[idx], L2, 'b-o')
            plt.plot(t[[0, -1]], L1 + BIC_threshold * np.ones((2,)), 'r:')
            if tcp is not None:
                plt.plot(tcp, L2opt, 'ro')
            plt.xlabel('Occurrence times')
            plt.ylabel('Log-likelihood')
            plt.show()

        return tcp


    def run(self, nmin=30, display=True):
        """
        Search for non-overlapping time-windows characterized by homogeneous waiting-time distributions,
        based on a change-point analysis, see Daniel et al., 2011.
        Distribution of waiting-times can be either gamma or expoential.
        """
        if display == 'iter':
            display_iter = True
        else:
            display_iter = False

        twLimits = np.array([self.t.min(), self.t.max()])
        nit = 1
        print(f'--Iteration: {nit}')
        cp = list()  # Initializes the list of change points

        # Search for an initial change-point from the entire timeseries:
        cp.append(self._find_cp(self.t, nmin=nmin, display=display_iter))
        if cp[0] is not None:
            twLimits = np.append(twLimits, cp)
            twLimits.sort()
        else:
            return []

        # Then, iterate over each half of timeseries above and after the previous change point:
        while cp[0] is not None:
            nit += 1
            print(f'--Iteration: {nit}')
            ncp = len(cp)
            cp_it = list()
            for i in range(ncp):  # Maximum: 2 additional change-points per iteration

                # Extract timeseries before and after each change-point:
                itw = np.where(twLimits == cp[i])[0]
                ideb = np.where(self.t == twLimits[itw - 1])[0]
                iend = np.where(self.t == twLimits[itw + 1])[0]
                isplit = np.where(self.t == twLimits[itw])[0]
                tpre, tpost = np.array_split(self.t, [int(ideb), int(isplit), int(iend)])[1:3]

                # Analyze timeseries preceeding change-point:
                cp_pre = self._find_cp(tpre, display=display_iter)
                if cp_pre is not None:
                    twLimits = np.append(twLimits, cp_pre)
                    twLimits.sort()
                    cp_it.append(cp_pre)

                # Analyze timeseries following change-point:
                cp_post = self._find_cp(tpost, display=display_iter)
                if cp_post is not None:
                    twLimits = np.append(twLimits, cp_post)
                    twLimits.sort()
                    cp_it.append(cp_post)

            if len(cp_it) > 0:
                cp = cp_it
            else:
                cp = [None]

        if display:
            self.plot_cp(self.t, twLimits)

        cp = twLimits[1:-1]
        if len(cp) == 0:
            print('\nno change-point found')
        else:
            print(f'\n{len(cp)} change-point(s) found: t={cp}')
        return cp

    def plot_cp(self, t: np.ndarray, twLimits, showCumulative=True):
        """
        Present change-points graphically in timeseries
        """
        ncp = len(twLimits[1:-1])
        plt.figure()
        plt.plot(t, np.arange(len(t)), 'k.')
        ax = plt.gca()
        ylim = ax.get_ylim()
        for k in range(ncp):
            plt.plot([twLimits[k + 1], twLimits[k + 1]], ylim, 'r-')
        plt.xlabel('Time')
        plt.grid(True)
        plt.ylabel('Cumulative number of occurrences')
        plt.show()


