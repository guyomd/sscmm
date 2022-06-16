"""
Routines for the declustering of seismicity based on the "space-time-magnitude
nearest-neighbor distance" approach as desribed in Zaliapin et al. (2008)

Ref: Zaliapin, I., Gabrielov, A., Keilis-Borok, V. and Wong, H., 2008, Clustering analysis of seismicity and aftershock identification, Phys. Res. Letters., 101, 1.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar
from pandas import Series
import sys  # for progress bar
from math import ceil # for progress bar

rng = np.random.default_rng()


def _spatial_distance(ref_event_coords, events_coords):
    """
    Compute the spatial distance between a reference event and a set of other events locations.

    :param ref_event_coords: list, reference event coordinates.
    :param events_coords: list, other events coordinates
    :return: array of 2-D or 3-D spatial distances
    """
    x = ref_event_coords[0]
    y = ref_event_coords[1]
    xs = events_coords[0]
    ys = events_coords[1]
    if (len(ref_event_coords) == 3) and (len(events_coords) == 3):
        z = ref_event_coords[2]
        zs = events_coords[2]
        r = np.sqrt(np.power(x - xs, 2) + np.power(y - ys, 2) + np.power(z - zs, 2))
    elif (len(ref_event_coords) == 2) and (len(events_coords) == 2):
        r = np.sqrt(np.power(x - xs, 2) + np.power(y - ys, 2))
    else:
        raise ValueError('Input arguments must have length equal to 2 or 3')
    return r


def _nearest_neighbor_distance(ref_event_coords, events_coords, dt, m, w, d, p, q):
    """
    Compute the space-time-magnitude nearest-neighbor distance between a reference earthquake (offspring)
    and a set of other earthquakes
    :param ref_event_coords: list, reference earthquake coordinates
    :param events_coords: list, other earthquake coordinates
    :param dt: list or np.ndarray, difference in origin times, i.e. (reference - other earthquakes)
    :param m: list or np.ndarray: magnitudes of other earthquakes
    :param w, float: magnitude-scaling parameter, i.e. used in the expression 10^(-w*m)
    :param d, float: fractal dimension of earthquake locations
    :param p, q: float, scalar parameters for the separation of the distance into space and time components R and T
    :return: (n, imin): tutple: (nearest-neighbor distance, index of closest earthquake)

    """
    r = _spatial_distance(ref_event_coords, events_coords)
    n = dt * np.power(r, d) * np.power(10, -w * m)
    # Identify nearest neighbor:
    imin = np.argmin(n)
    R = np.power(r[imin], d) * np.power(10, -w * p * m[imin])
    T = dt[imin] * np.power(10, -w * q * m[imin])
    return n[imin], imin, R, T


class NNanalysis():
    """
    Nearest-Neighbour "space-time-mag distance" object definition
    """
    def __init__(self, b: float, d: float, t, m, x, y, z=None, mc=None, time_norm=1.0):
        """
        Initialize a space-time-magnitude nearest-neighbor distance object.
        Warning: Input arguments T, M, X, Y [,Z] must only account for earthquakes above
        an implicit completeness threshold.
        :param b: float, slope of Gutenberg-Richter frequency-magnitude relationship
        :param d: float, spatial fractal dimension of epicentral cloud
        :param t: list or pandas.Series, earthquake origin times, expressed as floating number in any time unit
        (e.g. sec, hour, year...).Note that parameter "tmin" in self.calc() should be expressed in the same unit.
        :param m: list or pandas.Series, earthquake magnitudes
        :param x: list or pandas.Series, earthquake longitudes, expressed in meters or kilometers
        :param y: list or pandas.Series, earthquake latitudes, expressed in meters or kilometers
        :param z: list or pandas.Series, earthquake depths, expressed in meters or kilometers
        :param mc: scalar, minimum magnitude threshold
        :param time_norm: float, normalization time unit, expressed as a floating number in the same unit than
        parameters t and tmin. Set value to None or 1 to avoid normalization.
        """

        # Check input arguments:
        if isinstance(t, Series):
            t = t.values
        if isinstance(m, Series):
            m = m.values
        if isinstance(x, Series):
            x = x.values
        if isinstance(y, Series):
            y = y.values
        if z is not None:
            if isinstance(z, Series):
                z = z.values
            z_nan = np.isnan(z).sum()
            if z_nan > 0:
                raise ValueError(f'Input Z data (depth) contains {z_nan} NaN values.')

        if isinstance(b, list) or isinstance(b, tuple):
          raise ValueError('Error: Input parameter b must be a scalar! (Current length={})'.format(len(b)))
        self.mc = mc
        self.b = b
        self.d = d

        if mc is None:
            self.t = t
            self.m = m
            self.x = x
            self.y = y
            if z is not None:
                self.z = z
        else:
            # Keep only events with magnitude greater than or equal to MC:
            im = (m>=mc).nonzero()[0]
            self.t = t[im]
            self.m = m[im]
            self.x = x[im]
            self.y = y[im]
            if z is not None:
                self.z = z[im]

        if time_norm is None:
            self.t_norm = 1.0
        else:
            self.t_norm = time_norm

        # Sort events in increasing time order:
        isort = np.argsort(self.t)
        self.t = self.t[isort]
        self.m = self.m[isort]
        self.x = self.x[isort]
        self.y = self.y[isort]
        if z is not None:
            self.z = self.z[isort]


    def calc(self, w, p = 0.5, q = 0.5):
        """
        Compute nearest-neighbour distances using Zaliapin & Bein-Zion (2007) algorithm

        :param w: float, magnitude-scaling parameter. Recommended value can be w = b or w = 0 to remove dependence on m
        :param p, q: float, scalar parameters for the separation of the distance into space and time components R and T
        """
        ne = len(self.t)
        self.eta = np.zeros((len(self.t)-1,))
        self.parent = []
        self.offspring = []
        self.R = np.zeros_like(self.eta)
        self.T = np.zeros_like(self.eta)
        cmax = ne * (ne - 1) / 2
        bar = ProgressBar(cmax, title = 'Compute nearest-neighbor distances', nsym=25)
        ccur = 0
        for jj in range(1,ne):
            predecessors = list(range(jj))
            ccur += len(predecessors)
            evt_coords = [self.x[jj], self.y[jj]]
            predecessors_coords = [self.x[predecessors], self.y[predecessors]]
            if hasattr(self, 'z'):
                evt_coords.append(self.z[jj])
                predecessors_coords.append(self.z[predecessors])
            self.eta[jj-1], imin, self.R[jj-1], self.T[jj-1] = \
                _nearest_neighbor_distance(evt_coords,
                                           predecessors_coords,
                                           self.t[jj] - self.t[predecessors],
                                           self.m[predecessors],
                                           w,
                                           self.d,
                                           p,
                                           q)
            self.offspring.append(jj)
            self.parent.append(predecessors[imin])
            bar.update(ccur)


    def select_from_threshold(self, eta_threshold, method='above', output_format='logical'):
        """
        Return array indices or logicals of event with ETA value above/below specified threshold.

        :param eta_threshold: float, Threshold value for self.eta parameter
        :param method: str, Type of boundary for the selection method. Can be either 'above', 'below', 'strictly_above', 'strictly_below'
        :param output_format: str, Type of output: 'index' or 'logical'
        """
        comparison_rule = {
                          'above': np.greater_equal,
                          'below': np.less_equal,
                          'strictly_above': np.greater,
                          'strictly_below': np.less
                          }
        logic = comparison_rule[method](self.eta, eta_threshold)
        if output_format=='index':
          output = self.current(logic.nonzero()[0])
          if (method=='above') or (method=='strictly_above'):
            # Add the first event to selection (first event is an independent, parent-free event):
            output = np.insert(output, 0, 0)
        elif output_format=='logical':
          # Aadd the first event to selection (first event is an independent, parent-free event):
          if (method=='above') or (method=='strictly_above'):
            output = np.insert(logic, 0, True)
          else:
            output = np.insert(logic, 0, False)
        return output


    def decluster(self, w, eta0, alpha0, ncat = 100):
        """
        Apply the declustering method proposed by Zaliapin & Ben-Zion (2020)
        :param w: float, magnitude-scaling parameter. Recommended value can be w = b or w = 0 to remove dependence on m
        :param eta0, float: Initial cutoff threshold. Recommended values marks the separation of the two modes
        in a logT vs. log R graph.
        :param alpha0, float: cluster threshold alpha0 = log10(A0). Recommended value is -1 < alpha0 < 1:

        :return:
        """
        print('>> Step 1: Identify the most clustered events')
        i1 = (self.eta > eta0).nonzero()[0]
        N0 = len(i1)
        indices = [self.parent[i] for i in i1]

        print('>> Step 2: Coarsely estimate location-specific background intensity')
        tmin = self.t.min()
        tmax = self.t.max()
        ne = len(self.t)
        K = np.zeros((ne - 1, ncat))
        bar = ProgressBar(ncat, title = f'Generate {ncat} randomly-reshuffled catalogues of mainshocks', nsym=25)
        for k in range(ncat):
            m = rng.permutation(self.m[indices])
            t = rng.uniform(tmin, tmax, len(indices))
            for i in range(1, ne):
                evt_coords = [self.x[i], self.y[i]]
                cat_coords = [self.x[indices], self.y[indices]]
                if hasattr(self, 'z'):
                    evt_coords.append(self.z[i])
                    cat_coords.append(self.z[indices])
                K[i-1,k] = _nearest_neighbor_distance(evt_coords,
                                                      cat_coords,
                                                      self.t[i] - t,
                                                      m,
                                                      w,
                                                      self.d,
                                                      0.5,
                                                      0.5)[0]
            bar.update(k + 1)

        print('>> Step 3: Proximity normalization')
        a = np.power(10, np.log10(self.eta) - np.mean(np.log10(K), axis=1))

        print('>> Step 4: Select background events (random thinning)')
        A0 =  np.power(10, alpha0)
        norm_prox = np.insert(a * A0, 0, 1)  # Add normalized proximity of 1 for the first event (set as background)
        self.prob_bgnd = np.minimum(norm_prox, np.ones_like(norm_prox))
        is_bgnd = norm_prox > rng.uniform(0, 1, ne)
        return is_bgnd


    ### Plot methods: ###
    def plotTR(self, density=True, colmap='binary', marker='ok', nbins=20, TRconst=None):
        """
        Produces a plot of rescaled distance R vs. rescaled time T for a set of space-time-magnitude
        distances obtained using Zaliapin (2007)'s approach
        :param density: bool, If True, build density T-R plot. Otherwise, use one marker per (T,R) point
        :param colmap: str, Name of colormap.
                            Default: 'binary' (B&W).
                            Alternatives: 'jet', 'viridis', 'Blues' or any other Matplotlib colormap
        :param marker: str, Symbol character string. Only used when "density" parameter is False.
        :param nbins: int, Number of (hexagonal) bins along X and Y axes for density plot
        :param TRconst: float, If specified, add a "T.R=const" line on plot. Value corresponds to its slope in a log-log plot.

        :returns matplotlib.pyplot.Figure instance
        """
        xlabel = r'$T = {\tau}.10^{-b.m/2}$'
        ylabel = r'$R = r^d.10^{-b.m/2}$'
        title = 'Distribution of time and space components  (b={:.2f}, d={:.2f})'.format(self.b, self.d)
        h = plt.figure()
        cMap = plt.cm.get_cmap(colmap)
        if density is True:
          hb = plt.hexbin(self.T, self.R, xscale='log', gridsize=nbins, yscale='log', cmap=cMap, mincnt=1)
          # Equivalent to plt.hist2d(self.T, self.R, bins=nbins, cmap=cMap)
          cax, kw = colorbar.make_axes(plt.gca(), location='right')
          cbar = colorbar.ColorbarBase(cax, cmap=hb.cmap, norm=hb.norm)
          cbar.set_label('Counts')
          hax = hb.axes
        else:
          plt.loglog(self.T, self.R, marker)
          hax = plt.gca()
        if TRconst is not None:
          Tval = hax.get_xlim()
          print(Tval)
          hax.loglog(Tval, np.divide(TRconst,Tval), 'k', linewidth=2)
        hax.set_xlabel(xlabel)
        hax.set_ylabel(ylabel)
        hax.set_title(title)
        h.show()
        return h


    def plotTRsmooth(self, colmap='jet', nbins=20, TRconst=None):
        """
        Produces a smoothed plot of rescaled distance R vs. rescaled time T for a set of space-time-magnitude
        distances obtained using Zaliapin (2007)'s approach. Smoothing is done using a 2-D Gaussian kernel density estimate.

        :param colmap: str, Name of colormap (any  Matplotlib colormap, e.g. 'binary', 'jet', 'Blues', 'viridis').
        :param nbins: int, Number of (hexagonal) bins along X and Y axes for density plot.
        :param TRconst: float, If specified, add a "T.R=const" line on plot. Value corresponds to its slope in a log-log plot.

        :returns matplotlib.pyplot.Figure instance
        """
        from scipy.stats import gaussian_kde
        xlabel = r'Rescaled time, $\log_{10} T$'
        ylabel = r'Rescaled distance, $\log_{10} R$'
        title = 'Distribution of time and space components  (b={:.2f}, d={:.2f})'.format(self.b, self.d)
        h = plt.figure()
        cMap = plt.cm.get_cmap(colmap)  #'Blues'

        x = np.log10(self.T)
        y = np.log10(self.R)
        data = np.vstack([x, y])
        kde = gaussian_kde(data)

        # evaluate on a regular grid
        xgrid = np.linspace(x.min(), x.max(), nbins)
        ygrid = np.linspace(y.min(), y.max(), nbins)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

        # Plot the result as an image
        plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[x.min(), x.max(), y.min(), y.max()],
           cmap=cMap)
        hax = plt.gca()
        cbar = plt.colorbar()
        cbar.set_label('Density')

        # If required, add power-law trend:
        if TRconst is not None:
          Tval = hax.get_xlim()
          Rval = hax.get_ylim()
          print(Tval)
          hax.plot(Tval, np.subtract(np.log10(TRconst),Tval), 'k', linewidth=2)
          hax.set_xlim(Tval)
          hax.set_ylim(Rval)
        hax.set_xlabel(xlabel)
        hax.set_ylabel(ylabel)
        hax.set_title(title)
        h.show()
        return h


    def plotDistHistogram(self):
        """
        Produce histogram of the nearest-neighbor space-time-magnitude distance SELF.ETA
        :returns matplotlib.pyplot.Figure instance
        """
        etaLog = np.log10(self.eta)
        emin = etaLog[np.isfinite(etaLog)].min()
        emax = etaLog[np.isfinite(etaLog)].max()
        h = plt.figure()
        plt.hist(etaLog, bins=50, log=False, range=(emin, emax))
        plt.xlabel(r'$\log_{10} \;{\eta}^{\star}$')
        plt.ylabel('Counts')
        h.show()
        return h


# Progress bar (duplicated from myLib.progressbar)
class ProgressBar():
    """
    Progress-bar object definition
    """
    def __init__(self, imax: float, title: str='', nsym: int=20):
        """
        :param imax: float, Maximum counter value, corresponding to 100% advancment
        :param title: str, (Optional) Title string for the progress bar
        :param nsym: int, (Optional) Width of progress bar, in number of "=" symbols (default: 20)
        """
        self.imax = imax
        self.title = title
        self.nsym = nsym


    def update(self, i: float, imax: float=None, title: str=None):
        """ Display an ASCII progress bar with advancement level at (i/imax) %

        :param i: float, Current counter value
        :param imax: float, Maximum counter value, corresponding to 100% advancment
        :param title: str, (Optional) Title string for the progress bar
        :param nsym: int, (Optional) Width of progress bar, in number of "=" symbols (default: 20)
        """
        if imax is not None:
            self.imax = imax
        if title is not None:
            self.title = title
        sys.stdout.write('\r')
        fv = float(i)/float(self.imax)  # Fractional value, between 0 and 1
        sys.stdout.write( ('{0} [{1:'+str(self.nsym)+'s}] {2:3d}%').format(self.title, '='*ceil(
            fv*self.nsym), ceil(fv*100)) )
        if i==self.imax:
            sys.stdout.write('\n')
        sys.stdout.flush()

