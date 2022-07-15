from sscmm.methods.zaliapin2008 import NNanalysis
from sscmm.methods.gk1974 import GardnerKnopoff1974

class DeclusteringAlgorithm(object):

    def __init__(self, method, prms: dict):
        self.method = method.lower()
        if self.method == "zaliapin2008":
            required_parameters = ['b', 'd', 'w']
            optional_parameters = ['mc', 'time_norm', 'p', 'q', 'z', 'eta0', 'alpha0']

        elif self.method == 'reasenberg1985':
            required_parameters = []
            optional_parameters = []
            print('Placeholder, not yet available')
            pass

        elif self.method == 'gardner1974':
            required_parameters = ['rtwin']
            optional_parameters = ['fs_time_prop']

        self._parse_parameters(prms, required_parameters, optional=optional_parameters)


    def _parse_parameters(self, prms: dict, required, optional=[]):
        """
        Parsing of parameters for class DeclusteringAlgorithm
        """
        prms_keys = list(prms.keys())
        for key in required:
            if key not in prms_keys:
                raise ValueError(f'Missing required parameter "{key}"')
        # optional parameters:
        kwargs = {}
        for opt in optional:
            if isinstance(opt, str):
                if opt in prms.keys():
                    kwargs.update({opt: prms[opt]})
        self.prms = prms
        self.opts = kwargs


    def apply(self, cat, **kwargs):
        output = getattr(self, f'_run_{self.method}')(cat, **kwargs)
        return output


    def _run_zaliapin2008(self, cat, declustering_method=None, **kwargs):
        """
        Run Nearest-Neighbor analysis based on Zaliapin (2008) algorithm

        :param cat: sscmm.catalogue.EarthquakeCatalogue instance
        :param declustering_method: str, Name of algorithm used for declustering based on Nearest-Neighbor distance.
                               Available values: None, 'zaliapin2020'
        :param kwargs:
        :return:
        """
        # Nearest-neighbor analysis:
        nn = NNanalysis(cat.dates,
                        cat.mags,
                        cat.x,
                        cat.y,
                        self.prms['b'],
                        self.prms['d'],
                        self.prms['w'],
                        **self.opts,
                        **kwargs)
        nn.calc()

        # Declustering, sensu stricto:
        if declustering_method is None:
            return {'nn': nn}

        elif declustering_method == 'zaliapin2020':
            flag = nn.decluster(self.prms['w'], self.prms['eta0'], self.prms['alpha0'], ncat=100)
            return {'flag': flag}  # =0 for mainshocks, =1 for aftershocks


    def _run_reasenberg1985(self, cat, **kwargs):
        print('Placeholder for future development')
        pass


    def _run_gardner1974(self, cat, return_indices=True):
        """
        Run Gardner and Knopoff (1974) declustering algorithm

        :param cat: sscmm.catalogue.EarthquakeCatalogue instance
        :param return_indices: boolean, specficy whether, or not, to return cluster indices as output
        :return: (flag, vcl): flag, array of event type index: == 0 for mainshocks, == 1 for aftershocks and == -1 for foreshocks
                              vcl, array of cluster indices
        """
        cat_hmtk = cat.convert2hmtk()
        flag, vcl = GardnerKnopoff1974(self.prms['rtwin'], **self.opts).run(cat_hmtk)
        if return_indices:
            return {'flag': flag, 'cluster_indx': vcl}
        else:
            return {'flag': flag}