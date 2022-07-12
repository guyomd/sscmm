from openquake.hmtk.seismicity.declusterer.dec_gardner_knopoff import GardnerKnopoffType1
from openquake.hmtk.seismicity.declusterer.distance_time_windows import GardnerKnopoffWindow, GruenthalWindow,UhrhammerWindow


class GardnerKnopoff1974():
    def __init__(self, rt_window, fs_time_prop=0.0):
        if rt_window.lower() == 'gardnerknopoff':
            self.rtwin = GardnerKnopoffWindow()
        elif rt_window.lower() == 'urhammer':
            self.rtwin = UhrhammerWindow()
        elif rt_window.lower() == 'gruenthal':
            self.rtwin = GruenthalWindow()
        else:
            raise ValueError(f'Unrecognized time-window "{rt_window}"')
        self.fs_time_prop = fs_time_prop


    def run(self, cat_hmtk):
        """
        Run Gardner & Knopoff (1974) declustering algorithm to separate mainshocks from aftershocks/foreshocks in a
        catalogue of earthquakes.

        :param cat_hmtk: Catalogue of earthquakes in the HMTK format
        :param fs_time_prop: float, proportion of the aftershock time-window used for the identification of foreshocks
        :return: flag, numpy.ndarray of integer values: == 0 for mainshocks, == 1 for aftershocks and == -1 for foreshocks
        """
        config = {'time_distance_window': self.rtwin,
                       'fs_time_prop': self.fs_time_prop}
        dec = GardnerKnopoffType1(self.params)
        # flag: == 0 for mainshocks, == 1 for aftershocks and == -1 for foreshocks
        # vcl: cluster index
        vcl, flag = dec.decluster(cat_hmtk, config)
        return flag, vcl