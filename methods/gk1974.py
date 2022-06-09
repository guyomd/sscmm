from sscmm.data.catalogue import EarthquakeCatalogue

from openquake.hmtk.seismicity.declusterer.dec_gardner_knopoff import GardnerKnopoffType1
from openquake.hmtk.seismicity.declusterer.distance_time_windows import GardnerKnopoffWindow, GruenthalWindow,UhrhammerWindow

dec = GardnerKnopoffType1()
#win = GardnerKnopoffWindow()
#win = GruenthalWindow()
win = UhrhammerWindow()
config = {'time_distance_window':   win,
                'fs_time_prop': 0.1 }
Vcl, flag = dec.decluster(catalog, config)


class GardnerKnopoff1974():
    def __init__(self, rt_window, params):
        pass


    def _set_rt_window(self, window_name):
        self.rtwin = ...  # Link to OQ object
        pass


    def decluster(self, cat: EarthquakeCatalogue):
        pass