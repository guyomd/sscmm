import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime
from openquake.hmtk.seismicity.catalogue import Catalogue
from pyproj import Proj, Transformer
import pygmt
from matplotlib import pyplot as plt
from matplotlib import axes, path

from sscmm.methods.declustering import DeclusteringAlgorithm


DEFAULT_CATALOGUE_KEYS = ['longitude', 'latitude', 'depth', 'magnitude', 'year', 'month', 'day', 'hour', 'minute', 'second']

class EarthquakeCatalogue(object):
    def __init__(self, data=None, epsg='epsg:4326', **kwargs):
        self.epsg_latlon = epsg  # WGS84 by default
        if isinstance(data, str):
            # Use the value of input field "data" as name of an input CSV file:
            self.load_from_csv(data, verbose=True, **kwargs)
        elif isinstance(data, pd.DataFrame):
            # Catalogue provided as a pandas.DataFrame instance:
            self.load_from_dataframe(data)


    """
    def __repr__(self):
        pass
    """


    def load_from_dataframe(self, df):
        """
        Load EarthquakeCatalogue instance from a Pandas.DataFrame, where columns names must correspond to keys of
        DEFAULT_CATALOGUE_KEYS above.
        :param df: Pandas.DataFrame instance
        """
        self.lons = df['longitude'].values
        self.lats = df['latitude'].values
        self.deps = df['depth'].values
        self.mags = df['magnitude'].values
        self.years = df['year'].values
        self.months = df['month'].values
        self.days = df['day'].values
        self.hours = df['hour'].values
        self.minutes = df['minute'].values
        self.seconds = df['second'].values
        self.x = deepcopy(self.lons)  # Initialization
        self.y = deepcopy(self.lats)  # Initialization
        self.epsg_xy = self.epsg_latlon  # Initialization
        self.convert2decimalyear()


    def load_from_catalogue(self, cat):
        """
        Populates attributes of an EarthquakeCatalogue instance by copying attributes from another instance

        :param cat: EarthquakeCatalogue instance
        :return: newcat: EarthquakeCatalogue instance
        """
        for key in DEFAULT_CATALOGUE_KEYS:
            setattr(self, key, getattr(cat, key))
        for key in ['x', 'y', 'epsg_xy', 'epsg_latlon', 'dates']:
            if hasattr(cat, key):
                setattr(self, key, getattr(cat, key))


    def load_from_csv(self, csvfile, verbose=False, **kwargs):
        """
        :param csvfile: Earthquake catalogue in CSV format.
                        Required column names are ginve the list DEFAULT_CATALOGUE_KEYS above.
        :param kwargs: additional keyword-value optional arguments passed to the pandas.read_csv method
        """
        catalog = pd.read_csv(csvfile, usecols=DEFAULT_CATALOGUE_KEYS, header=0, **kwargs)
        if verbose:
            catalog.info()
        self.load_from_dataframe(catalog)


    def decimate(self, indices, inplace=False):
        """
        Decimate the catalog, and return a copy of the EarthquakeCatalogue instance containing
        only rows matching input boolean indices.

        :param indices: array or list of boolean values, specifying which values to keep (True) or remove (False)
        :param inplace: boolean, Flag specifying whether decimation should occur in-place, or not.

        :returns: if inplace==False returns an EarthquakeCatalogue instance, otherwise nothing
        """
        indices = np.array(indices)
        if inplace:
            for att in ['dates', 'lons', 'lats', 'deps', 'mags']:
                setattr(self, att, getattr(self, att)[indices])
        else:
            cat = np.column_stack((self.lons, self.lats, self.depth, self.mags, self.dates))
            return EarthquakeCatalogue(data=cat[indices,:])


    def convert2decimalyear(self):
        """
        Converts dates expressed as YEAR/MONTH/DAY/HOUR/MIN/SEC to decimal years.

        :return: yr_dec, array of dates in units of decimal year
        """
        yr_dec = list()
        for i in range(len(self.years)):
            # Convert dates into decimals years
            yr_dec.append(self.years[i] +
                          datetime(self.years[i],
                                   self.months[i],
                                   self.days[i],
                                   self.hours[i],
                                   self.minutes[i],
                                   self.seconds[i]).timetuple().tm_yday
                          / datetime(self.years[i], 12, 31, 23, 59, 59).timetuple().tm_yday )
        self.dates = np.array(yr_dec)
        return self.dates


    def project2epsg(self, epsg: str):
        """
        Project longitude, latitude coordinates of the original coordinate system (given in self.epsg) to any other
        coordinate system specified in the input EPSG code
        :param epsg: str, EPSG code for the target coordiante system, e.g. 'epsg:2154' for Lambert 93
        :return:
        """
        tr = Transformer.from_crs(self.epsg_latlon.upper(), epsg.upper())
        x_proj = list()
        y_proj = list()
        for lon, lat in zip(self.lons, self.lats):
            new_coords = tr.transform(lon, lat)
            x_proj.append(new_coords[0])
            y_proj.append(new_coords[1])
        self.x = np.array(x_proj)
        self.y = np.array(y_proj)
        self.epsg_xy = epsg
        print(f'Converted {len(self.lons)} epicentral locations:')
        print(f'  Easting range: from [{self.lons.min():.2f}; {self.lons.max():.2f}] ({self.epsg_latlon}) to '+
              f'[{self.x.min():.2f}; {self.x.max():.2f}] ({self.epsg_xy})')
        print(f'  Northing range: from [{self.lats.min():.2f}; {self.lats.max():.2f}] ({self.epsg_latlon}) to ' +
              f'[{self.y.min():.2f}; {self.y.max():.2f}] ({self.epsg_xy})')


    def in_polygon(self, lonlat):
        """
        Return an array of boolean specifying whether each earthquake in the catalogue is located within (True) or
        outside (False) of the polygon which vertices are provided in input variable LONLAT.

        :param lonlat: list of tuples, each element contains a pair of (LON, LAT) coordinates
        :return: isin: numpy array of boolean
        """
        p = path.Path(lonlat)
        coords = [(self.lons[k], self.lats[k]) for k in range(len(self.lons))]
        isin = p.contains_points(coords)
        return np.array(isin)


    def map_events(self, savefig=False, filename='map.jpg'):
        '''
        Produce a map of earthquakes with topography

        :return:
        '''
        bounds = [self.lons.min() ,self.lons.max() ,self.lats.min() ,self.lats.max() ]
        grid = pygmt.datasets.load_earth_relief(resolution="06m", region=bounds)

        inf2_3 = np.where((self.mags >= 2) & (self.mags < 3))[0]
        inf3_4 = np.where((self.mags >= 3) & (self.mags < 4))[0]
        inf4_5 = np.where((self.mags >= 4) & (self.mags < 5))[0]
        inf5_6 = np.where((self.mags >= 5) & (self.mags < 6))[0]
        sup_6 = np.where(self.mags > 6)[0]

        fig = pygmt.Figure()
        fig.grdimage(grid=grid, projection="M15c", frame="a", cmap="geo")
        fig.plot(x=self.lons[inf2_3], y=self.lats[inf2_3], style="c0.1c", pen="red", color="white", label=f"2<=Mw<3")
        fig.plot(x=self.lons[inf3_4], y=self.lats[inf3_4], style="c0.15c", pen="red", color="white", label=f"3<=Mw<4")
        fig.plot(x=self.lons[inf4_5], y=self.lats[inf4_5], style="c0.2c", pen="red", color="white", label=f"4<=Mw<5")
        fig.plot(x=self.lons[inf5_6], y=self.lats[inf5_6], style="c0.3c", pen="red", color="white", label=f"5<=Mw<6")
        fig.plot(x=self.lons[sup_6], y=self.lats[sup_6], style="c0.4c", pen="red", color="white", label=f"Mw>6")
        fig.coast(borders=["1/0.5p,black"], shorelines=True, rivers=["1/0.5p,blue"], lakes=["blue"], resolution='i')
        fig.basemap(frame=True)
        fig.legend(transparency=20)
        if savefig:
            fig.savefig(filename)
        fig.show()


    def convert2hmtk(self):
        """
        Convert the current catalogue instance into an Openquake's HMTK Catalogue object

        :return: an openquake.hmtk.seismicity.catalogue.Catalogue instance
        """
        cat = Catalogue()
        keys = ['longitude', 'latitude', 'year', 'month', 'day', 'magnitude']
        data_array = np.stack((self.lons, self.lats, self.years, self.days, self.mags), axis=1)
        cat.load_from_array(keys, data_array)
        return cat


    def decluster(self, return_catalogue=False, **kwargs):
        """
        Apply a Declustering Algorithm to the current catalogue

        :return:
        """
        algo = DeclusteringAlgorithm(method, prms)
        output = algo.run(self, **kwargs)

        if return_catalogue and ('flag' in output.keys()):
            main_indx = np.where(output['flag'] == 0)[0]
            catms = EarthquakeCatalogue()
            catms.load_from_catalogue(self)
            catms.decimate(main_indx, inplace=True)
            return output, catms
        else:
            return output