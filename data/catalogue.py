import pandas as pd
import numpy as np
from copy import deepcopy

from pyproj import Proj, Transformer
import pygmt

from matplotlib import pyplot as plt
from matplotlib import axes, path



class EarthquakeCatalogue(object):
    def __init__(self, data=None, epsg='epsg:4326', **kwargs):
        self.epsg_latlon = epsg  # WGS84 by default
        if isinstance(data, str):
            # Use the value of input field "data" as name of an input CSV file:
            self.load_from_csv(data, verbose=True, **kwargs)
        elif isinstance(data, pd.DataFrame):
            # Catalogue provided as a pandas.DataFrame instance:
            self.load_from_dataframe(data)


    def __repr__(self):
        pass


    def load_from_dataframe(self, df):
        """
        Load EarthquakeCatalogue instance from a Nx5 2-D numpy array, where columns are ordered in the following
        order: "lon, lat, depth, mag, date"
        :param df: Pandas.DataFrame instance, containing at least these fields: latitude, longitude, depth, magnitude, date.
        """
        self.lons = df['longitude'].values
        self.lats = df['latitude'].values
        self.deps = df['depth'].values
        self.mags = df['magnitude'].values
        self.dates = df['date'].values
        self.east = deepcopy(self.lons)  # Initialization
        self.north = deepcopy(self.lats)  # Initialization
        self.epsg_eastnorth = self.epsg_latlon  # Initialization


    def load_from_csv(self, csvfile, verbose=False, **kwargs):
        """
        :param csvfile: Earthquake catalogue in CSV format.
                        Required column names are: latitude, longitude, depth, magnitude, date.
        :param kwargs: additional keyword-value optional arguments passed to the pandas.read_csv method
        """
        fields = ['date', 'longitude', 'latitude', 'depth', 'magnitude']
        catalog = pd.read_csv(csvfile, names=fields, **kwargs)
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
        self.east = np.array(x_proj)  # Not sure if really useful to preserve original coordinates after projection (to preserve mapping capacity ?)
        self.north = np.array(y_proj)
        self.epsg_eastnorth = epsg


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
        """
        Produce a map of earthquakes with topography

        :return:
        """
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
