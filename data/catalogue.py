import pandas as pd
import numpy as np
import utm


class EarthquakeCatalogue(object):
    def __init__(self, data=None, **kwargs):
        if isinstance(data, str):
            # Use the value of input field "data" as name of an input CSV file:
            self.load_from_csv(data, **kwargs)
        elif isinstance(data, np.ndarray):
            # Catalogue provided as a Nx5 2-D array:
            self.load_from_2d_array(data)


    def __repr__(self):
        pass


    def load_from_2d_array(self, array):
        """
        Load EarthquakeCatalogue instance from a Nx5 2-D numpy array, where columns are ordered in the following
        order: "lon, lat, depth, mag, date"
        :param array: Nx5 2-D numpy array
        """
        self.lons = array[:, 0]
        self.lats = array[:, 1]
        self.deps = array[:, 2]
        self.mags = array[:, 3]
        self.dates = array[:, 4]


    def load_from_csv(self, csvfile, **kwargs):
        """
        :param csvfile: Earthquake catalogue in CSV format.
                        Required column names are: latitude, longitude, depth, magnitude, date.
        :param kwargs: additional keyword-value optional arguments passed onthe numpy.genfromtxt method
        """
        fields = ['longitude', 'latitude', 'depth', 'magnitude', 'date']
        cat = np.genfromtxt(csvfile, names=fields, **kwargs)
        self.load_from_2d_array(cat)


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
            for att in ['lons', 'lats', 'deps', 'mags', 'dates']:
                setattr(self, att, getattr(self, att)[indices])
        else:
            cat = np.column_stack((self.lons, self.lats, self.depth, self.mags, self.dates))
            return EarthquakeCatalogue(data=cat[indices,:])


    def lonlat2utm(self):
        """
        Convert coordinates expressed in Latitude and Longitude (decimal degrees, WGS84) to
        UTM coordinates (expressed in meters)
        """
        if (not hasattr(self, 'lons')) or (not hasattr(self, 'lats')):
            raise AttributeError('Missing longitude and latitude attributes in EarthquakeCatalogue instance')
        utm_east = list()
        utm_north = list()
        self.utm_zone = list()
        for coord in zip(self.lats, self.lons):
            x, y, num, letter = utm.from_latlon(coord[0], coord[1])
            utm_east.append(x)
            utm_north.append(y)
            self.utm_zone.append(str(num)+letter)
        self.utm_east = np.array(utm_east)
        self.utm_north = np.array(utm_north)