"""
Define StormEurope class.
"""

__all__ = ['StormEurope']

import logging
import numpy as np
import xarray as xr
import pandas as pd
from scipy import sparse

from climada.hazard.base import Hazard
from climada.hazard.centroids.base import Centroids
from climada.hazard.tag import Tag as TagHazard
from climada.util.files_handler import get_file_names

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'WS'
""" Hazard type acronym for Winter Storm """


class StormEurope(Hazard):
    """ Contains european winter storm events. Historic storm events can be
        downloaded at http://wisc.climate.copernicus.eu/

    Attributes:
        ssi_wisc (np.array, float): Storm Severity Index as recorded in the
            footprint files; this is _not_ the same as that computed by the
            Matlab climada version. Apparently not reproducible from the
            max_wind_gust values only.
            Definition presumably used:
            ssi = sum(area_on_land) * mean(intensity > threshold)^3
            see wisc.climate.copernicus.eu/wisc/#/help/products#tier1_section
        ssi_dawkins (np.array, float): Storm Severity Index as defined in
            Dawkins, 2016, doi:10.5194/nhess-16-1999-2016
    """
    intensity_thres = 15
    """ intensity threshold for storage in m/s """

    vars_opt = Hazard.vars_opt.union({'ssi_wisc', 'ssi_dawkins'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        self.ssi_wisc = np.array([], float)

    def read_footprints(self, path, description=None,
                        ref_raster=None, centroids=None,
                        files_omit='fp_era20c_1990012515_701_0.nc'):
        """ Clear instance and read WISC footprints. Read Assumes that all
            footprints have the same coordinates as the first file listed/first
            file in dir.

            Parameters:
                path (str, list(str)): A location in the filesystem. Either a
                    path to a single netCDF WISC footprint, or a folder
                    containing only footprints, or a globbing pattern to one or
                    more footprints.
                description (str, optional): description of the events, defaults
                    to 'WISC historical hazard set'
                ref_raster (str, optional): Reference netCDF file from which to
                    construct a new barebones Centroids instance. Defaults to
                    the first file in path.
                centroids (Centroids, optional): A Centroids struct, overriding
                    ref_raster
                files_omit (str, list(str), optional): List of files to omit;
                    defaults to one duplicate storm present in the WISC set as
                    of 2018-09-10.
        """

        self.clear()

        file_names = get_file_names(path)

        if ref_raster is not None and centroids is not None:
            LOGGER.warning('Overriding ref_raster with centroids')

        if centroids is not None:
            pass
        elif ref_raster is not None:
            centroids = self._centroids_from_nc(ref_raster)
        elif ref_raster is None:
            centroids = self._centroids_from_nc(file_names[0])

        if isinstance(files_omit, str):
            files_omit = [files_omit]

        LOGGER.info('Commencing to iterate over netCDF files.')

        for fn in file_names:
            if any(fo in fn for fo in files_omit):
                LOGGER.info("Omitting file %s", fn)
                continue
            new_haz = self._read_one_nc(fn, centroids)
            if new_haz is not None:
                self.append(new_haz)

        self.event_id = np.arange(1, len(self.event_id)+1)

        self.tag = TagHazard(
            HAZ_TYPE, 'Hazard set not saved, too large to pickle',
            description='WISC historical hazard set.'
        )
        if description is not None:
            self.tag.description = description

    def _read_one_nc(self, file_name, centroids):
        """ Read a single WISC footprint. Assumes a time dimension of length
            1. Omits a footprint if another file with the same timestamp has
            already been read.

            Parameters:
                file_name (str): Absolute or relative path to *.nc
                centroids (Centroids): Centr. instance that matches the
                    coordinates used in the *.nc, only validated by size.
        """
        ncdf = xr.open_dataset(file_name)

        if centroids.size != (ncdf.sizes['latitude'] * ncdf.sizes['longitude']):
            ncdf.close()
            LOGGER.warning(('Centroids size doesn\'t match NCDF dimensions. '
                            'Omitting file %s.'), file_name)
            return None

        # xarray does not penalise repeated assignments, see
        # http://xarray.pydata.org/en/stable/data-structures.html
        stacked = ncdf.max_wind_gust.stack(
            intensity=('latitude', 'longitude', 'time')
        )
        stacked = stacked.where(stacked > self.intensity_thres)
        stacked = stacked.fillna(0)

        # fill in values from netCDF
        new_haz = StormEurope()
        new_haz.event_name = [ncdf.storm_name]
        new_haz.date = np.array([
            _datetime64_toordinal(ncdf.time.data[0])
        ])
        new_haz.intensity = sparse.csr_matrix(stacked)
        new_haz.ssi_wisc = np.array([float(ncdf.ssi)])
        new_haz.time_bounds = np.array(ncdf.time_bounds)

        # fill in default values
        new_haz.centroids = centroids
        new_haz.units = 'm/s'
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.fraction = new_haz.intensity.copy().tocsr()
        new_haz.fraction.data.fill(1)
        new_haz.orig = np.array([True])

        ncdf.close()
        return new_haz

    @staticmethod
    def _centroids_from_nc(file_name):
        """ Construct Centroids from the grid described by 'latitude'
            and 'longitude' variables in a netCDF file.
        """
        LOGGER.info('Constructing centroids from %s', file_name)
        ncdf = xr.open_dataset(file_name)
        lats = ncdf.latitude.data
        lons = ncdf.longitude.data
        cent = Centroids()
        cent.coord = np.array([
            np.repeat(lats, len(lons)),
            np.tile(lons, len(lats)),
        ]).T
        cent.id = np.arange(0, len(cent.coord))
        cent.resolution = (float(ncdf.geospatial_lat_resolution),
                           float(ncdf.geospatial_lon_resolution))
        cent.tag.description = 'Centroids constructed from: ' + file_name
        ncdf.close()

        cent.set_area_per_centroid()
        cent.set_on_land()

        return cent

    def plot_ssi(self):
        """ Ought to plot the SSI versus the xs_freq, which presumably is the
            excess frequency. """
        pass

    def set_ssi_dawkins(self, on_land=True):
        """ Calculate the SSI according to Dawkins. Differs from the SSI that
            is delivered with the footprint files in that we only use the
            centroids that are on land.

            Parameters:
                on_land (bool): Only calculate the SSI for areas on land,
                    ignoring the intensities at sea.
        """
        if on_land is True:
            assert self.centroids.area_per_centroid.all(),\
                "Have you run set_area_per_centroid yet?"
            area = self.centroids.area_per_centroid \
                * self.centroids.on_land
        else:
            area = self.centroids.area_per_centroid

        self.ssi_dawkins = np.zeros(self.intensity.shape[0])
        
        for i, intensity in enumerate(self.intensity):
            ssi = area * intensity.power(3).todense().T
            self.ssi_dawkins[i] = ssi.item(0)


def _datetime64_toordinal(datetime):
    """ Converts from a numpy datetime64 object to an ordinal date.
        See https://stackoverflow.com/a/21916253 for the horrible details. """
    return pd.to_datetime(datetime.tolist()).toordinal()
