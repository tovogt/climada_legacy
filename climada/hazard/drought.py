
"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---


Define Drought class.
"""

__all__ = ['Drought']


import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
from scipy import sparse

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard

from climada.util.files_handler import download_file
from climada.util.dates_times import datetime64_to_ordinal
from climada.util.constants import DATA_DIR

logging.root.setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)


DFL_THRESHOLD = -1
DFL_INTENSITY_DEF = 1


SPEI_FILE_URL = r'http://digital.csic.es/bitstream/10261/153475/8'
SPEI_FILE_DIR = os.path.join(DATA_DIR, 'system')
SPEI_FILE_NAME = r'spei06.nc'

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'DR'
""" Hazard type acronym Drought """


class Drought(Hazard):
    """Contains drought events.

    Attributes:
        SPEI (float): Standardize Precipitation Evapotraspiration Index
    """
    vars_opt = Hazard.vars_opt.union({'spei'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)

        self.file_url = SPEI_FILE_URL
        self.file_dir = SPEI_FILE_DIR
        self.file_name = SPEI_FILE_NAME
        self.threshold = DFL_THRESHOLD
        self.intensity_definition = DFL_INTENSITY_DEF


    def set_area(self, latmin, lonmin, latmax, lonmax):
        """Set the area to analyse"""
        self.latmin = latmin
        self.lonmin = lonmin
        self.latmax = latmax
        self.lonmax = lonmax


    def set_file_path(self, path):
        """Set path of the SPEI data"""
        self.file_dir = os.path.dirname(path)
        self.file_name = os.path.basename(path)

        self.path = os.path.join(self.file_dir, self.file_name)

#    def set_file_url(self, file_url):
#        """Set url to download the file, if not already in the folder"""
#        self.file_url = file_url

    def set_file_dir(self, file_dir):
        """Set file directory with data"""
        self.file_dir = file_dir

    def set_file_name(self, file_name):
        """Set the file name of the data"""
        self.file_name = file_name

    def set_threshold(self, threshold):
        """Set threshold"""
        self.threshold = threshold

    def set_intensity_def(self, intensity_definition):
        """Set intensity definition"""
        self.intensity_definition = intensity_definition



    def __read_indices_spei(self, dataset):
        """Read the NETCDF file containing SPEI"""

        lat_total = dataset.lat.data
        lon_total = dataset.lon.data
        index_lon = np.where(np.logical_and(lon_total >= self.lonmin,
                                            lon_total <= self.lonmax))[0]
        index_lat = np.where(np.logical_and(lat_total >= self.latmin,
                                            lat_total <= self.latmax))[0]

        lat_vector = dataset.lat[index_lat[0]:index_lat[len(index_lat)-1]].data
        lon_vector = dataset.lon[index_lon[0]:index_lon[len(index_lon)-1]].data
        self.time_vector = dataset.time.data
        self.lat_vector = lat_vector
        self.lon_vector = lon_vector
        self.timeforname = self.time_vector

        spei_matrix = dataset.spei[:, index_lat[0]:index_lat[len(index_lat)-1],
                                   index_lon[0]:index_lon[len(index_lon)-1]].data

        return spei_matrix


    def setup(self):
        """Set up the hazard drought"""

        try:

            file_path = os.path.join(self.file_dir, self.file_name)

            if not os.path.isfile(file_path):

                try:
                    path_dwl = download_file(self.file_url + '/'+ self.file_name)

                    try:
                        os.rename(path_dwl, file_path)

                    except:
                        raise FileNotFoundError('The file ' + str(path_dwl)\
                                + ' could not be moved to ' + str(self.file_dir))

                except:
                    raise FileExistsError('The file ' + str(file_path)\
                                + ' could not '\
                                + 'be found. Please download the file '\
                                + 'first or choose a different folder. '\
                                + 'The data can be downloaded from '\
                                + SPEI_FILE_URL)

            LOGGER.debug('Importing %s', str(SPEI_FILE_NAME))
            dataset = xr.open_dataset(file_path)

        except:
            LOGGER.error('Importing the SPEI data file failed. ' \
                         'Operation aborted.')
            raise

        spei_3d = self.__read_indices_spei(dataset)
        spei_2d = self.__traslate_matrix(spei_3d)

        intensity_matrix_min = self.__get_intensity_from_2d(spei_2d, self.intensity_definition)
        self.hazard_def(intensity_matrix_min)

        return self


    def __traslate_matrix(self, spei_3d):
        """ return hazard intensity as a simple threshold on the SPEI values
        Parameters: see read_indices_spei, just call before
        Returns: matrix
        sparse.csr_matrix
        """

        intensity_thres = self.threshold

        n_centroids = spei_3d.shape[1] * spei_3d.shape[2]
        n_timesteps = spei_3d.shape[0]
        spei_2d = np.zeros((n_timesteps, n_centroids))


        for i in range(n_timesteps):

            one_event_1d = spei_3d[i, :, :]

              # get rid of nan's
            nan_pos = np.isnan(one_event_1d)
            one_event_1d[nan_pos] = 0

            # apply threshold
            non_drought_pos = np.where(one_event_1d > intensity_thres)
            one_event_1d[non_drought_pos] = 0

            one_event_array = one_event_1d.reshape(n_centroids)

            spei_2d[i, :] = one_event_array

        return spei_2d


    def hazard_def(self, intensity_matrix):
        """ return hazard set
        Parameters: see intensity_from_spei
        Returns:
            Drought, full hazard set
            check using new_haz.check()
        """

        self.tag = TagHazard(HAZ_TYPE, 'TEST')

        self.intensity = sparse.csr_matrix(intensity_matrix)

        self.units = 'SPEI'

        # fill centroids th bad way (there must be a code like grid...)
        lat_2d = np.zeros([self.lat_vector.shape[0], self.lon_vector.shape[0]])
        lon_2d = np.zeros([self.lat_vector.shape[0], self.lon_vector.shape[0]])
        n_centroids = self.lat_vector.shape[0] * self.lon_vector.shape[0]
        for lat_i in range(0, self.lat_vector.shape[0]):
            for lon_i in range(0, self.lon_vector.shape[0]):
                lat_2d[lat_i, lon_i] = self.lat_vector[lat_i]
                lon_2d[lat_i, lon_i] = self.lon_vector[lon_i]

        lon_1d = lon_2d.reshape(n_centroids,)
        lat_1d = lat_2d.reshape(n_centroids,)

        self.centroids.coord = np.zeros((n_centroids, 2))
        self.centroids.coord[:, 0] = lat_1d
        self.centroids.coord[:, 1] = lon_1d
        self.centroids.id = np.arange(n_centroids)

        self.event_id = np.arange(1, self.n_years+1, 1)
        # frequency set when all eventsavailable
        #self.frequency = np.array([1])
        #per default equal to event_id
        name_list = []

        time = pd.to_datetime(self.timeforname)

        for i in range(13, len(time), 12):
            name_list.append(str(time[i].year))
        self.event_name = name_list

        self.frequency = np.ones(self.n_years)/self.n_years

        self.fraction = self.intensity.copy()
        self.fraction = self.intensity.copy().tocsr()
        self.fraction.data.fill(1)
        self.date = np.arange(1, self.n_years+1, 1)
        #new_haz.orig =
        self.check()
        return self


    def __get_intensity_from_2d(self, spei_2d, intensity_definition=1):
        """Parameters: the 2D matrix called 'spei_2D' defined in
        intensity_from_spei, which containes every time and spacial resolution
        pixel with either the SPEI value or zero if the pixel value doesn't
        reach the threshold value.
        Returns: matrix
        The matrix with the intensity of every event (maximum one per year).
        The intensity is simply the maximum value for
        the event."""

        n_centroids = spei_2d.shape[1]

        time = pd.to_datetime(self.time_vector)
        first_year = time[0].year + 1

        first_month = time[0].month

        #index_offset to get index of january of first year considered
        index_offset = 12 - first_month + 1


        if time[0].month > 10:
            first_year += 1
            index_offset += 12


        last_year = time[len(time)-1].year

        if time[len(time)-1].month < 9:
            last_year -= 1


        n_years = last_year - first_year + 1 # the first year not counted
        years_vector = np.arange(first_year, last_year)
        self.date = years_vector
        self.n_years = n_years

        intensity_min_matrix = np.zeros((n_years, n_centroids))
        intensity_sum_matrix = np.zeros((n_years, n_centroids))
        intensity_sum_without_th_matrix = np.zeros((n_years, n_centroids))
        date_start_matrix = np.zeros((n_years, n_centroids))
        date_end_matrix = np.zeros((n_years, n_centroids))


        time = time[index_offset - 3: index_offset + 12*n_years - 3]
        self.time_vector = self.time_vector[index_offset - 3: index_offset +
                                            12*n_years - 3]

        for pixel in range(n_centroids):

            array_time_centroid = spei_2d[index_offset - 3: index_offset +
                                          12*n_years - 3, pixel]


            list_events = self.__create_list(array_time_centroid)

            [intmin, intsum, intsumthr, start, end] = self.__read_list(list_events, years_vector, first_year)

            intensity_min_matrix[:, pixel] = intmin
            intensity_sum_matrix[:, pixel] = intsum
            intensity_sum_without_th_matrix[:, pixel] = intsumthr
            date_start_matrix[:, pixel] = start
            date_end_matrix[:, pixel] = end

            self.date_start = date_start_matrix
            self.date_end = date_end_matrix
            self.date_start = sparse.csr_matrix(date_start_matrix)


        if intensity_definition == 1:
            return intensity_min_matrix
        if intensity_definition == 2:
            return intensity_sum_without_th_matrix


        return intensity_sum_matrix


    def __create_list(self, array_time_in_centroid):
        """Return a list of all the events exceeding the threshold.
        The list contains start end date, the minimum value, the sum (integral)
        and the sum minus threshold"""

        event = 0
        min_spei = 0
        sum_spei = 0

        list_events_info = list()
        #create a list with every event exeeding the threshold
        for time_idx in range(len(array_time_in_centroid)):

            if array_time_in_centroid[time_idx] == 0:

                if event:
                    event = 0
                    list_events_info.append([start_time, end_time, min_spei,
                                             sum_spei, sum_spei_thr])
                    min_spei = 0
                    sum_spei = 0
                    sum_spei_thr = 0

            else:
                if event:
                    end_time = self.time_vector[time_idx]
                    sum_spei += array_time_in_centroid[time_idx]
                    sum_spei_thr += (array_time_in_centroid[time_idx] - self.threshold)
                    if array_time_in_centroid[time_idx] < min_spei:
                        min_spei = array_time_in_centroid[time_idx]

                else:
                    start_time = self.time_vector[time_idx]
                    end_time = self.time_vector[time_idx]
                    min_spei = array_time_in_centroid[time_idx]
                    sum_spei = array_time_in_centroid[time_idx]
                    sum_spei_thr = sum_spei - self.threshold

                    event = 1

        return list_events_info


    def __read_list(self, list_events_info, years_vector, first_year):
        """read the list created in method __create_list and choose the
        events to include in the hazard set. For every year maximum one
        event is taken, the one with the lowest spei value on it"""

        intensity_min_array = np.zeros((self.n_years))
        intensity_sum_array = np.zeros((self.n_years))
        intensity_sum_thr_array = np.zeros((self.n_years))
        date_start_array = np.zeros((self.n_years))
        date_end_array = np.zeros((self.n_years))

        year_offset = first_year
        min_spei_offset = 0



        for idx_event in range(0, len(list_events_info)):

            start = list_events_info[idx_event][0]
            end = list_events_info[idx_event][1]

            min_spei = list_events_info[idx_event][2]
            sum_spei = list_events_info[idx_event][3]
            sum_spei_thr = list_events_info[idx_event][4]


            year_start = pd.to_datetime(list_events_info[idx_event][0]).year
            month_start = pd.to_datetime(list_events_info[idx_event][0]).month

            if month_start > 10:
                year_start += 1

            idx_year = np.where(years_vector == year_start)

            if year_offset == year_start:
                if min_spei < min_spei_offset:
                    intensity_min_array[idx_year] = min_spei
                    intensity_sum_array[idx_year] = sum_spei
                    intensity_sum_thr_array[idx_year] = sum_spei_thr
                    date_start_array[idx_year] = datetime64_to_ordinal(start)
                    date_end_array[idx_year] = datetime64_to_ordinal(end)

                    min_spei_offset = min_spei

            else:
                intensity_min_array[idx_year] = min_spei
                intensity_sum_array[idx_year] = sum_spei
                intensity_sum_thr_array[idx_year] = sum_spei_thr
                date_start_array[idx_year] = datetime64_to_ordinal(start)
                date_end_array[idx_year] = datetime64_to_ordinal(end)

                min_spei_offset = min_spei

            year_offset = year_start

        return intensity_min_array, intensity_sum_array, intensity_sum_thr_array, date_start_array, date_end_array