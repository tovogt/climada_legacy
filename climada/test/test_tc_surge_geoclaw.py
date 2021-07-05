"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test non-trivial runs of TCSurgeGeoClaw class
"""

import datetime as dt
import unittest

import numpy as np
import rasterio
import xarray as xr

from climada import CONFIG
from climada.hazard.tc_surge_geoclaw import setup_clawpack, GeoclawRunner, GEOCLAW_WORK_DIR

DATA_DIR = CONFIG.hazard.test_data.dir()
ZOS_PATH = DATA_DIR.joinpath("zos_monthly.nc")
TOPO_PATH = DATA_DIR.joinpath("surge_topo.tif")


class TestGeoclawRun(unittest.TestCase):
    """Test functions that set up and run GeoClaw instances"""

    def test_surge_from_track(self):
        """Test geoclaw_surge_from_track function"""
        # create artificial data in new files
        topo_path = DATA_DIR.joinpath("surge_topo2.tif")
        zos_path = DATA_DIR.joinpath("zos_monthly2.nc")

        ds = xr.open_dataset(ZOS_PATH)
        ds.zos[:] = 0
        ds.to_netcdf(zos_path)

        with rasterio.open(TOPO_PATH, "r") as src:
            data = src.read(1)

        data[:] = -1000
        # steep increase offshore
        data[100:400] = np.linspace(-1000, -10, 300)[:,None]
        data[800:1100] = np.linspace(-10, -1000, 300)[:,None]
        # shallow waters closer to coast
        data[400:500] = np.linspace(-10, 0.5, 100)[:,None]
        data[700:800] = np.linspace(0.5, -10, 100)[:,None]
        # flat inland area
        data[500:700] = 0.5

        # add levee-enclosed areas at coast
        data[650:750,400:500] = 1
        data[675:725,425:475] = 0.5
        data[650:750,510:610] = 1
        data[675:725,535:585] = 0.5

        kwargs = dict(width=src.width, height=src.height, transform=src.transform,
                      compress="deflate", crs=src.crs, count=1, dtype=data.dtype,
                      nodata=-1000)
        with rasterio.open(topo_path, "w", **kwargs) as dst:
            dst.write_band(1, data)

        # import matplotlib.pyplot as plt
        # plt.imshow(data)
        # plt.gca().axis("equal")
        # plt.show()
        # return

        npositions = 15
        dampen = 1 - np.linspace(-0.1, 0.9, npositions)**2
        track = xr.Dataset({
            'radius_max_wind': ('time', dampen * 15 + (1 - dampen) * 50),
            'radius_oci': ('time', dampen * 180 + (1 - dampen) * 200),
            'max_sustained_wind': ('time', dampen * 130 + (1 - dampen) * 34),
            'central_pressure': ('time', dampen * 900 + (1 - dampen) * 1013),
            'time_step': ('time', np.full((npositions,), 3, dtype=np.float64)),
        }, coords={
            'time': (np.datetime64('2010-02-05T09')
                     + np.arange(npositions) * np.timedelta64(3, 'h')),
            'lat': ('time', -27 + np.arange(npositions) * 0.55),
            'lon': ('time', np.full((npositions,), -149.4, dtype=np.float64)),
        }, attrs={
            'sid': '2010029S12177_test',
        })
        centroids = np.array([
            # Out of reach of TC
            [-25, -151.0],
            # Within reach of TC
            [-24.5, -150.5], [-24.0, -150.0], [-23.5, -149.5], [-24.0, -149.0], [-24.5, -148.5],
        ])
        gauges = [
            (-25.0, -149.9),  # offshore
            (-25.0, -148.9),  # offshore
            (-24.5, -149.9),  # coastal (on top of levee)
            (-24.5, -148.9),  # coastal (on top of levee)
            (-24.25, -149.9), # within levee
            (-24.25, -148.9), # within levee
            (-23.5, -149.9),  # inland (behind levee)
            (-23.5, -148.9),  # inland (behind levee)
        ]

        base_dir = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{track.sid}"
        base_dir = GEOCLAW_WORK_DIR.joinpath(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        areas = {
            "wind_area": (-152.4, -29.5, -146.4, -18.6),
            "landfall_area": (-151.12, -26.96, -147.68, -20.88),
            "surge_areas": [(-150.001, -24.001, -148.999, -23.499)],
        }
        time_offset = np.datetime64('2010-02-06T00')

        setup_clawpack()
        runner = GeoclawRunner(base_dir, track, time_offset, areas, centroids, zos_path,
                               topo_path, gauges=gauges)
        runner.run()
        intensity = runner.surge_h
        gauge_data = runner.gauge_data

        print(intensity)
        print(gauge_data[3]['height_above_geoid'])
        import pickle
        np.savez("results.npz", intensity=intensity, centroids=centroids)
        with open("gauges.pickle", "wb") as fp:
            pickle.dump(gauge_data, fp)

        self.assertEqual(intensity.shape, (centroids.shape[0],))
        self.assertTrue(np.all(intensity[:7] > 0))
        self.assertTrue(np.all(intensity[7:] == 0))
        for gdata in gauge_data:
            self.assertTrue((gdata['time'][0][0] - track.time[0]) / np.timedelta64(1, 'h') >= 0)
            self.assertTrue((track.time[-1] - gdata['time'][0][-1]) / np.timedelta64(1, 'h') >= 0)
            self.assertAlmostEqual(gdata['base_sea_level'][0], 1.3008515)
        self.assertLess(gauge_data[0]['topo_height'][0], 0)
        self.assertTrue(0 <= gauge_data[1]['topo_height'][0] <= 10)
        self.assertGreater(gauge_data[2]['topo_height'][0], 10)
        offshore_h = gauge_data[0]['height_above_geoid'][0]
        self.assertGreater(offshore_h.max() - offshore_h.min(), 0.5)
        self.assertEqual(np.unique(gauge_data[2]['height_above_geoid'][0]).size, 1)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGeoclawRun)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
