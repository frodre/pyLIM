pyLIM
=====
[![DOI](https://zenodo.org/badge/21477157.svg)](https://zenodo.org/badge/latestdoi/21477157)

A python-based linear inverse modeling suite.

**pyLIM** is based on the linear inverse model (LIM) described by Penland & Sardeshmukh (1995).
This package provides the machinery to both calibrate and forecast/integrate a LIM from the
publications Perkins & Hakim ([2017](http://dx.doi.org/10.5194/cp-13-421-2017), submitted-2019).

The documentation and test updates are still in progress, but I hope to update those in the
near future.

## Installation

pyLIM requires Python 3.6+ and the following packages: `numpy, numexpr, netCDF4, dask, pytables,
scipy, and scikit-learn`.

To install pyLIM, `cd` into the package directory after downloading or cloning this repository.

    $ cd /path/to/pylim
    $ python setup.py install
    # -- or if altering pyLIM code --
    $ python setup.py develop

This will install pyLIM for the current python environment.

## Examples

When working with with pyLIM, we start with some data which is
well approximated as a predictably linear system forced by white noise.  Sea-surface temperatures
are a good example of this type of field.  This data will be processed to convert the data to
anomalies and remove seasonal information / long-term drift (if desired).  After pre-processing
we convert the data to its components of primary variability using PCA (specific to geospatial
fields).  `pylim.DataTools` has some generally helpful tools to process data and load from 
netCDF and HDF5 sources. (These are quite specific to CMIP5 climate model data currently)

### Pre-processing
Pre-processing from a netCDF file

    import pylim.DataTools as DT

    sst = DT.BaseDataObject.from_netcdf('sst_dat.nc', 'sst')
    sst.calc_anomaly(12) # calculate monthly anomalies
    sst.time_average_resample('annual', 12) # annual average
    sst.detrend_data() # linearly detrend data
    sst.area_weight_data(use_sqrt=True) 
    sst.eof_proj_data(num_eofs=10)
    # sst.data now has dimensions of ntimes x 10

If the data is too large for the system to hold in memory,  `Hdf5DataObject` stores intermediate data
in an HDF5 container using pytables.  Use `DT.netcdf_tohdf5_container` to convert the netCDF to
HDF5, which I found to be more compatible with Dask.

    import tables as tb 
    dobj_h5 = tb.open_file('tmp_sst_dobj.h5', mode='w',
                           filters=tb.Filters(complib='blosc', complevel=2))
    sst_h5 = DT.Hdf5DataObject.from_hdf5('sst_dat.h5', 'sst', dobj_h5)

### Calibrating a LIM

LIM calibration takes in data of shape `ntimes x nfeatures` and by default calibrates the LIM
using lag-1 covariance statistics.  If data are annually averaged, the base forecast unit
would be 1-year.

    import pylim.LIM as LIM
    
    # data should be ntimes x features
    lim = LIM.LIM(tau0_data=sst.data, fit_noise=True)

For other lags, one can specify the lagged data to calibrate to:

    lim = LIM.LIM(tau0_data=sst.data[:-3], tau1_data=sst.data[3:], fit_noise=True)

### Forecasting using the LIM

Deterministic forecasts for different leads can be done using

    # 1- and 2-year forecasts initialized for every year of the first 10 from SSTs
    t0_data = sst.data[0:10]
    fcast_out = lim.forecast(t0_data, [1, 2])

Noise integration for two years at ~3 hr timestep.  (`t0_data` can be considered a
10-member ensemble for this integration)
   
    final_state = lim.noise_integration(t0_data, 2, timesteps=2880)    

