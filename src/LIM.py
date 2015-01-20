#-*- coding: utf-8 -*-
from numpy import sqrt, cos, radians, dot, log, exp, zeros, array, int16, copy
from numpy import linspace, unique, concatenate
from numpy.linalg import pinv
from scipy.signal import detrend
from math import ceil
import tables as tb

from Stats import calc_eofs, run_mean
import LIMTools as Lt
import DataTools as Dt


def _area_wgt(data, lats):
    """Apply area weighting to data based on provided latitude values."""
    assert(data.shape[-1] == lats.shape[-1])
    scale = sqrt(cos(radians(lats)))
    return data * scale


def _calc_m(x0, xt):
    """Calculate either L or G for forecasting"""
    
    # These represent the C(tau) and C(0) covariance matrices
    #    Note: x is an anomaly vector, no division by N-1 because it's undone
    #    in the inversion anyways
    
    x0x0 = dot(x0, x0.T)
    x0xt = dot(xt, x0.T)
    
    # Calculate tau-lag G value
    return dot(x0xt, pinv(x0x0))


def _create_h5_fcast_grps(h5f, dest, atom, shape, fcast_times):
    try:
        out_fcast = [h5f.create_carray(dest,
                                       'f%i' % lead,
                                       atom=atom,
                                       shape=shape,
                                       createparents=True,
                                       title='%i Year Forecast' % lead)
                     for lead in fcast_times]
    except tb.NodeError:
        for lead in fcast_times:
            h5f.remove_node('/'.join((dest, 'f%i' % lead)))
        out_fcast = [h5f.create_carray(dest,
                                       'f%i' % lead,
                                       atom=atom,
                                       shape=shape,
                                       createparents=True,
                                       title='%i Year Forecast' % lead)
                     for lead in fcast_times]
    return out_fcast


class LIM(object):
    """Linear inverse forecast model.
    
    This class uses a calibration dataset to make simple linear forecasts. 
    Can perform forecasts using random or contiguous resampling, or with
    separate calibration and forecast datasets.
    
    Notes
    -----
    It's based on the LIM described by M. Newman (2013) [1].  Right now it
    assumes the use of monthly data (i.e. each timestep should represent a
    single month).
    
    References
    ----------
    .. [1] Newman, M. (2013), An Empirical Benchmark for Decadal Forecasts of 
       Global Surface Temperature Anomalies, J. Clim., 26(14), 5260–5269, 
       doi:10.1175/JCLI-D-12-00590.1.
       
    Examples
    --------
    ....
    """
    
    def __init__(self, calibration, wsize, fcast_times, fcast_num_pcs,
                 area_wgt_lats=None, lons=None, h5file=None):
        """
        Parameters
        ----------
        calibration: ndarray
            Dataset for determining LIM forecast EOFs.  Provided data should be
            in a 2D MxN matrix where M (rows) represent temporal samples and
            N(columns) represent spatial samples.
        wsize: int
            Windowsize for running mean.  For this implementation it should
            be equal to a year's worth of samples
        fcast_times: array_like
            1D array-like object containing all times to forecast at with the
            LIM. Times should be in wsize units. i.e. 1yr forecast should
            be integer value "1" not 12 (if wsize=12).
        fcast_num_pcs: int
            Number of principal components to include in forecast calculations
        area_wgt_lats: ndarray, Optional
            Latitude vector pertaining to spatial dimension N.  If used area-
            weighting will be performed for pricipal component calculations.
            TODO: add ability to provide a simple non-tiled lat vector
        lons: ndarray, Optional
            Longitude vector flattened to spatial dimension N.
        H5file: HDF5_Object, Optional
            File object to store LIM output.  It will create a series of 
            directories under the given group
        """
        
        self._calibration = calibration
        self._wsize = wsize
        self.fcast_times = array(fcast_times, dtype=int16)
        self._neigs = fcast_num_pcs
        self._lats = area_wgt_lats
        self._lons = lons
        self._H5file = h5file
        self._obs_use = self._anomaly_srs = self._climo = None

    def forecast(self, t0_data, use_lag1=True, detrend_data=False,
                 use_h5=True):
        """Run LIM forecast from given data.
        
        Performs LIM forecast over the times specified by the
        fcast_times class attribute.  Forecast can be performed by calculating
        G for each time period or by L for a 1-year(or window_size) lag and
        then calculating each fcast_Time G from that L matrix.
        
        Parameters
        ----------
        t0_data: ndarray
            MxN array to forecast from.  M is the sample dimension, while N is 
            the spatial dimension.  1-window length chunk will be removed from
            each edge from the anomaly calculation procedure.  M^ = M - 2*wsize
        use_lag1: bool
            Flag for using only the G_1-matrix for forecasting
        detrend_data: bool
            Apply linear detrending to anomaly timeseries data
        use_h5: bool
            Use H5file to store forecast data instead of an ndarray.

            
        Returns
        -----
        fcast_out: ndarray-like
            LIM forecasts in a KxJxM^ matrix where K corresponds to each
            forecast time.
        eofs: ndarray-like
            EOFs for converting forecast output between EOF and physical
            space.  Returned in an NxJ matrix.

            
        Notes
        -----
        This method will set the fcast_out attribute for the LIM. If an HDF5
        obj is provided it will output the forecast to this file if desired.
        """

        # Calculate anomaly time series from the data
        self._anomaly_srs, _bedge, _tedge = run_mean(self._calibration,
                                                     self._wsize,
                                                     self._H5file,
                                                     shave_yr=True)
        self._anomaly_srs, self._climo = Lt.calc_anomaly(self._anomaly_srs,
                                                         self._wsize)
        
        # Calculate anomalies for initial data
        t0_data, _, _ = run_mean(t0_data, self._wsize, shave_yr=True)
        t0_data, _ = Lt.calc_anomaly(t0_data, self._wsize, self._climo)  # M^xN

        # Create output locations for our forecasts
        fcast_out_shp = [len(self.fcast_times), self._neigs, t0_data.shape[0]]
        if self._H5file is not None and use_h5:
            h5f = self._H5file
            # Create forecast groups
            fcast_out = _create_h5_fcast_grps(h5f,
                                              '/data/fcast_bin',
                                              tb.Atom.from_dtype(t0_data.dtype),
                                              fcast_out_shp[1:],
                                              self.fcast_times)
        else:
            fcast_out = zeros(fcast_out_shp)
        
        # Area Weighting if _lats is set
        if self._lats is not None:
            data = _area_wgt(self._anomaly_srs, self._lats)
        else:
            data = self._anomaly_srs

        # Detrend data if desired
        if detrend_data:
            data = detrend(data, axis=0, type='linear')
        
        # Calibrate the LIM with (J=neigs) EOFs from training data
        eofs, _, var_pct = calc_eofs(data.T, self._neigs)         # eofs is NxJ
        train_data = dot(eofs.T, self._anomaly_srs.T)             # JxM^
        
        # Project our testing data into eof space
        proj_t0_data = dot(eofs.T, t0_data.T)                     # JxM^
        
        # Forecasts using L to determine G-values
        if use_lag1:
            # Calculate L from time-lag of one window size (1-year for our LIM)
            tau = self._wsize  
            train_tdim = train_data.shape[1] - tau
            x0 = train_data[:, 0:train_tdim]
            xt = train_data[:, tau:(train_tdim+tau)]
            
            g_1 = _calc_m(x0, xt)
            for i, tau in enumerate(self.fcast_times):
                g = g_1**tau
                xf = dot(g, proj_t0_data)
                if use_h5:
                    fcast_out[i][:] = xf
                else:
                    fcast_out[i] = xf
        
        # Forecasts using G only    
        else:
            train_tdim = train_data.shape[1] - self.fcast_times[-1]*self._wsize
            x0 = train_data[:, 0:train_tdim]
            
            for i, tau in enumerate(self.fcast_times*self._wsize):
                xt = train_data[:, tau:(train_tdim+tau)]
                g = _calc_m(x0, xt)
                xf = dot(g, proj_t0_data)
                if use_h5:
                    fcast_out[i][:] = xf
                else:
                    fcast_out[i] = xf

        # Save EOFs to HDF5 file if needed
        if self._H5file is not None and use_h5:
            eof_out = Dt.var_to_hdf5_carray(h5f, '/data', 'eofs', eofs)
        else:
            eof_out = eofs

        return fcast_out, eof_out

    def save(self, h5file=None):
        """
        This function will attempt to save data from the LIM object into an
        HDF5 pytables file.  I am currently using pytables because they provide
        nice compression options, and the ability to perform out of core
        operations.

        :param h5file: new HDF5 file to store data
        :return: created HDF5 file
        """

        if h5file is not None:
            assert(type(h5file) == tb.File)

            if self._H5file is not None:
                self._H5file.close()
                self._H5file = h5file
            h5f = self._H5file
        else:
            assert(type(self._H5file) == tb.File)
            h5f = self._H5file

        try:
            data_grp = h5f.create_group(h5f.root, 'data',
                                        title='Observations & Forecast Data')
        except tb.NodeError:
            # group already exists
            data_grp = h5f.root.data

        Dt.var_to_hdf5_carray(h5f, data_grp, 'obs', self._calibration,
                              title='Observation Data')
        if self._anomaly_srs is not None:
            Dt.var_to_hdf5_carray(h5f, data_grp, 'anomaly_srs',
                                  self._anomaly_srs,
                                  title='Observation Anomaly Data')
            Dt.var_to_hdf5_carray(h5f, data_grp, 'climo', self._climo,
                                  title='Monthly climatology from Obs')
        Dt.var_to_hdf5_carray(h5f, data_grp, 'fcast_times', self.fcast_times,
                              title='Forecast Lead Times')
        Dt.var_to_hdf5_carray(h5f, data_grp, 'lats', self._lats,
                              title='Latitude Grid')
        if self._lons is not None:
            Dt.var_to_hdf5_carray(h5f, data_grp, 'lons', self._lons,
                                  title='Longitude Grid')


class ResampleLIM(LIM):

    def __init__(self, calibration, wsize, fcast_times, fcast_num_pcs,
                 hold_chk_pct, num_trials, area_wgt_lats=None, lons=None,
                 h5file=None):

        LIM.__init__(self, calibration, wsize, fcast_times, fcast_num_pcs,
                     area_wgt_lats, lons, h5file)

        # Need original input dataset for resampling
        self._original_obs = self._calibration # potential erase reference
        self._num_trials = num_trials

        # Initialize important indice limits for resampling procedure
        _fcast_tdim = self.fcast_times[-1]*wsize

        # 2*self._wsize is to account for edge removal from running mean
        _sample_tdim = self._original_obs.shape[0] - _fcast_tdim - 2*wsize
        hold_chk = int(ceil(_sample_tdim/self._wsize * hold_chk_pct))
        self._test_tdim = hold_chk * self._wsize
        _useable_tdim = (_sample_tdim - self._test_tdim)
        self._trials_out_shp = [len(self.fcast_times), self._num_trials,
                                self._neigs, self._test_tdim]
        self._test_start_idx = unique(linspace(0,
                                               _useable_tdim,
                                               self._num_trials
                                               ).astype(int16))

        # Calculate edge concatenation lengths for anomaly procedure
        obs_run_mean, bedge, tedge = run_mean(calibration,
                                              wsize,
                                              shave_yr=True)
        self._full_anomaly_srs, self._full_climo = \
            Lt.calc_anomaly(obs_run_mean, self._wsize)
        self._anom_edges = [bedge, tedge]

    def forecast(self, use_lag1=True, detrend_data=False):

        eof_shp = [self._num_trials,
                   self._original_obs.shape[1],
                   self._neigs]
        if self._H5file is not None:
            h5f = self._H5file

            fcast_atom = tb.Atom.from_dtype(self._original_obs.dtype)
            eof_atom = tb.Atom.from_dtype(self._original_obs.dtype)

            _fcast_out = _create_h5_fcast_grps(h5f,
                                               '/data/fcast_bin',
                                               fcast_atom,
                                               self._trials_out_shp[1:],
                                               self.fcast_times)
            _eofs_out = Dt.empty_hdf5_carray(h5f,
                                             '/data/',
                                             'eofs',
                                             eof_atom,
                                             eof_shp,
                                             title='Calculated EOFs by Trial')

        else:
            _fcast_out = zeros(self._trials_out_shp)
            _eofs_out = zeros(eof_shp)

        for j, trial in enumerate(self._test_start_idx):

            # beginning and end indices for test chunk
            bot_idx, top_idx = (self._anom_edges[0] + trial,
                                self._anom_edges[0] + trial + self._test_tdim)

            # create testing and training sets
            obs_dat = self._original_obs
            train_set = concatenate((obs_dat[0:bot_idx],
                                     obs_dat[top_idx:]),
                                    axis=0)
            test_set = obs_dat[(bot_idx - self._anom_edges[0]):
                               (top_idx + self._anom_edges[1])]
            self._calibration = train_set

            _fcast, _eofs = LIM.forecast(self,
                                         test_set,
                                         detrend_data=detrend_data,
                                         use_h5=False)

            # Place forecasts in the right trial, for the corresponding
            #   forecast bin.
            for i, fcast_bin in enumerate(_fcast_out):
                fcast_bin[j] = _fcast[i]
            _eofs_out[j] = _eofs

        return _fcast_out, _eofs_out

    def save(self, h5file=None):
        """
        This function will attempt to save data from the LIM object into an
        HDF5 pytables file.  I am currently using pytables because they provide
        nice compression options, and the ability to perform out of core
        operations.

        WARNING: If filename already exists it will be overwritten!

        :param h5file: new HDF5 file to store data
        :return: created HDF5 file
        """

        if h5file is not None:
            assert(type(h5file) == tb.File)

            if self._H5file is not None:
                self._H5file.close()
                self._H5file = h5file
            h5f = self._H5file
        else:
            assert(type(self._H5file) == tb.File)
            h5f = self._H5file

        try:
            data_grp = h5f.create_group(h5f.root, 'data',
                                        title='Observations & Forecast Data')
        except tb.NodeError:
            # group already exists
            data_grp = h5f.root.data

        Dt.var_to_hdf5_carray(h5f, data_grp, 'obs', self._original_obs,
                              title='Observation Data')
        Dt.var_to_hdf5_carray(h5f, data_grp, 'anomaly_srs',
                              self._full_anomaly_srs,
                              title='Observation Anomaly Data')
        Dt.var_to_hdf5_carray(h5f, data_grp, 'climo', self._full_climo,
                              title='Monthly climatology from Obs')
        Dt.var_to_hdf5_carray(h5f, data_grp, 'fcast_times', self.fcast_times,
                              title='Forecast Lead Times')
        Dt.var_to_hdf5_carray(h5f, data_grp, 'lats', self._lats,
                              title='Latitude Grid')
        if self._lons is not None:
            Dt.var_to_hdf5_carray(h5f, data_grp, 'lons', self._lons,
                                  title='Longitude Grid')
        Dt.var_to_hdf5_carray(h5f, data_grp, 'test_start_idxs',
                              self._test_start_idx,
                              title='Forecast Trial Starting Indices')
        data_grp._v_attrs.test_tdim = self._test_tdim
        data_grp._v_attrs.yrsize = self._wsize



# This class will be experimental at most.
# Have to make the assumption that anomaly uses entire sample average
class RandomResampleLIM(LIM):
    pass