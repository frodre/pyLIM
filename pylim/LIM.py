# coding=utf-8
"""
Main Linear Inverse Model classes and methods.

Author: Andre Perkins
"""

import numpy as np
from numpy.linalg import pinv, cond, eigvals
from math import ceil
import tables as tb
import cPickle as cpk
import logging
import time

from Stats import calc_eofs
import DataTools as Dt


logger = logging.getLogger(__name__)


def _calc_m(x0, xt, tau=1):
    """Calculate either L or G for forecasting (using nomenclature
    from Newman 2013"""
    
    # These represent the C(tau) and C(0) covariance matrices
    #    Note: x is an anomaly vector, no division by N-1 because it's undone
    #    in the inversion anyways
    
    x0x0 = np.dot(x0, x0.T)
    x0xt = np.dot(xt, x0.T)

    # Calculate the mapping term G_tau
    G = np.dot(x0xt, pinv(x0x0))

    # Calculate the forcing matrix to check that all modes are damped
    L = (1/tau) * np.log(G)
    Leigs = eigvals(L)

    if np.any(Leigs.real >= 0):
        logger.debug('L eigenvalues: \n' + str(Leigs))
        raise ValueError('Positive eigenvalues detected in forecast matrix L.')

    if cond(x0x0) > 20:
        logger.warn(('C_0 condition number is large ({:2.2f}). Too many '
                     'features (or EOFs) are likely provided and further '
                     'truncation is suggested.'))

    return G


def _create_h5_fcast_grps(h5f, dest, atom, shape, fcast_times):
    """
    Helper method for creating forecast groups in the hdf5 file.
    """

    out_fcast = []

    # if not h5f.__contains__(dest):
    #     grp, node = path.split(dest)
    #     h5f.create_group(grp, node)

    for lead in fcast_times:
        out_fcast.append(
            Dt.empty_hdf5_carray(h5f,
                                 dest,
                                 'f{:d}'.format(lead),
                                 atom,
                                 shape,
                                 createparents=True,
                                 title='{:d} Year Forecast'.format(lead)))

    return out_fcast


class LIM(object):
    """Linear inverse forecast model.
    
    This class uses a calibration dataset to make simple linear forecasts.
    
    Notes
    -----
    Based on the LIM described by M. Newman (2013) [1].  Right now it
    assumes the use of monthly data (i.e. each timestep should represent a
    single month).
    
    References
    ----------
    .. [1] Newman, M. (2013), An Empirical Benchmark for Decadal Forecasts of 
       Global Surface Temperature Anomalies, J. Clim., 26(14), 5260–5269, 
       doi:10.1175/JCLI-D-12-00590.1.
    ....
    """

    def __init__(self, calib_data, nelem_in_tau1, h5file=None):
        """
        Parameters
        ----------
        calib_data: ndarray
            Data for calibrating the LIM.  Expects
            a 2D MxN matrix where M (rows) represent the sampling dimension and
            N(columns) represents the feature dimension (e.g. spatial grid
            points).
        nelem_in_tau1: int
            Number of time samples that span tau=1.  E.g. for monthly data when
            a forecast tau is equivalent to 1 year, nelem_in_tau should be 12.
        h5file: HDF5_Object, Optional
            File object to store LIM output.  It will create a series of
            directories under the given group
        """
        logger.info('Initializing LIM forecasting object...')

        if calib_data.ndim != 2:
            logger.error(('LIM calibration data is not 2D '
                          '(Contained ndim={:d}').format(calib_data.ndim))
            raise ValueError('Input LIM calibration data is not 2D')

        self._h5file = h5file
        self._from_precalib = False
        self._nelem_in_tau1 = nelem_in_tau1
        self._eof_var_stats = {}

        x0 = calib_data[:, 0:-nelem_in_tau1]
        x1 = calib_data[:, nelem_in_tau1:]

        self.G_1 = _calc_m(x0, x1, tau=1)

    def save_precalib(self, filename):

        with open(filename, 'w') as f:
            cpk.dump(self, f)

        print 'Saved pre-calibrated LIM to {}'.format(filename)

    @staticmethod
    def from_precalib(filename):
        with open(filename, 'r') as f:
            obj = cpk.load(f)

        obj._from_precalib = True
        return obj

    def forecast(self, t0_data, fcast_leads, use_h5=True):
        """Forecast on provided data.
        
        Performs LIM forecast over the times specified by the
        fcast_times class attribute.  Forecast can be performed by calculating
        G for each time period or by L for a 1-year(or window_size) lag and
        then calculating each fcast_Time G from that L matrix.
        
        Parameters
        ----------
        t0_data: ndarray
            Data to forecast from.  Expects
            a 2D MxN matrix where M (rows) represent the sampling dimension and
            N(columns) represents the feature dimension (e.g. spatial grid
            points).
        fcast_leads: List<int>
            A list of forecast lead times.  Each value is interpreted as a
            tau value, for which the forecast matrix is determined as G_1^tau.
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

        """

        if t0_data.ndim != 2:
            logger.error(('LIM forecast data is not 2D '
                          '(Contained ndim={:d}').format(t0_data.ndim))
            raise ValueError('Input LIM forecast data is not 2D')

        logger.info('Performing LIM forecast for tau values: '
                    + str(fcast_leads))

        num_fcast_times = len(fcast_leads)

        # Create output locations for our forecasts
        fcast_out_shp = (num_fcast_times, t0_data.shape[0], t0_data.shape[1])

        if self._h5file is not None and use_h5:
            # Create forecast groups
            fcast_out = _create_h5_fcast_grps(self._h5file,
                                              '/data/fcast_bin',
                                              tb.Atom.from_dtype(
                                                  t0_data.data_dtype),
                                              fcast_out_shp[1:],
                                              fcast_leads)
        else:
            fcast_out = np.zeros(fcast_out_shp)

        for i, tau in enumerate(fcast_leads):
            g = self.G_1**tau
            xf = np.dot(g, t0_data.T)
            if use_h5:
                fcast_out[i][:] = xf.T
            else:
                fcast_out[i] = xf.T

        return fcast_out


class ParamReducedLIM(LIM):
    """Linear inverse forecast model.
    
    This class uses a calibration dataset to make simple linear forecasts.
    Calibration data input into this class will be reduced to the specified
    number of EOFs.
    
    Notes
    -----
    Based on the LIM described by M. Newman (2013) [1].  Right now it
    assumes the use of monthly data (i.e. each timestep should represent a
    single month).
    
    References
    ----------
    .. [1] Newman, M. (2013), An Empirical Benchmark for Decadal Forecasts of 
       Global Surface Temperature Anomalies, J. Clim., 26(14), 5260–5269, 
       doi:10.1175/JCLI-D-12-00590.1.
    ....
    """

    def __init__(self, calib_data, nelem_in_tau1, num_eofs, h5file=None):
        """
        Parameters
        ----------
        calib_data: ndarray
            Data for calibrating the LIM.  Expects
            a 2D MxN matrix where M (rows) represent the sampling dimension and
            N(columns) represents the feature dimension (e.g. spatial grid
            points).
        num_eofs: int
            Number of principal components to include in forecast calculations
        nelem_in_tau1: int
            Number of time samples that span tau=1.  E.g. for monthly data when
            a forecast tau is equivalent to 1 year, nelem_in_tau should be 12.
        h5file: HDF5_Object, Optional
            File object to store LIM output.  It will create a series of
            directories under the given group
        """

        self._neigs = num_eofs

        logger.info('Calculating EOFs on calibration data.')
        stime = time.time()
        self._eofs = calc_eofs(calib_data, num_eofs,
                               var_stats_dict=self._eof_var_stats)
        eof_calib_data = np.dot(self._eofs.T, calib_data.T)
        etime = time.time() - stime
        logger.info('EOF truncation finished in {:2.2f}s'.format(etime))

        if self._eof_var_stats is not None:
            logger.debug(('\nEOF statistics:\n'
                          'Total variance: {:2.4e}\n'
                          'Var explained (ret modes): {:2.4e}'
                          ).format(self._eof_var_stats['total_var'],
                                   self._eof_var_stats['var_expl_by_ret']))

        super(ParamReducedLIM, self).__init__(eof_calib_data, nelem_in_tau1,
                                              h5file=h5file)

        if h5file is not None:
            Dt.var_to_hdf5_carray(h5file, '/data', 'eofs', self._eofs)

    def forecast(self, t0_data, fcast_leads, use_h5=True):
        """Forecast on provided data.
        
        Performs LIM forecast over the times specified by the
        fcast_times class attribute.  Forecast is performed using G based 
        on tau=1 lag covariances.
        
        t0_data is projected into EOF space based on the calibration data EOFs
        
        Parameters
        ----------
        t0_data: ndarray
            Data to forecast from.  Expects
            a 2D MxN matrix where M (rows) represent the sampling dimension and
            N(columns) represents the feature dimension (e.g. spatial grid
            points).
        fcast_leads: List<int>
            A list of forecast lead times.  Each value is interpreted as a
            tau value, for which the forecast matrix is determined as G_1^tau.
        use_h5: bool
            Use H5file to store forecast data instead of an ndarray.

        Returns
        -----
        fcast_out: ndarray-like
            LIM forecasts in a KxMxN matrix where K corresponds to each
            forecast time.
        """

        # Project our testing data into eof space
        proj_t0_data = np.dot(t0_data[:], self._eofs)   # MxJ  where J=num_eofs

        fcasts = super(ParamReducedLIM, self).forecast(proj_t0_data,
                                                       fcast_leads,
                                                       use_h5=use_h5)

        # forecasts are KxMxJ, expand back into physical space using JxN EOFs
        expanded_fcasts = [np.dot(fcast_lead, self._eofs)
                           for fcast_lead in fcasts]
        return expanded_fcasts


class ResampleLIM(LIM):
    """
    Linear Inverse Model Forecasts using resampling experiments.  This
    will take in a single dataset and withold a certain portion during
    calibration.  Repeated trials are performed using the withheld data
    to forecast on.

    See the LIM Class docstring for references.
    """

    def __init__(self, calib_data_object, wsize, fcast_times, fcast_components,
                 hold_chk_pct, num_trials, detrend_data=False, h5file=None):
        """
        Parameters
        ----------
        calib_data_object: DataTools.BaseDataObject or subclass
            Dataset for determining LIM forecast EOFs.  DataInput provids
            a 2D MxN matrix where M (rows) represent temporal samples and
            N(columns) represent spatial samples.  It handles data with
            nan masking.  Note: If data is masked, saved output spatial
            dimensions will be reduced to valid data.
        data_in: DataTools.DataInput
            Dataset for determining LIM forecast EOFs.  DataInput provids
            a 2D MxN matrix where M (rows) represent temporal samples and
            N(columns) represent spatial samples.  It handles data with
            nan masking.  Note: If data is masked, saved output spatial
            dimensions will be reduced to valid data.
        wsize: int
            Windowsize for running mean.  For this implementation it should
            be equal to a year's worth of samples
        fcast_times: array_like
            1D array-like object containing all times to forecast at with the
            LIM. Times should be in wsize units. i.e. 1yr forecast should
            be integer value "1" not 12 (if wsize=12).
        fcast_components: int
            Number of principal components to include in forecast calculations
        hold_chk_pc: float
            The percentage of data to withhold during each trial of the
            resampling experiment.
        num_trials: int
            Number of trials to run during the resampling experiment.
            Note: Consider the windowsize when you determine the number of
            trials.  A large number of trials may result in significant
            overlaps between trials and skew your statistics due to
            repeated sampling of middle data.
        H5file: HDF5_Object, Optional
            File object to store LIM output.  It will create a series of
            directories under the given group
        """

        raise NotImplementedError('Updates to the base LIM class have broken '
                                  'this.')

        self._orig_is_run_mean = calib_data_object.is_run_mean
        self._orig_is_anomaly = calib_data_object.is_anomaly

        LIM.__init__(self, calib_data_object, wsize, fcast_times, fcast_components,
                     detrend_data=detrend_data, h5file=h5file)

        # Need original input dataset for resampling
        self._original_obs = calib_data_object.reset_data('orig')
        self._num_trials = num_trials

        # Initialize important indice limits for resampling procedure
        _fcast_tdim = self.fcast_times[-1]*wsize
        self._fcast_tdim = _fcast_tdim

        # 2*self._wsize is to account for edge removal from running mean
        _sample_tdim = self._data_obj._full_shp[0] - _fcast_tdim - 2*wsize
        hold_chk = int(ceil(_sample_tdim/self._wsize * hold_chk_pct))
        self._test_tdim = hold_chk * self._wsize
        _useable_tdim = (_sample_tdim - self._test_tdim)
        self._trials_out_shp = [len(self.fcast_times), self._num_trials,
                                self._neigs, self._test_tdim]
        self._test_start_idx = np.unique(np.linspace(0,
                                               _useable_tdim,
                                               self._num_trials
                                               ).astype(np.int16))

        # Calculate edge concatenation lengths for anomaly procedure
        self._anom_edges = [self._bedge, self._tedge]

    # TODO: Placed here to aid in figuring out resampled LIM
    # def set_calibration(self, data_obj=None):
    #     if data_obj is not None:
    #         assert isinstance(data_obj, Dt.BaseDataObject), \
    #             'data_obj must be an instance of BaseDataObject or'\
    #             'its subclass.'
    #     else:
    #         data_obj = self._data_obj
    #
    #     assert data_obj.forced_flat or data_obj.is_masked, \
    #         'data_obj expects flattened spatial dimension'
    #     assert data_obj._leading_time, \
    #         'data_obj expects a leading sampling dimension'
    #
    #     if not data_obj.is_run_mean:
    #         _, self._bedge, self._tedge = data_obj.calc_running_mean(
    #             self._wsize, save=False,  shave_yr=True)
    #     else:
    #         self._bedge = 0
    #         # TODO: set _tedge to something reasonable
    #
    #     if not data_obj.is_anomaly:
    #         data_obj.calc_anomaly(self._wsize)
    #         self._climo = data_obj.climo[:]
    #
    #     if not data_obj.is_area_weighted:
    #         data_obj.area_weight_data(save=False)
    #
    #     if self._detrend_data and not data_obj.is_detrended:
    #         data_obj.detrend_data(save=False)
    #
    #     self._calibration = data_obj.data
    #     self._eof_var_stats = {}
    #     self._eofs, _ = calc_eofs(self._calibration, self._neigs,
    #                               var_stats_dict=self._eof_var_stats)
    #
    #     train_data = np.dot(self._eofs.T, self._calibration[:].T)
    #     tdim = train_data.shape[1] - self._wsize
    #     x0 = train_data[:, 0:tdim]
    #     x1 = train_data[:, self._wsize:]
    #     self.G_1 = _calc_m(x0, x1)

    def forecast(self, use_lag1=True):
        """Run LIM forecast using resampling

        Performs LIM forecast over the times specified by the
        fcast_times class attribute.  Forecast can be performed by calculating
        G for each time period or by L for a 1-year(or window_size) lag and
        then calculating each fcast_Time G from that L matrix.

        Resampling is determined by hold_chk_pct and num_trials.

        Parameters
        ----------
        use_lag1: bool
            Flag for using only the G_1-matrix for forecasting
        detrend_data: bool
            Apply linear detrending to anomaly timeseries data
        use_h5: bool
            Use H5file to store forecast data instead of an ndarray.


        Returns
        -----
        _fcast_out: ndarray-like
            LIM forecasts in a KxTxJxM^ matrix where K corresponds to each
            forecast time and T corresponds to each trial.
        _eofs_out: ndarray-like
            EOFs for converting forecast output between EOF and physical
            space.  Returned in an TxNxJ matrix.


        Notes
        -----
        This method will set the fcast_out attribute for the LIM. If an HDF5
        obj is provided it will output the forecast to this file if desired.
        """

        print 'Beginning resampling forecast experiment.'

        eof_shp = [self._num_trials,
                   self._data_obj.data.shape[1],
                   self._neigs]

        if self._h5file is not None:
            h5f = self._h5file

            fcast_atom = tb.Atom.from_dtype(self._data_obj.data_dtype)
            eof_atom = tb.Atom.from_dtype(self._data_obj.data_dtype)

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
            _fcast_out = np.zeros(self._trials_out_shp)
            _eofs_out = np.zeros(eof_shp)

        for j, trial in enumerate(self._test_start_idx):

            # beginning and end indices for test chunk
            bot_idx, top_idx = (self._anom_edges[0] + trial,
                                self._anom_edges[0] + trial + self._test_tdim)

            # create testing and training sets
            obs_dat = self._original_obs
            if self._detrend_data:
                anom_dat = self._data_obj.detrended
            else:
                anom_dat = self._data_obj.anomaly
            time_key = self._data_obj.TIME
            lat_key = self._data_obj.LAT
            dim_coords = self._data_obj.get_dim_coords([time_key])
            time_coords = dim_coords[time_key][1]
            lat_dim_coords = \
                (1, self._data_obj.get_coordinate_grids([lat_key])[lat_key])

            train_set = np.concatenate((obs_dat[0:bot_idx],
                                       obs_dat[top_idx:]),
                                       axis=0)
            train_times = np.concatenate((time_coords[0:bot_idx],
                                         time_coords[top_idx:]),
                                         axis=0)
            # test_set = obs_dat[(bot_idx - self._anom_edges[0]):
            #                    (top_idx + self._anom_edges[1])]
            test_set = anom_dat[trial:(trial+self._test_tdim)]

            # test_times = time_coords[(bot_idx - self._anom_edges[0]):
            #                          (top_idx + self._anom_edges[1])]
            test_times = time_coords[bot_idx:top_idx]

            train_dim_coords = {time_key: (0, train_times),
                                lat_key: lat_dim_coords}
            resample_dat_obj = Dt.BaseDataObject(
                train_set, dim_coords=train_dim_coords,
                force_flat=True, save_none=True,
                is_anomaly=self._orig_is_anomaly,
                is_run_mean=self._orig_is_run_mean)
            # use LIM calibration to calculate EOFs
            LIM.set_calibration(self, data_obj=resample_dat_obj)

            test_dim_coords = {time_key: (0, test_times),
                               lat_key: lat_dim_coords}
            forecast_obj = Dt.BaseDataObject(
                test_set,
                dim_coords=test_dim_coords,
                force_flat=True,
                save_none=True,
                is_run_mean=True,
                is_anomaly=True,
                is_detrended=self._detrend_data)

            _fcast, _eofs = LIM.forecast(self,
                                         forecast_obj,
                                         use_h5=False)

            del resample_dat_obj
            del forecast_obj

            # Place forecasts in the right trial, for the corresponding
            #   forecast bin.
            for i, fcast_bin in enumerate(_fcast_out):
                fcast_bin[j] = _fcast[i]
            _eofs_out[j] = _eofs

            print 'Trial {} finished.'.format(j+1)

        return _fcast_out, _eofs_out

    def set_calibration(self, data_obj=None):
        if data_obj is not None:
            assert isinstance(data_obj, Dt.BaseDataObject), \
                'data_obj must be an instance of BaseDataObject or'\
                'its subclass.'
        else:
            data_obj = self._data_obj

        assert data_obj.forced_flat or data_obj.is_masked, \
            'data_obj expects flattened spatial dimension'
        assert data_obj._leading_time, \
            'data_obj expects a leading sampling dimension'

        if not data_obj.is_run_mean:
            _, self._bedge, self._tedge = data_obj.calc_running_mean(
                self._wsize, save=False,  shave_yr=True)

        if not data_obj.is_anomaly:
            # Saved if we aren't detrending
            data_obj.calc_anomaly(self._wsize, save=(not self._detrend_data))

        if not data_obj.is_detrended and self._detrend_data:
            data_obj.detrend_data()

        self._calibration = data_obj.data

    def save_attrs(self):
        h5f = self._h5file
        data_node = h5f.get_node('/data')
        data_node._v_attrs.yrsize = self._wsize
        data_node._v_attrs.test_start_idxs = self._test_start_idx
        data_node._v_attrs.fcast_times = self.fcast_times
        data_node._v_attrs.test_tdim = self._test_tdim

        coords = self._data_obj.get_dim_coords([self._data_obj.LAT,
                                                self._data_obj.LON])

        Dt.var_to_hdf5_carray(h5f, data_node, 'lat',
                              coords[self._data_obj.LAT][1])
        Dt.var_to_hdf5_carray(h5f, data_node, 'lon',
                              coords[self._data_obj.LON][1])
