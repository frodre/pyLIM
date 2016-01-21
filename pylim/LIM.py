# coding=utf-8
"""
Main Linear Inverse Model classes and methods.

Author: Andre Perkins
"""

import numpy as np
from numpy.linalg import pinv, matrix_rank, cond
from scipy.linalg import eig, inv
from math import ceil
import tables as tb
import cPickle as cpk

from Stats import calc_eofs
import DataTools as Dt


def _area_wgt(data, lats):
    """Apply area weighting to data based on provided latitude values."""
    assert(data.shape[-1] == lats.shape[-1])
    scale = np.sqrt(np.cos(np.radians(lats)))
    return data * scale


def _calc_m(x0, xt):
    """Calculate either L or G for forecasting (using nomenclature
    from Newman 2013"""
    
    # These represent the C(tau) and C(0) covariance matrices
    #    Note: x is an anomaly vector, no division by N-1 because it's undone
    #    in the inversion anyways
    
    x0x0 = np.dot(x0, x0.T)
    x0xt = np.dot(xt, x0.T)

    # print 'Matrix Ranks...
    # print 'C(0): ', matrix_rank(x0x0)
    # print 'C(1): ', matrix_rank(x0xt)
    # print 'inv(C(0)) ', matrix_rank(pinv(x0x0))
    # print 'C(1)*C(0)^-1: ', matrix_rank(dot(x0xt, pinv(x0x0)))

    # print 'Condition number'
    # print 'C(0): ', cond(x0x0)
    
    # Calculate tau-lag G value
    return np.dot(x0xt, pinv(x0x0))


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

    def __init__(self, calib_data_obj, wsize, fcast_times, fcast_num_pcs,
                 detrend_data=False, h5file=None, L_eig_bump=None):
        """
        Parameters
        ----------
        calib_data_object: DataTools.BaseDataObject or subclass
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
        fcast_num_pcs: int
            Number of principal components to include in forecast calculations
        H5file: HDF5_Object, Optional
            File object to store LIM output.  It will create a series of
            directories under the given group
        """
        assert isinstance(calib_data_obj, Dt.BaseDataObject), \
            'calib_data_obj must be an instance of BaseDataObject or subclass.'

        self._data_obj = calib_data_obj
        self._from_precalib = False
        self._calibration = None
        self._wsize = wsize
        self.fcast_times = np.array(fcast_times, dtype=np.int16)
        self._neigs = fcast_num_pcs
        self._h5file = h5file
        self._bedge = None
        self._tedge = None
        self._eofs = None
        self._climo = None
        self._detrend_data = detrend_data
        self.G_1 = None
        self.eig_bump = L_eig_bump

        self.set_calibration()

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
        else:
            self._bedge = 0
            # TODO: set _tedge to something reasonable

        if not data_obj.is_anomaly:
            data_obj.calc_anomaly(self._wsize)
            self._climo = data_obj.climo[:]

        if not data_obj.is_area_weighted:
            data_obj.area_weight_data(save=False)

        if self._detrend_data and not data_obj.is_detrended:
            data_obj.detrend_data(save=False)

        self._calibration = data_obj.data
        self._eofs, _ = calc_eofs(self._calibration, self._neigs)

        train_data = np.dot(self._eofs.T, self._calibration[:].T)
        tdim = train_data.shape[1] - self._wsize
        x0 = train_data[:, 0:tdim]
        x1 = train_data[:, self._wsize:]
        self.G_1 = _calc_m(x0, x1)

        if self.eig_bump is not None:
            self.G_1 = self.eig_adjust(self.G_1)

    def eig_adjust(self, G):

        evals, evecs = eig(G)
        evals = np.log(evals)
        evals += self.eig_bump
        evals = np.exp(evals)

        # E * lambda * E^-1
        return np.dot(evecs, np.dot(evals, inv(evecs))).real

    def save_precalib(self, filename):

        tmp_calib = self._calibration
        tmp_dobj = self._data_obj

        self._calibration = None
        self._data_obj = None

        with open(filename, 'w') as f:
            cpk.dump(self, f)

        print 'Saved pre-calibrated LIM to {}'.format(filename)

        self._calibration = tmp_calib
        self._data_obj = tmp_dobj

    @staticmethod
    def from_precalib(filename):
        with open(filename, 'r') as f:
            obj = cpk.load(f)

        obj._from_precalib = True
        return obj


    def forecast(self, t0_data, use_lag1=True, use_h5=True):
        """Run LIM forecast from given data.
        
        Performs LIM forecast over the times specified by the
        fcast_times class attribute.  Forecast can be performed by calculating
        G for each time period or by L for a 1-year(or window_size) lag and
        then calculating each fcast_Time G from that L matrix.
        
        Parameters
        ----------
        t0_data: DataTools.BaseDataObject or subclass
            Data to forecast from.  Expects leading sample dimension with a
            flattened spatial dimension.  1-window length chunk is removed from
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

        assert isinstance(t0_data, Dt.BaseDataObject), \
            't0_data must be an instance of BaseDataObject or subclass.'
        assert t0_data.forced_flat or t0_data.is_masked, \
            't0_data expects flattened spatial dimension'
        assert t0_data._leading_time, \
            't0_data expects a leading sampling dimension'

        if self._from_precalib and not use_lag1:
            print ('LIM class created from pre calibrated file. '
                   'Switching use_lag1 to True due to no _calibration data.')
            use_lag1 = True

        # Calculate anomalies for initial data
        if not t0_data.is_run_mean:
            t0_data.calc_running_mean(self._wsize, shave_yr=True, save=False)
        if not t0_data.is_anomaly:
            t0_data.calc_anomaly(self._wsize, save=False, climo=self._climo)

        if self._detrend_data and not t0_data.is_detrended:
            t0_data.detrend_data(save=False)

        forecast_data = t0_data.data

        # Create output locations for our forecasts
        fcast_out_shp = [len(self.fcast_times), self._neigs,
                         forecast_data.shape[0]]

        if self._h5file is not None and use_h5:
            h5f = self._h5file
            # Create forecast groups
            fcast_out = _create_h5_fcast_grps(h5f,
                                              '/data/fcast_bin',
                                              tb.Atom.from_dtype(
                                                  t0_data.data_dtype),
                                              fcast_out_shp[1:],
                                              self.fcast_times)
        else:
            fcast_out = np.zeros(fcast_out_shp)

        # Calibrate the LIM with (J=neigs) EOFs from training data
        eofs = self._eofs     # eofs is NxJ

        # Project our testing data into eof space
        proj_t0_data = np.dot(eofs.T, forecast_data[:].T)              # JxM^

        # Forecasts using L to determine G-values
        if use_lag1:
            # Calculate L from time-lag of one window size (1-year for our LIM)

            g_1 = self.G_1
            for i, tau in enumerate(self.fcast_times):
                g = g_1**tau
                xf = np.dot(g, proj_t0_data)
                if use_h5:
                    fcast_out[i][:] = xf
                else:
                    fcast_out[i] = xf

        # Forecasts using G only    
        else:
            # Training data has to allow for lag of max forecast time
            train_data = np.dot(eofs.T, self._calibration[:].T)    # JxM^
            train_tdim = train_data.shape[1] - self.fcast_times[-1]*self._wsize
            x0 = train_data[:, 0:train_tdim]

            for i, tau in enumerate(self.fcast_times*self._wsize):
                xt = train_data[:, tau:(train_tdim+tau)]
                g = _calc_m(x0, xt)

                if self.eig_bump is not None:
                    g = self.eig_adjust(g)

                xf = np.dot(g, proj_t0_data)
                if use_h5:
                    fcast_out[i][:] = xf
                else:
                    fcast_out[i] = xf

        # Save EOFs to HDF5 file if needed
        if self._h5file is not None and use_h5:
            eof_out = Dt.var_to_hdf5_carray(h5f, '/data', 'eofs', eofs)
        else:
            eof_out = eofs

        return fcast_out, eof_out


class ResampleLIM(LIM):
    """
    Linear Inverse Model Forecasts using resampling experiments.  This
    will take in a single dataset and withold a certain portion during
    calibration.  Repeated trials are performed using the withheld data
    to forecast on.

    See the LIM Class docstring for references.
    """

    def __init__(self, calib_data_object, wsize, fcast_times, fcast_num_pcs,
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
        fcast_num_pcs: int
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

        LIM.__init__(self, calib_data_object, wsize, fcast_times, fcast_num_pcs,
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
                force_flat=True, save_none=True)
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
