# coding=utf-8
"""
Main Linear Inverse Model classes and methods.

Author: Andre Perkins
"""

import numpy as np
from numpy.linalg import inv, eigvals, eig, eigh
from math import ceil
import tables as tb
import pickle as cpk
import logging
import time

from .Stats import calc_eofs
from . import DataTools as Dt


logger = logging.getLogger(__name__)


def _calc_m(x0, xt, tau=1):
    """Calculate either L or G for forecasting (using nomenclature
    from Newman 2013
    
    Parameters
    ----------
    x0: ndarray
        State at time=0.  MxN where M is number of samples, and N is the 
        number of features.
    xt: ndarray
        State at time=tau.  MxN where M is number of samples, and N is the 
        number of fatures.
    tau: float
        lag time (in units of tau) that we are calculating G for.  This is 
        used to check that all modes of L are damped.
        
        
    """
    
    # These represent the C(tau) and C(0) covariance matrices
    #    Note: x is an anomaly vector, no division by N-1 because it's undone
    #    in the inversion anyways
    
    x0x0 = np.dot(x0.T, x0)
    x0xt = np.dot(xt.T, x0)

    # Calculate the mapping term G_tau
    G = np.dot(x0xt, inv(x0x0))

    # Calculate the forcing matrix to check that all modes are damped
    Geigs = eigvals(G)
    Leigs = (1./tau) * np.log(Geigs)

    if np.any(Leigs.real >= 0):
        logger.debug('L eigenvalues: \n' + str(Leigs))
        raise ValueError('Positive eigenvalues detected in forecast matrix L.')

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

    def __init__(self, tau0_data, tau1_data=None, nelem_in_tau1=1,
                 fit_noise=False, max_neg_Qeval=5, h5file=None):
        """
        Parameters
        ----------
        tau0_data: ndarray
            Data for calibrating the LIM.  Expects
            a 2D MxN matrix where M (rows) represent the sampling dimension and
            N(columns) represents the feature dimension (e.g. spatial grid
            points).
        tau1_data: ndarray, optional
            Data with lag of tau=1.  Used to calculate the mapping term, G1,
            going from tau0 to tau1.  Must be the same shape as tau0_data.  If
            not provided, tau0_data is assumed to be sequential and
            nelem_in_tau1 and tau0_data is used to calculate lag covariance.
        nelem_in_tau1: int, optional
            Number of time samples that span tau=1.  E.g. for monthly data when
            a forecast tau is equivalent to 1 year, nelem_in_tau should be 12.
            Used if tau1_data is not provided.
        fit_noise: bool, optional
            Whether to fit the noise term from calibration data. Used for
            noise integration
        max_neg_Qeval: int, optional
            The maximum number of allowed negative eigenvalues in the Q matrix.
            Negative eigenvalues suggest inclusion of too many modes, but a few
            spurious ones are common.  I've been using a max of 5, but this
            isn't based on an objective knowledge.
        h5file: HDF5_Object, Optional
            File object to store LIM output.  It will create a series of
            directories under the given group
        """
        logger.info('Initializing LIM forecasting object...')

        if tau0_data.ndim != 2:
            logger.error(('LIM calibration data is not 2D '
                          '(Contained ndim={:d}').format(tau0_data.ndim))
            raise ValueError('Input LIM calibration data is not 2D')

        self._h5file = h5file
        self._from_precalib = False
        self._nelem_in_tau1 = nelem_in_tau1
        self._eof_var_stats = {}

        if tau1_data is not None:
            if not tau1_data.shape == tau0_data.shape:
                logger.error('LIM calibration data shape mismatch. tau1: {}'
                             ' tau0: {}'.format(tau1_data.shape,
                                                tau0_data.shape))
                raise ValueError('Tau1 and Tau0 calibration data shape '
                                 'mismatch')

            x0 = tau0_data
            x1 = tau1_data
        else:
            x0 = tau0_data[0:-nelem_in_tau1, :]
            x1 = tau0_data[nelem_in_tau1:, :]

        self.G_1 = self._calc_m(x0, x1, tau=1)

        if fit_noise:
            q_res = self._calc_Q(self.G_1, x0, tau=1,
                                 max_neg_evals=max_neg_Qeval)
            [self.L,
             self.Q_evals,
             self.Q_evects,
             self.num_neg_Q,
             self.neg_Q_rescale_factor] = q_res
        else:
            self.L = None
            self.Q_evals = None
            self.Q_evects = None
            self.num_neg_Q = None
            self.neg_Q_rescale_factor = None

    @staticmethod
    def _calc_m(x0, xt, tau=1):
        """Calculate either L or G for forecasting (using nomenclature
        from Newman 2013

        Parameters
        ----------
        x0: ndarray
            State at time=0.  MxN where M is number of samples, and N is the 
            number of features.
        xt: ndarray
            State at time=tau.  MxN where M is number of samples, and N is the 
            number of fatures.
        tau: float
            lag time (in units of tau) that we are calculating G for.  This is 
            used to check that all modes of L are damped.


        """

        # These represent the C(tau) and C(0) covariance matrices
        #    Note: x is an anomaly vector, no division by N-1 because it's undone
        #    in the inversion anyways

        # Division by number of samples ignored due to inverse
        x0x0 = np.dot(x0.T, x0)
        x0xt = np.dot(xt.T, x0)

        # Calculate the mapping term G_tau
        G = np.dot(x0xt, inv(x0x0))

        # Calculate the forcing matrix to check that all modes are damped
        Geigs = eigvals(G)
        Leigs = (1. / tau) * np.log(Geigs)

        if np.any(Leigs.real >= 0):
            logger.debug('L eigenvalues: \n' + str(Leigs))
            raise ValueError(
                'Positive eigenvalues detected in forecast matrix L.')

        return G

    @staticmethod
    def _calc_Q(G, x0, tau=1, max_neg_evals=5):
        C0 = x0.T @ x0 / (x0.shape[0] - 1)  # State covariance
        G_eval, G_evects = eig(G)
        L_evals = (1/tau) * np.log(G_eval)
        L = G_evects @ np.diag(L_evals) @ inv(G_evects)
        L = np.matrix(L)
        # L = L.real
        Q = -(L @ C0 + C0 @ L.H)  # Noise covariance

        # Check if Q is Hermetian
        is_adj = abs(Q - Q.H)
        tol = 1e-10
        if np.any(abs(is_adj) > tol):
            raise ValueError('Determined Q is not Hermetian (complex '
                             'conjugate transpose is equivalent.)')

        q_evals, q_evects = eigh(Q)
        sort_idx = q_evals.argsort()
        q_evals = q_evals[sort_idx][::-1]
        q_evects = q_evects[:, sort_idx][:, ::-1]
        num_neg = (q_evals < 0).sum()

        if num_neg > 0:
            num_left = len(q_evals) - num_neg
            if num_neg > max_neg_evals:
                logger.debug('Found {:d} modes with negative eigenvalues in'
                             ' the noise covariance term, Q.'.format(num_neg))
                raise ValueError('More than {:d} negative eigenvalues of Q '
                                 'detected.  Consider further dimensional '
                                 'reduction.'.format(max_neg_evals))

            else:
                logger.info('Removing negative eigenvalues and rescaling {:d} '
                            'remaining eigenvalues of Q.'.format(num_left))
                pos_q_evals = q_evals[q_evals > 0]
                scale_factor = q_evals.sum() / pos_q_evals.sum()
                logger.info('Q eigenvalue rescaling: {:1.2f}'.format(scale_factor))

                q_evals = q_evals[:-num_neg]*scale_factor
                q_evects = q_evects[:, :-num_neg]
        else:
            scale_factor = None

        # Change back to arrays
        L = np.array(L)
        q_evects = np.array(q_evects)

        return L, q_evals, q_evects, num_neg, scale_factor

    def save_precalib(self, filename):

        with open(filename, 'w') as f:
            cpk.dump(self, f)

        print('Saved pre-calibrated LIM to {}'.format(filename))

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
            g = np.linalg.matrix_power(self.G_1, tau)
            xf = np.dot(g, t0_data.T)
            if use_h5:
                fcast_out[i][:] = xf.T
            else:
                fcast_out[i] = xf.T

        return fcast_out

    def noise_integration(self, t0_data, length, timesteps=720,
                          out_arr=None, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # t0_data comes in as sample x spatial
        L = self.L
        Q_eval = self.Q_evals[:, None]
        Q_evec = self.Q_evects
        tdelta = 1/timesteps
        integration_steps = int(2*timesteps * length)
        num_evals = Q_eval.shape[0]
        nens = t0_data.shape[0]

        state_1 = t0_data.T

        if out_arr is not None:
            out_arr[0] = t0_data

        for i in range(integration_steps):
            deterministic = (L @ state_1) * tdelta
            random = np.random.normal(size=(num_evals, nens))
            stochastic = Q_evec @ (np.sqrt(Q_eval * tdelta) * random)
            state_2 = state_1 + deterministic + stochastic
            state_mid = (state_1 + state_2) / 2
            state_1 = state_mid
            if out_arr is not None and (i+1) % 2 == 0:
                out_arr[(i//2) + 1] = state_mid.T

        return state_mid.T.real


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

    def __init__(self, tau0_data, num_eofs, tau1_data=None, nelem_in_tau1=1,
                 h5file=None):
        """
        Parameters
        ----------
        tau0_data: ndarray
            Data for calibrating the LIM.  Expects
            a 2D MxN matrix where M (rows) represent the sampling dimension and
            N(columns) represents the feature dimension (e.g. spatial grid
            points).
        num_eofs: int
            Number of principal components to include in forecast calculations
        tau1_data: ndarray, optional
            Data with lag of tau=1.  Used to calculate the mapping term, G1,
            going from tau0 to tau1.  Must be the same shape as tau0_data.  If
            not provided, tau0_data is assumed to be sequential and
            nelem_in_tau1 and tau0_data is used to calculate lag covariance.
        nelem_in_tau1: int, optional
            Number of time samples that span tau=1.  E.g. for monthly data when
            a forecast tau is equivalent to 1 year, nelem_in_tau should be 12.
            Used if tau1_data is not provided.
        h5file: HDF5_Object, Optional
            File object to store LIM output.  It will create a series of
            directories under the given group
        """

        self._neigs = num_eofs

        logger.info('Calculating EOFs on calibration data.')
        stime = time.time()
        self._eofs = calc_eofs(tau0_data, num_eofs,
                               var_stats_dict=self._eof_var_stats)
        eof_tau0_data = np.dot(self._eofs.T, tau0_data.T)

        if tau1_data is not None:
            eof_tau1_data = np.dot(self._eofs.T, tau1_data.T)
        else:
            eof_tau1_data = None

        etime = time.time() - stime
        logger.info('EOF truncation finished in {:2.2f}s'.format(etime))

        if self._eof_var_stats is not None:
            logger.debug(('\nEOF statistics:\n'
                          'Total variance: {:2.4e}\n'
                          'Var explained (ret modes): {:2.4e}'
                          ).format(self._eof_var_stats['total_var'],
                                   self._eof_var_stats['var_expl_by_ret']))

        super(ParamReducedLIM, self).__init__(eof_tau0_data,
                                              tau1_data=eof_tau1_data,
                                              nelem_in_tau1=nelem_in_tau1,
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

        print('Beginning resampling forecast experiment.')

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

            print('Trial {} finished.'.format(j+1))

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
