# coding=utf-8
"""
Main Linear Inverse Model classes and methods.

Author: Andre Perkins
"""

import numpy as np
from numpy.linalg import pinv, eigvals, eig, eigh
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
    G = np.dot(x0xt, pinv(x0x0))

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
        G = np.dot(x0xt, pinv(x0x0))

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
        L = G_evects @ np.diag(L_evals) @ pinv(G_evects)
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

    def noise_integration(self, t0_data, length, timesteps=1440,
                          out_arr=None, length_out_arr=None,
                          seed=None):
        """Perform a stochastic noise forced integration.

        Performs LIM forecast over the times specified by the
        fcast_times class attribute.  Forecast can be performed by calculating
        G for each time period or by L for a 1-year(or window_size) lag and
        then calculating each fcast_Time G from that L matrix.

        Parameters
        ----------
        t0_data: ndarray
            Initialization data for the stochastic integration.  Expects
            a 2D MxN matrix where M (rows) represent the ensemble dimension and
            N(columns) represents the feature dimension (e.g. spatial grid
            points).  The integration will produce a randomly forced trajectory
            for each ensemble member.
        length: int
            Length (in units of the calibration lag, tau) of the noise
            integration
        timesteps: int
            Number of timesteps in a single length-tau segment of the noise
            integration.  This parameter sets the deltaT for the timestepper.
            E.g., for tau=1-year, 1440 timesteps is ~6hr timestep.
        out_arr: Optional, ndarray
            Optional output container for data at the resolution of deltaT.
            Expected dimensions of (timesteps * length + 1) x N
        length_out_arr: Optional, ndarray
            Optianal output container for data at the resolution of tau.
            Expected dimensions of length x N
        seed: Optional, int
            Seed for the random number generator to perform a reproducible
            stochastic forecast


        Returns
        -----
        ndarray
            Final state of the LIM noise integration forecast. Same dimension
            as input t0_data.
        """

        if seed is not None:
            np.random.seed(seed)

        # t0_data comes in as sample x spatial
        L = self.L
        Q_eval = self.Q_evals[:, None]
        Q_evec = self.Q_evects
        tdelta = 1/timesteps
        integration_steps = int(timesteps * length)
        num_evals = Q_eval.shape[0]
        nens = t0_data.shape[0]

        state_1 = t0_data.T
        state_mid = state_1

        if out_arr is not None:
            out_arr[0] = t0_data

        # Perform noise integration
        for i in range(integration_steps):
            deterministic = (L @ state_1) * tdelta
            random = np.random.normal(size=(num_evals, nens))
            stochastic = Q_evec @ (np.sqrt(Q_eval * tdelta) * random)
            state_2 = state_1 + deterministic + stochastic
            state_mid = (state_1 + state_2) / 2
            state_1 = state_2
            if out_arr is not None:
                out_arr[i + 1] = state_mid.T

            if length_out_arr is not None and i % timesteps == 0 and i != 0:
                len_out_arr_idx = i // timesteps
                length_out_arr[len_out_arr_idx] = state_mid.T

        return state_mid.T.real
