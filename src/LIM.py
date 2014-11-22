#-*- coding: utf-8 -*-
from numpy import sqrt, cos, radians, dot, log, exp, zeros, array, int16
from numpy.linalg import pinv

from Stats import calc_eofs, run_mean
import LIMTools as Lt


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
    
    #Calculate tau-lag G value
    return dot(x0xt, pinv(x0x0))


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
                 area_wgt_lats=None, h5file=None):
        """
        Parameters
        ----------
        calibration: ndarray
            Dataset for determining LIM forecast EOFs.  Provided data should be
            in a 2D MxN matrix where M (rows) represent temporal samples and
            N(columns) represent spatial samples. Data should be in spatial
            anomaly format.
        wsize: int
            Windowsize for running mean.  For this implementation it should
            be equal to a year's worth of samples
        fcast_times: array_like
            1D array-like object containing all times to forecast at with the
            LIM.
        fcast_num_pcs: int
            Number of principal components to include in forecast calculations
        area_wgt_lats: ndarray, Optional
            Latitude vector pertaining to spatial dimension N.  If used area-
            weighting will be performed for pricipal component calculations.
            TODO: add ability to provide a simple non-tiled lat vector
        H5file: HDF5_Object, Optional
            File object to store LIM output.  It will create a series of 
            directories under the given group
        """
        
        self._calibration = calibration
        self._wsize = wsize
        self.fcast_times = array(fcast_times, dtype=int16)
        self._neigs = fcast_num_pcs
        self._lats = area_wgt_lats
        self._H5file = h5file
        self._obs_use = self._anomaly_srs = self._climo = None

    def forecast(self, t0_data, use_lag1=True):
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
            each edge from the anomaly calculation procedure.  N^ = N - 2*wsize
        use_lag1: bool
            Flag for using only the G_1-matrix for forecasting
            
        Returns
        -----
        fcast_out: ndarray
            LIM forecasts in a KxMxN^ matrix where K corresponds to each
            forecast time.
            
        Notes
        -----
        This method will set the fcast_out attribute for the LIM. If an HDF5
        obj is provided it will output the forecast to this file.
        """

        #Calculate anomaly time series from the data
        self._anomaly_srs, _bedge, _tedge = run_mean(self._calibration,
                                                     self._wsize,
                                                     self._H5file,
                                                     shave_yr=True)
        self._obs_use = [_bedge, self._calibration.shape[0]-_tedge]
        self._anomaly_srs, self._climo = Lt.calc_anomaly(self._anomaly_srs,
                                                         self._wsize)
        
        #Calculate anomalies for initial data
        t0_data, _, _ = run_mean(t0_data, self._wsize, shave_yr=True)
        t0_data, _ = Lt.calc_anomaly(t0_data, self._wsize, self._climo)  # MxN^
        
        #This will be replaced with HDF5 stuff if provided
        fcast_out_shp = [len(self.fcast_times)] + list(t0_data.shape)  # KxMxN^
        fcast_out = zeros(fcast_out_shp)
        
        #Area Weighting if _lats is set
        if self._lats is not None:
            data = _area_wgt(self._anomaly_srs, self._lats)
        else:
            data = self._anomaly_srs
        
        #Calibrate the LIM with (J=neigs) EOFs from training data
        eofs, _, var_pct = calc_eofs(data.T, self._neigs)         # eofs is MxJ
        train_data = dot(eofs.T, self._anomaly_srs.T)
        
        #Project our testing data into eof space
        proj_t0_data = dot(eofs.T, t0_data.T)                      # JxN^
        
        # Forecasts using L to determine G-values
        if use_lag1:
            #Calculate L from time-lag of one window size (1-year for our LIM)
            tau = self._wsize  
            train_tdim = train_data.shape[1] - tau  #
            x0 = train_data[:, 0:train_tdim]
            xt = train_data[:, tau:(train_tdim+tau)]
            
            g_1 = _calc_m(x0, xt)
            for i, tau in enumerate(self.fcast_times):
                g = g_1**tau
                xf = dot(g, proj_t0_data)
                fcast_out[i] = dot(xf.T, eofs.T)
        
        # Forecasts using G only    
        else:
            train_tdim = train_data.shape[1] - self.fcast_times[-1]*self._wsize
            x0 = train_data[:, 0:train_tdim]
            
            for i, tau in enumerate(self.fcast_times*self._wsize):
                xt = train_data[:, tau:(train_tdim+tau)]
                g = _calc_m(x0, xt)
                xf = dot(g, proj_t0_data)
                fcast_out[i] = dot(xf.T, eofs.T)
        
        return fcast_out
