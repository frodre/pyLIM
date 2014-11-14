#-*- coding: utf-8 -*-
from numpy import sqrt, cos, radians, dot, log, exp, zeros
from numpy.linalg import pinv

from Stats import calc_EOFs, run_mean
import LIMTools as lt

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
    
    def __init__(self, calibration, wsize, fcast_times, fcast_num_PCs,
                  area_wgt_lats=None, use_G=False, H5file=None):
        """
        TODO UPDATE
        Parameters
        ----------
        calibration: ndarray
            Dataset for determining LIM forecast EOFs.  Provided data should be
            in a 2D MxN matrix where M (rows) represent temporal samples and
            N(columns) represent spatial samples. Data should be in spatial
            anomaly format.
        fcast_times: array_like
            1D array-like object containing all times to forecast at with the
            LIM.
        H5file: HDF5_Object
            File object to store LIM output.  It will create a series of 
            directories under the given group
        """
        
        self._calibration = calibration
        self._wsize = wsize
        self.fcast_times = fcast_times
        self._neigs = fcast_num_PCs
        self._lats = area_wgt_lats
        self._use_G = use_G
        self._H5file = H5file
        
        #Calculate anomaly time series from the data
        self._anomaly_srs, _bedge, _tedge = run_mean(self._calibration, 
                                             self._wsize,
                                             self._H5file,
                                             shaveYr=True)
        self._obs_use = [_bedge, calibration.shape[0]-_tedge]
        self._anomaly_srs, self._climo = lt.calc_anomaly(self._anomaly_srs,
                                                         self._wsize)
        
        # Use a 1-tau lag for calculating forecast variable L
        if not use_G:
            self._tau = wsize
        
    def _area_wgt(data, lats):
        """Apply area weighting to data based on provided latitude values.
        
        Uses sqrt(cos(lat)) as a scaling function to each data point.  Provided 
        latitude vector needs to have the same last shape dimension as the data
        for proper broadcasting over the data array.
        
        Parameters
        ----------
        data: ndarray
            Spatial data to apply the area weight scaling to.
        lats: ndarray
            Latitude array matching the spatial shape of the data
            
        Returns
        -------
        Area-weighted data array
        """
        assert(data.shape[-1] == lats.shape[-1])
        scale = sqrt(cos(radians(lats)))
        return data * scale
        
    def _cnvt_EOF_space(data, neigs):
        "Calculate EOFs and project data into EOF space."
        eofs, _, var_pct = calc_EOFs(data.T)
        eof_proj = dot(eofs.T, data.T)
        return (eof_proj, eofs)
        
    def _calc_M(x0, xt, tau, use_G):
        "Calculate either L or G for forecasting"
        x0x0 = dot(x0, x0.T)
        x0xt = dot(xt, x0.T)
        
        #Calculate tau-lag G value
        M = dot(x0xt, pinv(x0x0))
        
        if use_G:
            return M
        else:
            return log(M)/tau  
        
    def forecast(self, t0_data, use_G = False):
        """Run LIM forecast from given data.
        
        Performs LIM forecast over the times specified by the
        fcast_times class attribute.  Forecast can be performed by calculating
        G for each time period or by L for a 1-year(or window_size) lag and then
        calculating each fcast_Time G from that L matrix.
        
        Parameters
        ----------
        t0_data: ndarray
            MxN array to forecast from.  M is the sample dimension, while N is 
            the spatial dimension.  1-window length chunk will be removed from
            each edge from the anomaly calculation procedure.
        use_G: bool
            Flag for using only the G-matrix for forecasting
            
        Returns
        -----
        fcast_out: ndarray
            LIM forecasts in a KxMxN matrix where K corresponds to each forecast
            time.
            
        Notes
        -----
        This method will set the fcast_out attribute for the LIM. If an HDF5 obj
        is provided it will output the forecast to this file.
        """
        
        #Calculate anomalies for initial data
        t0_data, _, _ = run_mean(t0_data, self._wsize, shaveYr=True)
        t0_data = lt.calc_anomaly(t0_data, self._wsize, self._climo)
        
        #This will be replaced with HDF5 stuff if provided
        fcast_out = [zeros(t0_data.shape) for fcast in self.fcast_times]
        
        #Area Weighting if _lats is set
        if self._lats is not None:
            data = self._area_wgt(self._anmaly_srs, self._lats)
        else:
            data = self._anomaly_srs
        
        #Calibrate the LIM with EOFs from training data    
        train_data, eofs = self._cnvt_EOF_space(data, self._neigs)
        
        # Forecasts using L to determine G-values
        if not use_G:
            tau = self._tau 
            train_tdim = train_data.shape[0] - tau  #
            x0 = train_data[:, 0:train_tdim]
            xt = train_data[:, tau:(train_tdim+tau)]
            
            L = self._calc_M(x0, xt, tau, use_G)
            xf = t0_data
            for i,tau in enumerate(self.fcast_times*self._wsize):
                G = exp(L*tau)
                xf = dot(G, t0_data)
                fcast_out[i] = xf
        
        # Forecasts using G only    
        else:
            train_tdim = train_data.shape[0] - self.fcast_times[-1]*self._wsize
            x0 = train_data[:,0:train_tdim]
            
            for i,fcast in enumerate(self.fcast_times*self._wsize):
                xt = train_data[:, tau:(train_tdim+tau)]
                G = self._calc_M(x0, xt, tau, use_G=True)
                xf = dot(G, t0_data)
                fcast_out[i] = xf
                
        
        return fcast_out
