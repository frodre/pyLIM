# -*- coding: utf-8 -*-
from numpy import sqrt, cos, radians, dot

from Stats import calc_EOFs, run_mean
from LIMTools import calc_anomaly

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
        
        self._mean_srs, _bedge, _tedge = run_mean(self._calibration, 
                                             self.wsize,
                                             self._H5file,
                                             shaveYr=True)
        self._obs_use = [_bedge, calibration.shape[0]-_tedge]
        self._anomaly_srs = calc_anomaly(_mean_srs, self.wsize)
        
        del _mean_srs
        
    def _area_wgt(data, lats):
        assert(data.shape[-1] == lats.shape[-1])
        scale = sqrt(cos(radians(lats)))
        return data * scale
        
    def _cnvt_EOF_space(data, neigs):
        eofs, _, var_pct = calc_EOFs(data.T)
        eof_proj = dot(eofs.T, data.T)
        return (eof_proj, eofs)
        
    def _calc_M(data, tau_vals, use_G):
        if use_G:
            for i,tau in enumerate(tau_vals):
                pass
        else:
            pass
                
         
        
    def forecast(self, t0_data, use_G = False):
        train_data = self._cnvt_EOF_space(self._anomaly_srs, self._neigs)
        
        if not use_G:
            train_tdim = train_data.shape[0] - self._wsize
        else:
            train_tdim = train_data.shape[0] - self.fcast_times[-1]
        
        
        
        pass
        
    