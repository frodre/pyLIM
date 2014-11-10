# -*- coding: utf-8 -*-
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
                  H5file=None):
        """
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
        self.fcast_times = fcast_times
        self._H5file = H5file
        self._wsize = wsize
        
        _mean_srs, _bedge, _tedge = run_mean(self._calibration, 
                                             self.wsize,
                                             self._H5file,
                                             shaveYr=True)
        _obs_use = [_bedge, calibration.shape[0]-_tedge]
        _anomaly_srs = calc_anomaly(_mean_srs, self.wsize)
        
    def forecast(t0_data):
        pass
        
    