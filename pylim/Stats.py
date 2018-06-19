"""
Toolbox for statistical methods.  For all functions, this toolbox assumes that
the first dimension is the temporal sampling dimension.

Author: Andre Perkins
"""

import numpy as np
import numexpr as ne
import dask.array as da
import logging
from scipy.linalg import svd
from scipy.ndimage import convolve1d
from sklearn import linear_model

logger = logging.getLogger(__name__)


def detrend_data(data, output_arr=None):
    """
    Detrend data using a linear fit.
    
    Parameters
    ----------
    data: ndarray-like
        Input dataset to detrend.  Assumes leading axis is sampling dimension.
    output_arr: ndarray-like, optional
        Output array with same shape as data to store detrended data.
    """

    dummy_time = np.arange(data.shape[0])[:, None]
    model = linear_model.LinearRegression(fit_intercept=False, n_jobs=-1)
    model.fit(dummy_time, data)

    linfit = model.predict(dummy_time)
    detrended = data - linfit

    if output_arr is not None:
        output_arr[:] = detrended
    else:
        output_arr = detrended

    return output_arr


def dask_detrend_data(data, output_arr):
    """
    Detrend data using a linear fit.

    Parameters
    ----------
    data: dask.array
        Input dataset to detrend.  Assumes leading axis is sampling dimension.
    output_arr: ndarray-like
        Output array with same shape as data to store detrended data.

    Notes
    -----
    This is a very expensive operation if using a large dataset.  May slow down
    if forced to spill onto the disk cache  It does not currently take into 
    account X data.  Instead, it creates a dummy array (using arange) for 
    sampling points.
    """

    dummy_time = np.arange(data.shape[0])[:, None]
    dummy_time = da.from_array(dummy_time, chunks=dummy_time.shape)

    # intercept handling
    x_offset = dummy_time.mean(axis=0)
    x_centered = dummy_time - x_offset
    y_offset = data.mean(axis=0)
    y_centered = data - y_offset

    coefs, resid, rank, s = da.linalg.lstsq(x_centered, y_centered)

    intercepts = y_offset - x_offset*coefs
    predict = da.dot(dummy_time, coefs) + intercepts
    detrended = data - predict

    da.store(detrended, output_arr)

    return output_arr


def calc_anomaly(data, yrsize, climo=None, output_arr=None):
    """
    Caculate anomaly for the given data.  Right now it assumes sub-annual data
    input so that the climatology subtracts means for each month instead
    of the mean of the entire series.

    Note: May take yrsize argument out and leave it to user to format data
    as to take the desired anomaly.

    Parameters
    ----------
    data: ndarray
        Input data to calculate the anomaly from.  Leading dimension should be
        the temporal axis.
    yrsize: int
        Number of elements that compose a full year.  Used to reshape the data
        time axis to num years x size year for climatology purposes.
    climo: ndarray, optional
        User-provided climatology to subtract from the data.  Must be
        broadcastable over the time-dimension of data
    output_arr: ndarray-like, optional
        Array to place output of anomaly calculation that supports 
        ndarray-like slicing.  This is required for dask array input.
    Returns
    -------
    anomaly: ndarray-like
        Data converted to its anomaly form.
    climo: ndarray
        The calculated climatology that was subtracted from the data
    """

    yrsize = int(yrsize)
    if not yrsize >= 1:
        raise ValueError('yrsize must be an integer >= 1')

    # Reshape to take monthly mean
    old_shp = data.shape
    new_shp = (old_shp[0]//yrsize, yrsize, old_shp[1])
    data = data.reshape(new_shp)

    # Use of data[:] should work for ndarray or ndarray-like
    if climo is None:
        climo = data.mean(axis=0, keepdims=True)

    if is_dask_array(data):
        if output_arr is None:
            raise ValueError('calc_anomaly requires an output array keyword '
                             'argument when operating on a Dask array.')

        anomaly = data - climo
        old_shp_anom = anomaly.reshape(old_shp)
        da.store(old_shp_anom, output_arr)
        out_climo = climo.compute()
    else:
        if output_arr is not None:
            output_arr[:] = np.squeeze(ne.evaluate('data - climo'))
        else:
            output_arr = np.squeeze(ne.evaluate('data - climo'))
        output_arr = output_arr.reshape(old_shp)
        out_climo = climo

    return output_arr, out_climo


# def calc_ce(fcast, trial_obs, obs):
#     """
#     Method to calculate the Coefficient of Efficiency as defined by Nash and
#     Sutcliffe 1970.
#
#     Parameters
#     ----------
#     fcast: ndarray
#         Time series of forecast data. M x N where M is the temporal dimension.
#     obs: ndarray
#         Time series of observations. M x N
#
#     Returns
#     -------
#     CE: ndarray
#         Coefficient of efficiency for all locations over the time range.
#     """
#
#     assert(fcast.shape == trial_obs.shape)
#
#     # Climatological variance
#     cvar = obs.var(axis=0, ddof=1)
#
#     # Error variance
#     error = ne.evaluate('(trial_obs - fcast)**2')
#     evar = error.sum(axis=0)/(len(error))
#
#     return 1 - evar/cvar


def calc_eofs(data, num_eigs, ret_pcs=False, var_stats_dict=None):
    """
    Method to calculate the EOFs of given  dataset.  This assumes data comes in as
    an m x n matrix where m is the temporal dimension and n is the spatial
    dimension.

    Parameters
    ----------
    data: ndarray
        Dataset to calculate EOFs from
    num_eigs: int
        Number of eigenvalues/vectors to return.  Must be less than min(m, n).
    ret_pcs: bool, optional
        Return principal component matrix along with EOFs
    var_stats_dict: dict, optional
        Dictionary target to star some simple statistics about the EOF
        calculation.  Note: if this is provided for a dask array it prompts two
        SVD calculations for both the compressed and full singular values.

    Returns
    -------
    eofs: ndarray
        The eofs (as column vectors) of the data with dimensions n x k where
        k is the num_eigs.
    svals: ndarray
        Singular values from the svd decomposition.  Returned as a row vector
        in order from largest to smallest.
    """

    if is_dask_array(data):
        pcs, full_svals, eofs = da.linalg.svd_compressed(data, num_eigs)
        var = da.var(data, axis=0)

        out_svals = np.zeros(num_eigs)
        out_eofs = np.zeros((num_eigs, data.shape[1]))
        out_pcs = np.zeros((data.shape[0], num_eigs))
        out_var = np.zeros((data.shape[1]))
        da.store([eofs, full_svals, pcs, var],
                 [out_eofs, out_svals, out_pcs, out_var])

        out_eofs = out_eofs.T
        out_pcs = out_pcs.T

    else:
        eofs, full_svals, pcs = svd(data[:].T, full_matrices=False)
        out_eofs = eofs[:, :num_eigs]
        out_svals = full_svals[:num_eigs]
        out_pcs = pcs[:num_eigs]
        out_var = data[:].var(ddof=1, axis=0)

    # variance stats
    if var_stats_dict is not None:
        try:
            nt = data.shape[0]
            ns = data.shape[1]
            eig_vals = (out_svals ** 2) / nt
            total_var = out_var.sum()
            var_expl_by_mode = eig_vals / total_var
            var_expl_by_retained = var_expl_by_mode[0:num_eigs].sum()

            var_stats_dict['nt'] = nt
            var_stats_dict['ns'] = ns
            var_stats_dict['eigvals'] = eig_vals
            var_stats_dict['num_ret_modes'] = num_eigs
            var_stats_dict['total_var'] = total_var
            var_stats_dict['var_expl_by_mode'] = var_expl_by_mode
            var_stats_dict['var_expl_by_ret'] = var_expl_by_retained
        except TypeError as e:
            print('Must past dictionary type to var_stats_dict in order to ' \
                  'output variance statistics.')
            print(e)

    if ret_pcs:
        return out_eofs, out_svals, out_pcs
    else:
        return out_eofs, out_svals


def calc_lac(fcast, obs):
    """
    Method to calculate the Local Anomaly Correlation (LAC).  Uses numexpr
    for speed over larger datasets.

    Note: If necessary (memory concerns) in the future, the numexpr statements
    can be extended to use pytable arrays.  Would need to provide means to
    function, as summing over the dataset is still very slow it seems.

    Parameters
    ----------
    fcast: ndarray
        Time series of forecast data. M x N where M is the temporal dimension.
    obs: ndarray
        Time series of observations. M x N

    Returns
    -------
    lac: ndarray
        Local anomaly corellations for all locations over the time range.
    """
    # Calculate means of data
    f_mean = fcast.mean(axis=0)
    o_mean = obs.mean(axis=0)
    f_anom = fcast - f_mean
    o_anom = obs - o_mean

    # Calculate covariance between time series at each gridpoint
    cov = (f_anom * o_anom).sum(axis=0)

    # Calculate standardization terms
    f_std = (f_anom**2).sum(axis=0)
    o_std = (o_anom**2).sum(axis=0)
    if is_dask_array(f_std):
        f_std = da.sqrt(f_std)
    else:
        f_std = np.sqrt(f_std)

    if is_dask_array(o_std):
        o_std = da.sqrt(o_std)
    else:
        o_std = np.sqrt(o_std)

    std = f_std * o_std
    lac = cov / std

    return lac


def calc_mse(fcast, obs):
    sq_err = (obs - fcast)**2
    mse = sq_err.mean(axis=0)
    return mse


def calc_ce(fcast, obs):

    sq_err = (obs - fcast)**2
    obs_mean = obs.mean(axis=0)
    obs_var = (obs - obs_mean)**2
    ce = 1 - (sq_err.sum(axis=0) / obs_var.sum(axis=0))
    return ce


def calc_n_eff(data1, data2=None):
    """
    Calculate the effective degrees of freedom for data using lag-1
    autocorrelation.

    Parameters
    ----------
    data1: ndarray
        Dataset to calculate effective degrees of freedom for.  Assumes
        first dimension is the temporal dimension.
    data2: ndarray, optional
        A second dataset to calculate the effective degrees of freedom
        for covariances/correlations etc.

    Returns
    -------
    n_eff: ndarray
        Effective degrees of freedom for input data.
    """

    if data2 is not None:
        assert data1.shape == data2.shape,\
            'Data must have have same shape for combined n_eff calculation'

    # Lag-1 autocorrelation
    r1 = calc_lac(data1[0:-1], data1[1:])
    n = len(data1)

    if data2 is not None:
        r2 = calc_lac(data2[0:-1], data2[1:])
        n_eff = n*((1 - r1*r2)/(1+r1*r2))
    else:
        n_eff = n*((1-r1)/(1+r1))

    return n_eff


def run_mean(data, window_size, trim_edge=None, output_arr=None):
    """
    A function for calculating the running mean on data.

    Parameters
    ----------
    data: ndarray
        Data matrix to perform running mean over. Expected to be in time(row) x
        space(column) format. And that samples span full years.
    window_size: int
        Size of the window to compute the running mean over.
    trim_edge: int, optional
        Remove specified items from the start and end of the sampling
        dimension of the running mean.  Otherwise the window_size/2 items at
        the start and the end will have reflected padding effects.
    output_arr: ndarray-like, optional
        Array to place output of running mean that supports
        ndarray-like slicing.  This is required for dask array input.

    Returns
    -------
    result: ndarray
        Running mean result of given data.
    bot_edge: int
        Number of elements removed from beginning of the time series
    top_edge: int
        Number of elements removed from the ending of the time series
    """

    sample_len = data.shape[0]
    if sample_len < window_size:
        raise ValueError("Window size must be smaller than or equal to the "
                         "length of the time dimension of the data.")

    if trim_edge is not None:
        sample_len -= trim_edge*2

        if sample_len < 1:
            raise ValueError('Not enough data to trim edges. Please try with '
                             'trim_edge=None')

    weights = [1.0/float(window_size) for _ in range(window_size)]
    if is_dask_array(data):
        if output_arr is None:
            raise ValueError('calc_anomaly requires an output array keyword '
                             'argument when operating on a Dask array.')

        def _run_mean_block(block):
            return convolve1d(block, weights, axis=0)

        old_chunk_shape = data
        pad = window_size // 2
        ghost = da.ghost.ghost(data, depth={0: pad}, boundary={0: 'reflect'})
        filt = ghost.map_blocks(_run_mean_block)
        unpadded = da.ghost.trim_internal(filt, {0: pad})
        if trim_edge is not None:
            unpadded = unpadded[trim_edge:-trim_edge]

        da.store(unpadded, output_arr)
    else:
        res = convolve1d(data, weights, axis=0)
        if trim_edge:
            res = res[trim_edge:-trim_edge]

        if output_arr is not None:
            output_arr[:] = res
        else:
            output_arr = res

    return output_arr


def is_dask_array(arr):
    return hasattr(arr, 'dask')
