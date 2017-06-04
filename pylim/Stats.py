"""
Toolbox for statistical methods.  For all functions, this toolbox assumes that
the first dimension is the temporal sampling dimension.

Author: Andre Perkins
"""

import numpy as np
import numexpr as ne
import dask.array as da
from math import ceil
from scipy.linalg import svd
from scipy.ndimage import convolve1d


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
        climo = data.mean(axis=0)

    if hasattr(data, 'dask'):
        if output_arr is None:
            raise ValueError('calc_anomaly requires an output array keyword '
                             'argument when operating on a Dask array.')

        anomaly = data - climo
        old_shp_anom = anomaly.reshape(old_shp)
        da.store(old_shp_anom, output_arr)
        out_climo = climo.compute()
    else:
        if output_arr is not None:
            output_arr[:] = ne.evaluate('data - climo')
        else:
            output_arr = ne.evaluate('data - climo')
        output_arr = output_arr.reshape(old_shp)
        out_climo = climo

    return output_arr, out_climo


def calc_ce(fcast, trial_obs, obs):
    """
    Method to calculate the Coefficient of Efficiency as defined by Nash and
    Sutcliffe 1970.

    Parameters
    ----------
    fcast: ndarray
        Time series of forecast data. M x N where M is the temporal dimension.
    obs: ndarray
        Time series of observations. M x N

    Returns
    -------
    CE: ndarray
        Coefficient of efficiency for all locations over the time range.
    """

    assert(fcast.shape == trial_obs.shape)

    # Climatological variance
    cvar = obs.var(axis=0, ddof=1)

    # Error variance
    error = ne.evaluate('(trial_obs - fcast)**2')
    evar = error.sum(axis=0)/(len(error))

    return 1 - evar/cvar


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
    retPCs: bool, optional
        Return principal component matrix along with EOFs

    Returns
    -------
    eofs: ndarray
        The eofs (as column vectors) of the data with dimensions n x k where
        k is the num_eigs.
    svals: ndarray
        Singular values from the svd decomposition.  Returned as a row vector
        in order from largest to smallest.
    """

    eofs, svals, pcs = svd(data[:].T, full_matrices=False)
    eofs = eofs[:, :num_eigs]
    trunc_svals = svals[:num_eigs]
    pcs = pcs[:num_eigs]

    # variance stats
    if var_stats_dict is not None:
        try:
            nt = pcs.shape[1]
            ns = eofs.shape[0]
            eig_vals = (svals**2) / (nt*ns)
            total_var = eig_vals.sum()
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
            print 'Must past dictionary type to var_stats_dict in order to ' \
                  'output variance statistics.'
            print e

    if ret_pcs:
        return eofs, trunc_svals, pcs
    else:
        return eofs, trunc_svals


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

    # Calculate covariance between time series
    cov = ne.evaluate('(fcast - f_mean) * (obs - o_mean)')
    cov = cov.sum(axis=0)

    # Calculate standarization terms
    f_std = ne.evaluate('(fcast - f_mean)**2')
    f_std = np.sqrt(f_std.sum(axis=0))
    o_std = ne.evaluate('(obs - o_mean)**2')
    o_std = np.sqrt(o_std.sum(axis=0))
    std = f_std * o_std

    return cov / std


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

    weights = [1.0/float(window_size) for _ in xrange(window_size)]
    if hasattr(data, 'dask'):
        if output_arr is None:
            raise ValueError('calc_anomaly requires an output array keyword '
                             'argument when operating on a Dask array.')

        def _run_mean_block(block):
            return convolve1d(block, weights, axis=0)

        pad = window_size // 2
        ghost = da.ghost.ghost(data, depth={0: pad}, boundary={0: 'reflect'})
        filt = ghost.map_blocks(_run_mean_block, window_size)
        unpadded = da.ghost.trim_internal(filt, {0: pad})
        if trim_edge is not None:
            unpadded = unpadded[trim_edge:-trim_edge]

        da.store(unpadded, output_arr)
    else:
        res = convolve1d(data, weights, axis=0)
        if trim_edge:
            res = res[trim_edge:-trim_edge]
        output_arr[:] = res

    return output_arr
