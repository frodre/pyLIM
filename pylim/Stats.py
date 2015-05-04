"""
Toolbox for statistical methods.  For all functions, this toolbox assumes that
the first dimension is the temporal sampling dimension.

Author: Andre Perkins
"""

import numpy as np
import numexpr as ne
from math import ceil
from scipy.sparse.linalg import svds


def calc_anomaly(data, yrsize, climo=None):
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

    Returns
    -------
    anomaly: ndarray
        Data converted to its anomaly form.
    """

    # Reshape to take monthly mean
    old_shp = data.shape
    new_shp = (old_shp[0]//yrsize, yrsize, old_shp[1])

    if climo is None:
        climo = data.reshape(new_shp).mean(axis=0)

    anomaly = data.reshape(new_shp) - climo
    return anomaly.reshape(old_shp), climo


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


def calc_eofs(data, num_eigs, ret_pcs=False):
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

    eofs, svals, pcs = svds(data.T, k=num_eigs)
    eofs = eofs[:, ::-1]
    svals = svals[::-1]
    pcs = pcs[::-1]

    if ret_pcs:
        return eofs, svals, pcs
    else:
        return eofs, svals


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


def run_mean(data, window_size, shave_yr=False, year_len=12):
    """
    A function for calculating the running mean on data.

    Parameters
    ----------
    data: ndarray
        Data matrix to perform running mean over. Expected to be in time(row) x
        space(column) format.
    window_size: int
        Size of the window to compute the running mean over.
    shave_yr: bool, optional
        The running mean will remove ends off the data. If shave_yr is true it
        will remove an full year_len chunk from the ends instead of a
        partial chunk.
    year_len: int, optional
        Number of elements in a timeseries that represents a full year.

    Returns
    -------
    result: ndarray
        Running mean result of given data.
    bot_edge: int
        Number of elements removed from beginning of the time series
    top_edge: int
        Number of elements removed from the ending of the time series
    """
    
    dshape = data.shape
    assert(dshape[0] >= window_size), ("Window size must be smaller than or "
                                       "equal to the length of the time "
                                       "dimension of the data.")

    # Determine how much data is removed from beginning and end
    if shave_yr:
        tedge = window_size//2
        cut_from_top = year_len * int(ceil(tedge/float(year_len)))
        bedge = (window_size//2) + (window_size % 2) - 1
        cut_from_bot = year_len * int(ceil(bedge/float(year_len)))
    else:
        cut_from_top = window_size//2
        bedge = cut_from_bot = (window_size//2) + (window_size % 2) - 1
    tot_cut = cut_from_top + cut_from_bot
    new_shape = list(dshape)
    new_shape[0] -= tot_cut
    
    assert(new_shape[0] > 0), ("Not enough data to trim partial years from "
                               "edges.  Please try with shaveYr=False")

    # Allocate memory and perform running mean
    result = np.zeros(new_shape, dtype=data.dtype)
        
    for i in xrange(new_shape[0]):
        cntr = cut_from_bot - bedge + i
        result[i] = (data[cntr:(cntr+window_size)].sum(axis=0) /
                     float(window_size))
    
    return result, cut_from_bot, cut_from_top