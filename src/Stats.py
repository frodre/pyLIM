"""
Toolbox for statistical methods.

"""

import numpy as np
import numexpr as ne
import tables as tb
from math import ceil
from scipy.sparse import svds


def run_mean(data, window_size, shave_yr=False):
    """
    A function for calculating the running mean on data.

    Parameters
    ----------
    data: ndarray
        Data matrix to perform running mean over. Expected to be in time(row) x
        space(column) format.
    window_size: int
        Size of the window to compute the running mean over.

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
    yrsize = 12
    assert(dshape[0] >= window_size), ("Window size must be smaller than or "
                                       "equal to the length of the time "
                                       "dimension of the data.")
    if shave_yr:
        tedge = window_size/2
        cut_from_top = yrsize*int(ceil(tedge/12.0))
        bedge = (window_size/2) + (window_size % 2) - 1
        cut_from_bot = yrsize*int(ceil(bedge/12.0))
    else:
        cut_from_top = window_size/2
        bedge = cut_from_bot = (window_size/2) + (window_size % 2) - 1
    tot_cut = cut_from_top + cut_from_bot
    new_shape = list(dshape)
    new_shape[0] -= tot_cut
    
    assert(new_shape[0] > 0), ("Not enough data to trim partial years from "
                               "edges.  Please try with shaveYr=False")

    result = np.zeros(new_shape, dtype=data.dtype)
        
    for i in xrange(new_shape[0]):
        #if i % 100 == 0:
        #    print 'Calc for index %i' % i
        cntr = cut_from_bot - bedge + i
        result[i] = (data[cntr:(cntr+window_size)].sum(axis=0) /
                     float(window_size))
    
    return result, cut_from_bot, cut_from_top
   

def calc_eofs(data, num_eigs, ret_pcs=False):
    """
    Method to calculate the EOFs of given  dataset.  This assumes data comes in as
    an m x n matrix where m is the spatial dimension and n is the sampling
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

    """  # TODO: Finish returns
    
    eofs, svals, pcs = svds(data, k=num_eigs)
    eofs = eofs[:, ::-1]
    svals = svals[::-1]
    pcs = pcs[::-1]
    # eig_vals = (svals ** 2) / (len(e) - 1.)
    # tot_var = (eig_vals[0:num_eigs].sum()) / eig_vals.sum()

    if ret_pcs:
        return eofs, svals, pcs
    else:
        return eofs, svals


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
    """  # TODO: Finish returns

    assert(fcast.shape == trial_obs.shape)

    cvar = obs.var(axis=0)
    error = ne.evaluate('(trial_obs - fcast)**2')
    evar = error.sum(axis=0)/(len(error))
    return 1 - evar/cvar


def calc_lac(fcast, obs):
    """
    Method to calculate the Local Anomaly Correlation (LAC).
    
    Note: If necessary (memory concerns) in the future, the numexpr statements
    can be extended to use pytable arrays.  Would need to provide means to 
    function, as summing over the dataset is still very slow it seems. 
    
    Parameters
    ----------
    fcast: ndarray
        Time series of forecast data. M x N where M is the temporal dimension.
    obs: ndarray
        Time series of observations. M x N
    """
    
    f_mean = fcast.mean(axis=0)
    o_mean = obs.mean(axis=0)
    cov = ne.evaluate('(fcast - f_mean) * (obs - o_mean)')
    cov = cov.sum(axis=0)
    f_std = ne.evaluate('(fcast - f_mean)**2')
    f_std = np.sqrt(f_std.sum(axis=0))
    o_std = ne.evaluate('(obs - o_mean)**2')
    o_std = np.sqrt(o_std.sum(axis=0))
    std = f_std * o_std
    
    return cov / std


def calc_n_eff(data1, data2=None):

    r1 = calc_lac(data1[0:-1], data1[1:])
    n = len(data1)

    if data2 is not None:
        r2 = calc_lac(data2[0:-1], data2[1:])
        n_eff = n*((1 - r1*r2)/(1+r1*r2))
    else:
        n_eff = n*((1-r1)/(1+r1))

    return n_eff