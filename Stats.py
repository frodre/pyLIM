"""
Toolbox for statistical methods.

"""

import numpy as np
import numexpr as ne
import tables as tb
import pandas as pd
from scipy.sparse.linalg import eigs
from math import ceil

def runMean(data, window_size, h5_file=None, h5_parent=None, shaveYr=False):
    """
    A function for calculating the running mean on data.

    Parameters
    ----------
    data: ndarray
        Data matrix to perform running mean over. Expected to be in time(row) x
        space(column) format.
    window_size: int
        Size of the window to compute the running mean over.
    h5_file:  tables.file.File, optional
       Output hdf5 file (utilizes pyTables) for the calculated running mean.
    h5_parent: tables.group.*, optional
        Parent node to place run_mean dataset under.

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
    assert( dshape[0] >= window_size ), ("Window size must be smaller than or "
                                          "equal to the length of the time "
                                          "dimension of the data.")
    if shaveYr:
        tedge = window_size/2
        cut_from_top = yrsize*int(ceil(tedge/12.0))
        bedge = (window_size/2) + (window_size%2) - 1
        cut_from_bot = yrsize*int(ceil(bedge/12.0))
    else:
        cut_from_top = window_size/2
        bedge = cut_from_bot = (window_size/2) + (window_size%2) - 1
    tot_cut = cut_from_top + cut_from_bot
    new_shape = list(dshape)
    new_shape[0] -= tot_cut
    
    assert(new_shape[0] > 0), ("Not enough data to trim partial years from "
                                "edges.  Please try with shaveYr=False")

    if h5_file is not None:
        is_h5 = True
        if h5_parent is None:
            h5_parent = h5_file.root
        
        try:
            result = h5_file.create_carray(h5_parent, 
                                       'run_mean',
                                       atom = tb.Atom.from_dtype(data.dtype),
                                       shape = new_shape,
                                       title = '12-month running mean')
        except tb.NodeError:
            h5_file.remove_node(h5_parent.run_mean)
            result = h5_file.create_carray(h5_parent, 
                                       'run_mean',
                                       atom = tb.Atom.from_dtype(data.dtype),
                                       shape = new_shape,
                                       title = '12-month running mean')
            
    else:
        is_h5 = False                                       
        result = np.zeros(new_shape, dtype=data.dtype)
        
    for i in xrange(new_shape[0]):
        #if i % 100 == 0:
        #    print 'Calc for index %i' % i
        cntr = cut_from_bot - bedge + i
        result[i] = (data[(cntr):(cntr+window_size)].sum(axis=0) / 
                        float(window_size))
                        
    if is_h5:
        result = result.read()
    
    return (result, cut_from_bot, cut_from_top)
   

def calcEOF(data, num_eigs, retPCs = False):
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

    """ #TODO: Finish returns
    
    eofs, E, pcs = np.linalg.svd(data, full_matrices=False)
    eig_vals = (E ** 2) / (len(E) - 1.)
    tot_var = (eig_vals[0:num_eigs].sum()) / eig_vals.sum()

    return (eofs[:,0:num_eigs], eig_vals[0:num_eigs], tot_var)
    
def calcCE(fcast, obs):
    """
    Method to calculate the Coefficient of Efficiency as defined by Nash and
    Sutcliffe 1970.
    
    Parameters
    ----------
    fcast: ndarray
        Time series of forecast data. M x N where M is the temporal dimension.
    obs: ndarray
        Time series of observations. M x N
    """ #TODO: Finish returns
     
    cvar = obs.var(axis=0)
    error = ne.evaluate('(obs - fcast)**2')
    evar = error.sum(axis=0)/(len(error))
    return 1 - evar/cvar
    
def calcLCA(fcast, obs):
    """
    Method to calculate the Local Anomaly Correlation (LCA).
    
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
    
    f_mean = fcast.sum(axis=0) / float(fcast.shape[0])
    o_mean = obs.sum(axis=0) / float(obs.shape[0])
    cov = ne.evaluate('(fcast - f_mean) * (obs - o_mean)')
    cov = cov.sum(axis=0)
    f_std = ne.evaluate('(fcast - f_mean)**2')
    f_std = np.sqrt(f_std.sum(axis=0))
    o_std = ne.evaluate('(obs - o_mean)**2')
    o_std = np.sqrt(o_std.sum(axis=0))
    std = f_std * o_std
    
    return cov / std
    
    

