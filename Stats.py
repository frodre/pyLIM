"""
Toolbox for statistical methods.

"""

import numpy as np
import threading
import subprocess

def runMean(nc_file, nc_varname, window_size, threading=False, num_threads=None, useNCO=False):
    """
    A function for calculating the running mean on data.

    Parameters
    ----------
    data: ndarray
        Data to compute the running mean over.
    window_size: int
        Size of the window to compute the running mean over.
    threading: bool, optional
        Try threading the process for speedup. Default is false.
    num_threads: int, optional
        Number of threads for calculating the running mean.  Only
        valid if threading is set to True.  If value is not 
        specified and threading is requested, the default value 
        is 4.
    useNCO: bool, optional
        If the calculation is over a large dataset where memory use
        may be an issue, the use of netCDF operators is likely a good
        idea.  It will speed up the process and save memory...

    Returns
    -------
    result: ndarray
        Running mean result of given data.
    bot_edge: int
        Number of elements removed from beginning of the time series
    top_edge: int
        Number of elements removed from the ending of the time series
    """

    if threading and num_threads is None:
        num_threads = 4

    #Calculate edges of running mean calculatable from given data
    data = nc_file.variables[nc_varname].data
    dshape = data.shape
    top_edge = window_size/2 + 1
    bot_edge = (window_size/2) + (window_size%2) - 1
    tot_cut = top_edge + bot_edge
    new_shape = list(dshape)
    new_shape[0] -= tot_cut

    if useNCO:
        inFile = nc_file.filename 
        for idx in range(new_shape[0]):
            outFile = '.'.join([inFile, str(idx), 'tmp'])
            execArgs = ['ncra', '-d', 'time,%i,%i'%(idx, idx+window_size), inFile, outFile]
            p = subprocess.Popen(execArgs)
            p.wait()
            
    else:
        result = np.zeros(new_shape)

        for cntr in range(new_shape[0]):
            print 'Calc for index %i' % cntr
            result[cntr] = data[(cntr-bot_edge):(cntr+top_edge)].sum(axis=0)
        result = result/float(window_size)
    
    return (result, bot_edge, top_edge)
    
