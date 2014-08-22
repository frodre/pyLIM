"""
Toolbox for statistical methods.

"""

import numpy as np
import tables as tb
import scipy.io.netcdf as _ncf
from subprocess import call, Popen
from multiprocessing import Pool
from os.path import exists, splitext
from scipy.sparse.linalg import eigs

def runMean(data, window_size, h5_file=None, h5_parent=None):
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
    top_edge = window_size/2 + 1
    bot_edge = (window_size/2) + (window_size%2) - 1
    tot_cut = top_edge + bot_edge - 1
    new_shape = list(dshape)
    new_shape[0] -= tot_cut

    if h5_file is not None:
        ish5 = True
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
        ish5 = False                                       
        result = np.zeros(new_shape, dtype=data.dtype)
        
    for cntr in xrange(new_shape[0]):
        if cntr % 100 == 0:
            print 'Calc for index %i' % cntr
        result[cntr] = (data[(cntr):(cntr+top_edge+bot_edge)].sum(axis=0) / 
                        float(window_size))
                        
    if ish5:
        result = result.read()
    
    return (result, bot_edge, top_edge)
   

def calcEOF(data, num_eigs, useSVD = True, retPCs = False):
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
    useSVD: bool, optional
        Use singular value decomposition to calculate the EOFs
    retPCs: bool, optional
        Return principal component matrix along with EOFs

    Returns
    -------

    """

    if useSVD:
        eofs, E, pcs = np.linalg.svd(data, full_matrices=False)
        eig_vals = (E ** 2) / (len(E) - 1.)
        tot_var = (eig_vals[0:num_eigs].sum()) / eig_vals.sum()

    else:
        cov = np.cov(data)
        eig_vals, eofs = eigs(cov, k=num_eigs)
        tot_var = eig_vals.real.sum()/cov.trace()
        #pcs, trash = eigs(cov, k=num_eigs)

    #if retPCs:
    #    return (eofs[:,0:num_eigs], eig_vals[0:num_eigs], tot_var, pcs[0:num_eigs])
    #else:
    return (eofs[:,0:num_eigs], eig_vals[0:num_eigs], tot_var)

