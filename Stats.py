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

def runMean(data, window_size, h5_file=None):
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

    Returns
    -------
    result: ndarray
        Running mean result of given data.
    bot_edge: int
        Number of elements removed from beginning of the time series
    top_edge: int
        Number of elements removed from the ending of the time series
    """


    #_DEBUG = False
    #useNCO = False
    #PROCESSES = 2
    
    dshape = data.shape
    top_edge = window_size/2 + 1
    bot_edge = (window_size/2) + (window_size%2) - 1
    tot_cut = top_edge + bot_edge - 1
    new_shape = list(dshape)
    new_shape[0] -= tot_cut

#    #Use netCDF operators on system instead of python based running mean (saves
#    #  memory).  For now I suggest using this method.
#    # NCO RETIRED FOR NOW
#    if useNCO:
#        inFile = 'netcdf_file.nc'
#        inFile_no_ext, file_ext = splitext(inFile)
#        file_ext = file_ext.strip('.')
#        pad_len = len(str(new_shape[0])) #calculate padding for tmp file numbers
#        fmt_str = '%s0%ii' % ('%', pad_len)
#        wildcard = '?'*pad_len
#        out_file = '.'.join([inFile_no_ext, '%imon_runmean' % window_size, file_ext])
#
#        if _DEBUG:
#            print 'inFile: %s' % inFile
#            print 'out_file: %s' % out_file
#            print 'file_ext: %s' % file_ext
#            print 'inFile_no_ext: %s' % inFile_no_ext
#
#        #Check if running mean was already done and if recalc is requested
#        #if exists(out_file) and not recalc:
#            #_f = _ncf.netcdf_file(out_file, 'r')
#            #result = _f.variables[nc_varname].data
#            #return (result, bot_edge, top_edge)
#
#        #Calculate the running average
#        pool = Pool(PROCESSES)
#        cmds = [ ['ncra',
#                  '-O', 
#                  '-d', 
#                  'time,%i,%i' % (idx, idx+window_size-1),
#                  inFile,
#                  '.'.join([inFile_no_ext, fmt_str % idx, 'tmp']) #temporary out file
#                  ] for idx in range(new_shape[0])
#                ]
#        for i,retcode in enumerate(pool.imap_unordered(call, cmds, chunksize=100)):
#             if (i % (10**(pad_len-2)) == 0):
#                 print 'Processed %i of %i' % (i, new_shape[0])
#
#        if _DEBUG: print cmds[0:2]
#        
#        #Concatenate tmp files
#        tmp_files = '.'.join([inFile_no_ext, wildcard, 'tmp'])
#        execArgs = ' '.join(['ncrcat', '-O', tmp_files, out_file])
#        p = Popen(execArgs, shell=True)
#        p.wait()
#
#        #Remove tmp files
#        execArgs = ' '.join(['rm', '-f', '.'.join([inFile_no_ext, '*.tmp'])])
#        p = Popen(execArgs, shell=True)
#        p.wait()
#            
#        _f = _ncf.netcdf_file(out_file, 'r')
#        #result = _f.variables[nc_varname].data

    if h5_file is not None:
        result = h5_file.create_carray(h5_file.root.data, 
                                       'run_mean',
                                       atom = tb.Atom.from_dtype(data.dtype),
                                       shape = new_shape,
                                       title = '12-month running mean')
    else:                                       
        result = np.zeros(new_shape, dtype=data.dtype)
        
    for cntr in xrange(new_shape[0]):
        if cntr % 100 == 0:
            print 'Calc for index %i' % cntr
        result[cntr] = (data[(cntr):(cntr+top_edge+bot_edge)].sum(axis=0) / 
                        float(window_size))
    
    return (result.read(), bot_edge, top_edge)
   

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

