"""
Toolbox for statistical methods.

"""

import numpy as np
import scipy.io.netcdf as ncf
from subprocess import call, Popen
from multiprocessing import Pool
from os.path import exists, join, splitext
from scipy.sparse.linalg import eigs

def runMean(nc_file, nc_varname, window_size, num_procs=None, useNCO=False):
    """
    A function for calculating the running mean on data.

    Parameters
    ----------
    nc_file: scipy.io.netcdf.netcdf_file
        NetCDF file pointer for desired dataset.
    nc_varname: str
        Dictionary key of data to be averaged
    window_size: int
        Size of the window to compute the running mean over.
    num_procs: int, optional
        Number of processes for calculating the running mean.  
        The default value is 8. Only valid for useNCO=True.
    useNCO: bool, optional
        If the calculation is over a large dataset where memory use
        may be an issue, the use of netCDF operators is likely a good
        idea.  

    Returns
    -------
    result: ndarray
        Running mean result of given data.
    bot_edge: int
        Number of elements removed from beginning of the time series
    top_edge: int
        Number of elements removed from the ending of the time series
    """


    _DEBUG = False
    if num_procs is not None:
        PROCESSES = num_procs
    else: PROCESSES = 8

    #Load NetCDF data.  Try with scale_factor and offset first.
    nc_var = nc_file.variables[nc_varname]
    try:
        sf = nc_var.scale_factor
        offset = nc_var.add_offset
        data = nc_var.data * sf + offset
    except AttributeError as e:
        data = nc_var.data
      
    dshape = data.shape
    top_edge = window_size/2 + 1
    bot_edge = (window_size/2) + (window_size%2) - 1
    tot_cut = top_edge + bot_edge  #TODO: fix off by one for even numbers
    new_shape = list(dshape)
    new_shape[0] -= tot_cut

    #Use netCDF operators on system instead of python based running mean (saves
    #  memory).  For now I suggest using this method.
    if useNCO:
        inFile = nc_file.filename
        inFile_no_ext, file_ext = splitext(inFile)
        file_ext = file_ext.strip('.')
        pad_len = len(str(new_shape[0])) #calculate padding for tmp file numbers
        fmt_str = '%s0%ii' % ('%', pad_len)
        wildcard = '?'*pad_len
        out_file = '.'.join([inFile_no_ext, '%imon_runmean' % window_size, file_ext])

        if _DEBUG:
            print 'inFile: %s' % inFile
            print 'out_file: %s' % out_file
            print 'file_ext: %s' % file_ext
            print 'inFile_no_ext: %s' % inFile_no_ext

        #Check if running mean was already done, prompt to recalculate
        if exists(out_file):
            user_in = ''
            while user_in != 'y' and user_in != 'n':
                user_in = raw_input('Would you like to recalculate the running' 
                                     ' mean?[y or n]: ').rstrip()
            if user_in == 'n':
                f = ncf.netcdf_file(out_file, 'r')
                result = f.variables[nc_varname].data
                return (result, bot_edge, top_edge)

        #Calculate the running average
        pool = Pool(PROCESSES)
        cmds = [ ['ncra',
                  '-O', 
                  '-d', 
                  'time,%i,%i' % (idx, idx+window_size-1),
                  inFile,
                  '.'.join([inFile_no_ext, fmt_str % idx, 'tmp']) #temporary out file
                  ] for idx in range(new_shape[0])
                ]
        for i,retcode in enumerate(pool.imap_unordered(call, cmds, chunksize=100)):
             if (i % (10**(pad_len-2)) == 0):
                 print 'Processed %i of %i' % (i, new_shape[0])

        if _DEBUG: print cmds[0:2]
        
        #Concatenate tmp files
        tmp_files = '.'.join([inFile_no_ext, wildcard, 'tmp'])
        execArgs = ' '.join(['ncrcat', '-O', tmp_files, out_file])
        p = Popen(execArgs, shell=True)
        p.wait()

        #Remove tmp files
        execArgs = ' '.join(['rm', '-f', '.'.join([inFile_no_ext, '*.tmp'])])
        p = Popen(execArgs, shell=True)
        p.wait()
            
        f = ncf.netcdf_file(out_file, 'r')
        result = f.variables[nc_varname].data

    else:
        result = np.zeros(new_shape)

        for cntr in range(new_shape[0]):
            if cntr % 100 == 0:
                print 'Calc for index %i' % cntr
            result[cntr] = data[(cntr):(cntr+top_edge+bot_edge)].sum(axis=0)
        result = result/float(window_size)
    
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
        cov = np.linalg.cov(data)
        eig_vals, eofs = eigs(cov, k=num_eigs)
        tot_var = eig_vals.real.sum()/cov.trace()
        cov = np.linalg.cov(data.T)
        pcs, trash = eigs(cov, k=num_eigs)

    if retPCs:
        return (eofs[:,0:num_eigs], eig_vals[0:num_eigs], tot_var, pcs[0:num_eigs])
    else:
        return (eofs[:,0:num_eigs], eig_vals[0:num_eigs], tot_var)

