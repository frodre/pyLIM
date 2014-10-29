import os
import pandas as pd
import tables as tb
import numpy as np
import numexpr as ne
import Stats as st
import matplotlib.pyplot as plt
import scipy.io.netcdf as ncf
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

#custom colormap information, trying to reproduce Newman
lb = tuple(np.array([150, 230, 255])/255.0)
w = (1.0, 1.0, 1.0)
yl = tuple(np.array([243, 237, 48])/255.0)
rd = tuple(np.array([255, 50, 0])/255.0)
dk = tuple(np.array([110,0,0])/255.0)

cdict = {'red':     ((0.0, lb[0], lb[0]),
                     (0.1, w[0], w[0]),
                     (0.3, yl[0], yl[0]),
                     (0.7, rd[0], rd[0]),
                     (1.0, dk[0], dk[0])),

         'green':   ((0.0, lb[1], lb[1]),
                     (0.2, w[1], w[1]),
                     (0.4, yl[1], yl[1]),
                     (0.7, rd[1], rd[1]),
                     (1.0, dk[1], dk[1])),

         'blue':    ((0.0, lb[2], lb[2]),
                     (0.2, w[2], w[2]),
                     (0.4, yl[2], yl[2]),
                     (0.7, rd[2], rd[2]),
                     (1.0, dk[2], dk[2]))}

newm = LinearSegmentedColormap('newman', cdict)

def fcast_corr(h5file):
    leaf_name = 'corr_check'
    h5_datagrp = h5file.root.data
    obs = h5_datagrp.anomaly_srs.read()
    test_start_idxs = h5_datagrp.test_idxs.read()
    yrsize = h5_datagrp.fcast_bin._v_attrs.yrsize
    test_tdim = h5_datagrp.fcast_bin._v_attrs.test_tdim
    nfcasts = h5_datagrp.fcast_bin._v_nchildren
    try:
        corrs = h5file.create_carray('/stats', leaf_name,
                                     atom=tb.Atom.from_dtype(obs.dtype),
                                     shape=(nfcasts, obs.shape[1]),
                                     title="Local Anomaly Correlations",
                                     createparents=True)
    except tb.NodeError:
        h5file.remove_node(h5file.root.stats, leaf_name)
        corrs = h5file.create_carray('/stats', leaf_name,
                                     atom=tb.Atom.from_dtype(obs.dtype),
                                     shape=(nfcasts, obs.shape[1]),
                                     title="Local Anomaly Correlations",
                                     createparents=True)
    except tb.FileModeError:
        corrs = np.zeros((nfcasts, obs.shape[1]))
    
    fcasts = h5file.list_nodes(h5_datagrp.fcast_bin)
    for i,fcast in enumerate(fcasts):
        print 'Calculating LAC: %i yr fcast' % i
        compiled_obs = build_obs(obs, test_start_idxs, i*yrsize, test_tdim)
        corrs[i] = st.calcLCA(fcast.read(), compiled_obs)

    return corrs
    
def fcast_ce(h5file):
    leaf_name = 'ce_check' 
    h5_datagrp = h5file.root.data
    obs = h5_datagrp.anomaly_srs.read()
    test_start_idxs = h5_datagrp.test_idxs.read()
    yrsize = h5_datagrp.fcast_bin._v_attrs.yrsize
    test_tdim = h5_datagrp.fcast_bin._v_attrs.test_tdim
    nfcasts = h5_datagrp.fcast_bin._v_nchildren
    try:
        ces = h5file.create_carray('/stats', leaf_name,
                                     atom=tb.Atom.from_dtype(obs.dtype),
                                     shape=(nfcasts, obs.shape[1]),
                                     title="Coefficient of Efficiency",
                                     createparents=True)
    except tb.NodeError:
        h5file.remove_node(h5file.root.stats, leaf_name)
        ces = h5file.create_carray('/stats', leaf_name,
                                     atom=tb.Atom.from_dtype(obs.dtype),
                                     shape=(nfcasts, obs.shape[1]),
                                     title="Coefficient of Efficiency",
                                     createparents=True)
    except tb.FileModeError:
        ces = np.zeros((nfcasts, obs.shape[1]))

    fcasts = h5file.list_nodes(h5_datagrp.fcast_bin)
    for i,fcast in enumerate(fcasts):
        print 'Calculating CE: %i yr fcast' % i
        compiled_obs = build_obs(obs, test_start_idxs, i*yrsize, test_tdim)
        ces[i] = st.calcCE(fcast.read(), compiled_obs, obs)
        
        #ce_tmp = np.zeros( obs.shape[1] )
        #for j, start_idx in enumerate(test_start_idxs):
        #    compiled_obs = build_obs(obs, [start_idx], i*yrsize, test_tdim)
        #    start = j*test_tdim
        #    end = j*test_tdim + test_tdim
        #    fcast_chunk = fcast[start:end]
        #    ce_tmp += st.calcCE(fcast_chunk, compiled_obs, obs.var(axis=0))
        #
        #ces[i] = ce_tmp/len(test_start_idxs)
        #print "Averaging, len %i, j %i" % (len(test_start_idxs), j+1)
    
    return ces
    
def calc_anomaly(data, yrsize):
    old_shp = data.shape
    new_shp = (old_shp[0]/yrsize, yrsize, old_shp[1])
    climo = data.reshape(new_shp).sum(axis=0)/float(new_shp[0])
    anomaly = data.reshape(new_shp) - climo
    return anomaly.reshape(old_shp)
    
def build_obs(obs, start_idxs, tau, test_dim, h5f=None):     
    obs_data = np.zeros( (len(start_idxs)*test_dim, obs.shape[1]),
                          dtype = obs.dtype)
    
    for i,idx in enumerate(start_idxs):
        start = i*test_dim
        end = i*test_dim + test_dim
        obs_data[start:end] = obs[idx+tau:idx+tau+test_dim]
        
    return obs_data

def area_wgt(data, lats):
    assert(data.shape[-1] == lats.shape[-1])
    scale = np.sqrt(np.cos(np.radians(lats)))
    return data * scale
    
def load_landsea_mask(maskfile, tile_len):
    f_mask = ncf.netcdf_file(maskfile)
    land_mask = f_mask.variables['land']
    
    try:
        sf = land_mask.scale_factor
        offset = land_mask.add_offset
        land_mask = land_mask.data*sf + offset
    except AttributeError:
        land_mask = land_mask.data
        
    land_mask = land_mask.squeeze().astype(np.int16).flatten()
    sea_mask = np.logical_not(land_mask)
    
    tiled_landmask = np.repeat(np.expand_dims(land_mask, 0),
                               tile_len,
                               axis=0 )
    tiled_seamask = np.repeat(np.expand_dims(sea_mask, 0),
                              tile_len,
                              axis=0)
    return (tiled_landmask, tiled_seamask)
    
####  PLOTTING FUNCTIONS  ####
    
def plot_corrdata(lats, lons, data, title, outfile=None):
    plt.clf()
    contourlev = np.concatenate(([-1],np.linspace(0,1,11)))
    cbticks = np.linspace(0,1,11)
    plt.close('all')
    m = Basemap(projection='gall', llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=0, urcrnrlon=360, resolution='c')
    m.drawcoastlines()
    color = newm
    color.set_under('#9acce5')
    m.contourf(lons, lats, data, latlon=True, cmap=color,
               vmin=0, levels = contourlev)
    m.colorbar(ticks=cbticks)
    plt.title(title)
    
    if outfile is not None:
        plt.savefig(outfile, format='png')
    else:
        plt.show()

def plot_cedata(lats, lons, data, title, outfile=None):
    #contourlev = np.concatenate(([-1],np.linspace(0,1,11)))
    #cbticks = np.linspace(0,1,11)
    plt.close('all')
    m = Basemap(projection='gall', llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=0, urcrnrlon=360, resolution='c')
    m.drawcoastlines()
    
    contourlev = np.linspace(0,1,11)
    
    if data.min() < 0:
        color = cm.bwr
        neglev = np.linspace(-1, 0, 11)
        contourlev = np.concatenate((neglev, contourlev))
    else:
        color = cm.OrRd
        
    m.pcolor(lons, lats, data, latlon=True, cmap=color, vmin=-1, vmax=1)
    m.colorbar()
    plt.title(title)
    if outfile is not None:
        plt.savefig(outfile, format='png')
    else:
        plt.show()
    
def plot_spatial(lats, lons, data, title, outfile=None):
    """
    Method for basic spatial data plots.  Uses diverging color scheme, so 
    current implementation is best for anomaly data.  Created initially just
    to plot spatial EOFs
    
    Parameters
    ----------
    lats: ndarray
        MxN matrix of latitude values
    lons: ndarray
        MxN matrix of longitude values
    data: ndarray
        MxN matrix of spatial data to plot
    title: str
        Title string for the plot
    outfile: str
        Filename to save the png image as
    """
    plt.clf()
    plt_range = np.max(np.abs(data))
    m = Basemap(projection='gall', llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=0, urcrnrlon=360, resolution='c')
    m.drawcoastlines()
    color = cm.bwr
    m.pcolor(lons, lats, data, latlon=True, cmap=color, vmin=-plt_range,
             vmax = plt_range)
    m.colorbar()
    
    plt.title(title)
    if outfile is not None:
        plt.savefig(outfile, format='png')
    else:
        plt.show()
    
def plot_vstau(fcast_data, eof_data, obs, obs_tidxs, loc, title, outfile):
    fcast_tlim = fcast_data.shape[1]
    evar = np.zeros(fcast_tlim)
    for tau in range(fcast_tlim):
        tmpdata = fcast_data[:,tau]
        reconstructed = np.array([
                            np.dot(eof_data[loc], fcast)
                            for fcast in tmpdata
                        ])
        truth = np.array([obs.T[loc, idxs] for idxs in obs_tidxs])
        error = reconstructed - truth
        evar[tau] = error.var()
        
    fig, ax = plt.subplots()
    ax.plot(evar)
    ax.set_title(title)
    ax.set_xlabel('Lead time (months)')
    ax.set_ylabel('Error Variance (K)')
    fig.savefig(outfile, format='png')
    
def plot_vstime(obs, loc):
    #Variance and mean vs time sample in true space
    var_vs_time = np.array([obs.T[loc, 0:i].var() 
                            for i in range(1,obs.shape[0])])
    mean_vs_time = np.array([obs.T[loc, 0:i].mean()
                            for i in range(1, obs.shape[0])])
    varfig, varax = plt.subplots()
    varax.plot(var_vs_time, label='Variance')
    varax.plot(mean_vs_time, label = 'Mean')
    varax.axvline(x = 0, color = 'r')
    #varax.axvline(x = time_dim, color = 'r')
    #varax.axvline(x = forecast_tlim, color = 'y')
    #varax.axvline(x = shp_anomaly.shape[0], color = 'y')
    varax.axhline(y = 0, linewidth = 1, c='k')
    varax.set_title('variance and mean w/ increasing time sample')
    varax.set_xlabel('Times included (0 to this month)')
    varax.set_ylabel('Variance & Mean (K)')
    varax.legend(loc=9)
    varfig.show()
    
    runfig, runax = plt.subplots()
    runax.plot(obs.T[loc,:])
    runax.set_title('Time series at loc = %i (12-mon running mean)' % loc)
    runax.set_xlabel('Month')
    runax.set_ylabel('Temp Anomaly (K)')
    runfig.show()
    
def plot_vstrials(fcast_data, obs, test_tidxs, test_tdim, tau, loc):
    num_trials = fcast_data.shape[0]/test_tdim
    anom_truth = build_obs(obs, test_tidxs, tau, test_tdim)
    loc_tru_var = anom_truth[:,loc].var()
    print loc_tru_var
    loc_tru_mean = anom_truth[:,loc].mean()
    
    fcast_var = np.zeros( num_trials )
    fcast_mean = np.zeros( num_trials )
    
    for i in xrange(num_trials):
        end = i*test_tdim + test_tdim
        fcast_var[i] = fcast_data[0:end, loc].var()
        fcast_mean[i] = fcast_data[0:end, loc].mean()
    
    fig, ax = plt.subplots(2,1, sharex=True)
    
    
    ax[0].plot(fcast_var, color='b', linewidth=2, label='Fcast Var')
    ax[0].axhline(loc_tru_var, xmin=0, xmax=num_trials,
                  linestyle='--', color='k', label='True Var')
    ax[0].legend(loc=4)
    ax[1].plot(fcast_mean, linewidth=2, label='Fcast Mean')
    ax[1].axhline(loc_tru_mean, xmin=0, xmax=num_trials,
                  linestyle='--', color='k', label='True Mean')
    ax[1].legend()
    
    # Interesting case of line matching below here
    #for line, var in zip(ax[0].get_lines(), true_var):
    #    ax[0].axhline(y=var, linestyle = '--', color = line.get_color())
    
    ax[0].set_title('Forecast Variance & Mean vs. # Trials (Single Gridpoint)'
                     ' Tau = %i' % tau)
    ax[0].set_ylim(0,0.8)
    ax[1].set_xlabel('Trial #')
    ax[0].set_ylabel('Variance (K)')
    ax[1].set_ylabel('Mean (K)')
    fig.show()
    

if __name__ == "__main__":
    if os.name == 'nt':
        outfile = 'G:\Hakim Research\pyLIM\LIM_data.h5'
        #outfile = 'G:\Hakim Research\pyLIM\Trend_LIM_data.h5'
    else:
        #outfile = '/home/chaos2/wperkins/data/pyLIM/LIM_data.h5'
        #outfile = '/home/chaos2/wperkins/data/pyLIM/Detrend_LIM_data.h5'
        outfile = '/home/chaos2/wperkins/data/pyLIM/Trended_sepEOFs_LIM_data.h5'
    h5file = tb.open_file(outfile, mode='a')
    try:
        corr = fcast_corr(h5file)
        ce = fcast_ce(h5file)
    finally:
        h5file.close()
    
        