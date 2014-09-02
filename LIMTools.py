import os
import pandas as pd
import tables as tb
import numpy as np
import numexpr as ne
import Stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm


def fcast_corr(h5file):
    h5_datagrp = h5file.root.data
    obs = h5_datagrp.anomaly_srs.read()
    test_start_idxs = h5_datagrp.test_idxs.read()
    yrsize = h5_datagrp.fcast_bin._v_attrs.yrsize
    test_tdim = h5_datagrp.fcast_bin._v_attrs.test_tdim
    nfcasts = h5_datagrp.fcast_bin._v_nchildren
    try:
        corrs = h5file.create_carray('/stats', 'corr',
                                     atom=tb.Atom.from_dtype(obs.dtype),
                                     shape=(nfcasts, obs.shape[1]),
                                     title="Local Anomaly Correlations",
                                     createparents=True)
    except tb.NodeError:
        h5file.remove_node(h5file.root.stats, 'corr')
        corrs = h5file.create_carray('/stats', 'corr',
                                     atom=tb.Atom.from_dtype(obs.dtype),
                                     shape=(nfcasts, obs.shape[1]),
                                     title="Local Anomaly Correlations",
                                     createparents=True)
    except tb.FileModeError:
        corrs = np.zeros((nfcasts, obs.shape[1]))
    
    fcasts = h5file.list_nodes(h5_datagrp.fcast_bin)
    for i,fcast in enumerate(fcasts):
        print 'Calculating LCA: %i yr fcast' % i
        compiled_obs = build_obs(obs, test_start_idxs, i*yrsize, test_tdim)
        corrs[i] = st.calcLCA(fcast.read(), compiled_obs)
    
    return corrs
    
def fcast_ce(h5file): 
    h5_datagrp = h5file.root.data
    obs = h5_datagrp.anomaly_srs.read()
    test_start_idxs = h5_datagrp.test_idxs.read()
    yrsize = h5_datagrp.fcast_bin._v_attrs.yrsize
    test_tdim = h5_datagrp.fcast_bin._v_attrs.test_tdim
    nfcasts = h5_datagrp.fcast_bin._v_nchildren
    try:
        ces = h5file.create_carray('/stats', 'ce',
                                     atom=tb.Atom.from_dtype(obs.dtype),
                                     shape=(nfcasts, obs.shape[1]),
                                     title="Coefficient of Efficiency",
                                     createparents=True)
    except tb.NodeError:
        h5file.remove_node(h5file.root.stats, 'ce')
        ces = h5file.create_carray('/stats', 'ce',
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
        ces[i] = st.calcLCA(fcast.read(), compiled_obs)
    
    return ces
    
def climo(data, yrsize):
    old_shp = data.shape
    new_shp = (old_shp[0]/yrsize, yrsize, old_shp[1])
    climo = data.reshape(new_shp).sum(axis=0)/float(new_shp[0])
    return climo
    
def build_obs(obs, start_idxs, tau, test_dim, h5f=None):     
    obs_data = np.zeros( (len(start_idxs)*test_dim, obs.shape[1]),
                          dtype = obs.dtype)
    
    for i,idx in enumerate(start_idxs):
        start = i*test_dim
        end = i*test_dim + test_dim
        obs_data[start:end] = obs[idx+tau:idx+tau+test_dim]
        
    return obs_data

    
####  PLOTTING FUNCTIONS  ####
    
def plot_corrdata(lats, lons, data, title, outfile):
    contourlev = np.concatenate(([-1],np.linspace(0,1,11)))
    cbticks = np.linspace(0,1,11)
    plt.close('all')
    m = Basemap(projection='gall', llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=0, urcrnrlon=360, resolution='c')
    m.drawcoastlines()
    color = cm.OrRd
    color.set_under('#9acce5')
    m.contourf(lons, lats, data, latlon=True, cmap=color,
               vmin=0, levels = contourlev)
    m.colorbar(ticks=cbticks)
    plt.title(title)
    plt.savefig(outfile, format='png')

def plot_cedata(lats, lons, data, title, outfile):
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
    plt.savefig(outfile, format='png')
    
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
    
def plot_vstrials(fcast_data, eof_data, obs, obs_tidxs, num_trials, loc):
    lead_times = np.array([0, 1, 3, 6, 9, 12])*12
    anom_truth = lambda x: np.array([obs.T[loc, (obs_tidxs[i] + x)] 
                                      for i in range(len(obs_tidxs))])
    true_var = np.array([anom_truth(tau).var() for tau in lead_times]) 
    true_mean = np.array([anom_truth(tau).mean() for tau in lead_times])
    
    fcast_var = np.zeros( (len(lead_times), num_trials) )
    fcast_mean = np.zeros( fcast_var.shape )
    
    for i in range(len(lead_times)):
        loc_fcast = np.array([np.dot(eof_data[loc], fcast[lead_times[i]]) 
                              for fcast in fcast_data])
        varis = np.zeros( len(loc_fcast) )
        means = np.zeros( varis.shape )
        for j in range(len(loc_fcast)):
            varis[j] = loc_fcast[0:j+1].var() 
            means[j] = loc_fcast[0:j+1].mean()
        fcast_var[i,:] = varis
        fcast_mean[i,:] = means
    
    fig, ax = plt.subplots(2,1, sharex=True)
    x = range(fcast_var.shape[1])
    
    for i,var in enumerate(fcast_var):
        ax[0].plot(x, var, linewidth=2, label='Lead Time = %i yr' % (lead_times[i]/12))
        ax[0].legend()
        ax[1].plot(x, fcast_mean[i], linewidth=2)
    
    for line, var in zip(ax[0].get_lines(), true_var):
        ax[0].axhline(y=var, linestyle = '--', color = line.get_color())
    
    ax[0].set_title('Forecast Variance & Mean vs. # Trials (Single Gridpoint)')
    ax[1].set_xlabel('Trial #')
    ax[0].set_ylabel('Variance (K)')
    ax[1].set_ylabel('Mean (K)')
    fig.show()
    

if __name__ == "__main__":
    if os.name == 'nt':
        outfile = 'G:\Hakim Research\pyLIM\LIM_data.h5'
    else:
        outfile = '/home/chaos2/wperkins/data/pyLIM/LIM_data.h5'
    h5file = tb.open_file(outfile, mode='a')
    try:
        #corr = fcast_corr(h5file)
        ce = fcast_ce(h5file)
    finally:
        h5file.close()
    
        