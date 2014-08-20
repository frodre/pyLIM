import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm

def fcast_corr(fcast_data, eof_data, obs, obs_tidxs, outfile):
    ntrials = fcast_data.shape[0]
    nlocs = eof_data.shape[0]
    tslice = fcast_data.shape[3]
    nfcasts = fcast_data.shape[1]
    dshape = (ntrials*tslice, nlocs)
    tmp = np.zeros( dshape )
    true_state = np.zeros( dshape )
    corrDf = None
    #atom = tbl.Atom.from_dtype(tmp.dtype)
    #filters = tbl.Filters(complib='blosc', complevel=5)
    
    
    for tau in xrange(nfcasts):
        print 'Forecast #%i' % tau
        for i in xrange(ntrials):
            ii = i*tslice
            true_state[ii:ii+tslice, :] = obs[(obs_tidxs[i]+tau), :]
        for trial in xrange(ntrials):
            j = trial*tslice
            tmp[j:j+tslice, :] = np.dot(eof_data, fcast_data[trial, tau]).T
            
        df1 = pd.DataFrame(true_state)
        df2 = pd.DataFrame(tmp)
        corr = df1.corrwith(df2)
        
        if corrDf is not None:
            corrDf = pd.concat([corrDf,corr], axis=1)
        else:
            corrDf = corr
            
    corrDf = corrDf.transpose()
    corrDf = corrDf.reset_index(drop=True)        
    return corrDf
    
def fcast_ce(fcast_data, eof_data, obs, obs_tidxs):
    ntrials = fcast_data.shape[0]
    nlocs = eof_data.shape[0]
    tslice = fcast_data.shape[3]
    nfcasts = fcast_data.shape[1]
    dshape = (ntrials*tslice, nlocs)
    tmp = np.zeros( dshape )
    true_state = np.zeros( dshape )
    evarMatr = np.zeros( (ntrials, nlocs) )
    #atom = tbl.Atom.from_dtype(tmp.dtype)
    #filters = tbl.Filters(complib='blosc', complevel=5)
    cvar = obs.var(axis=0)
            
    for tau in xrange(nfcasts):
        print 'Forecast #%i' % tau
        for i in xrange(ntrials):
            ii = i*tslice
            true_state[ii:ii+tslice, :] = obs[(obs_tidxs[i]+tau), :]
        for trial in xrange(ntrials):
            j = trial*tslice
            tmp[j:j+tslice, :] = np.dot(eof_data, fcast_data[trial, tau]).T
            
        error = true_state - tmp
        evar = error.var(axis=0)
        ce = 1 - (evar**2)/cvar**2
        evarMatr[i,:] = ce
                
    return evarMatr
    
def climo_var(full_obs):
    pass

    
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
    if data.min < 0:
        color = cm.bwr
    else:
        color = cm.OrRd
    m.contourf(lons, lats, data, latlon=True, cmap=color,
               vmin=0)
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
    fcasts = np.load('forecasts.npy')
    eofs = np.load('eofs.npy')
    shp_anom = np.load('spatial_anomaly_srs.npy')
    idxs = np.load('fcast_idxs.npy')
    #result = fcast_corr(fcasts, eofs, shp_anom, idxs, 'hi')
    #result.to_hdf('fcast_corr.h5', 'w')
    #result = fcast_ce(fcasts, eofs, shp_anom, idxs)
    #result.to_hdf('fcast_ce.h5', 'ce')