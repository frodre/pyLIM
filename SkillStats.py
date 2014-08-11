import numpy as np
import matplotlib.pyplot as plt

#Verification Stuff
vs_trials = False
vs_time = False
vs_tau = True

if vs_tau:
    evar = np.zeros(forecast_tlim+1)
    for tau in range(forecast_tlim+1):
        tmpdata = forecasts[:,tau]
        reconstructed = np.array([
                            np.dot(eofs[loc], fcast)
                            for fcast in tmpdata
                        ])
        truth = np.array([shp_anomaly.T[loc, idxs] for idxs in fcast_idxs])
        error = reconstructed - truth
        evar[tau] = error.var()
        
    fig, ax = plt.subplots()
    ax.plot(evar)
    ax.set_title('Error Variance vs. Forecast Lead Time')
    ax.set_xlabel('Lead time (months)')
    ax.set_ylabel('Error Variance (K)')
    fig.show()

if vs_trials:
    lead_times = np.array([0, 1, 3, 6, 9, 12])*12
    anom_truth = lambda x: np.array([shp_anomaly.T[loc, (fcast_idxs[i] + x)] 
                                      for i in range(len(fcast_idxs))])
    true_var = np.array([anom_truth(tau).var() for tau in lead_times]) 
    true_mean = np.array([anom_truth(tau).mean() for tau in lead_times])
    
    fcast_var = np.zeros( (len(lead_times), num_trials) )
    fcast_mean = np.zeros( fcast_var.shape )
    
    for i in range(len(lead_times)):
        loc_fcast = np.array([np.dot(eofs[loc], fcast[lead_times[i]]) 
                              for fcast in forecasts])
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

if vs_time:
    #Variance and mean vs time sample in true space
    var_vs_time = np.array([shp_anomaly.T[loc, 0:i].var() 
                            for i in range(1,shp_anomaly.shape[0])])
    mean_vs_time = np.array([shp_anomaly.T[loc, 0:i].mean()
                            for i in range(1, shp_anomaly.shape[0])])
    varfig, varax = plt.subplots()
    varax.plot(var_vs_time, label='Variance')
    varax.plot(mean_vs_time, label = 'Mean')
    varax.axvline(x = 0, color = 'r')
    varax.axvline(x = time_dim, color = 'r')
    varax.axvline(x = forecast_tlim, color = 'y')
    varax.axvline(x = shp_anomaly.shape[0], color = 'y')
    varax.axhline(y = 0, linewidth = 1, c='k')
    varax.set_title('variance and mean w/ increasing time sample')
    varax.set_xlabel('Times included (0 to this month)')
    varax.set_ylabel('Variance & Mean (K)')
    varax.legend(loc=9)
    varfig.show()
    
    runfig, runax = plt.subplots()
    runax.plot(shp_anomaly.T[loc,:])
    runax.set_title('Time series at loc = %i (12-mon running mean)' % loc)
    runax.set_xlabel('Month')
    runax.set_ylabel('Temp Anomaly (K)')
    runfig.show()

