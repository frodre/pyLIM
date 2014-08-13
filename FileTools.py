import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm

def fcast_corr(fcast_data, _eofs, truth, idxs, outfile):
    ntrials = fcast_data.shape[0]
    nlocs = _eofs.shape[0]
    tslice = fcast_data.shape[3]
    nfcasts = fcast_data.shape[1]
    dshape = (nlocs, ntrials*tslice)
    tmp = np.zeros( dshape )
    true_state = np.zeros( dshape )
    corrDf = None
    #atom = tbl.Atom.from_dtype(tmp.dtype)
    #filters = tbl.Filters(complib='blosc', complevel=5)
    
    for i in xrange(len(idxs)):
        ii = i*tslice
        true_state[:,ii:ii+tslice] = truth[:,idxs[i]]
    
    for tau in xrange(nfcasts):
        print 'Forecast #%i' % tau
        true_state_lead = true_state + tau
        for trial in xrange(ntrials):
            j = trial*tslice
            tmp[:,j:j+tslice] = np.dot(_eofs, fcast_data[trial, tau])
            
        df1 = pd.DataFrame(true_state_lead.T)
        df2 = pd.DataFrame(tmp.T)
        corr = df1.corrwith(df2)
        
        if corrDf is not None:
            corrDf = pd.concat([corrDf,corr], axis=1)
        else:
            corrDf = corr
            
    corrDf = corrDf.transpose()
    corrDf = corrDf.reset_index(drop=True)        
    return corrDf
    
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
    
if __name__ == "__main__":
    fcasts = np.load('forecasts.npy')
    eofs = np.load('eofs.npy')
    shp_anom = np.load('spatial_anomaly_srs.npy').T
    idxs = np.load('fcast_idxs.npy')
    result = fcast_corr(fcasts, eofs, shp_anom, idxs, 'hi')
    result.to_hdf('fcast_corr.h5', 'w')