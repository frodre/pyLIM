"""
Various utility tools for the LIM package.

Author: Andre Perkins
"""

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from itertools import izip


import Stats as St
import DataTools as Dt


""" Methods to help with common LIM tasks."""

#custom colormap information, trying to reproduce Newman colorbar
lb = tuple(np.array([150, 230, 255])/255.0)
w = (1.0, 1.0, 1.0)
yl = tuple(np.array([243, 237, 48])/255.0)
rd = tuple(np.array([255, 50, 0])/255.0)
dk = tuple(np.array([110, 0, 0])/255.0)

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


def area_wgt(data, lats):
    """
    Area weighting function for an input dataset.  Assumes same length trailing
    dimension of two input parameters.

    Parameters
    ----------
    data: ndarray
        Input dataset to be area weighted.  Spatial dimensions should be last.
    lats: ndarray
        Latitude array corresponding to the data.  Right now trailing dimensions
        must match. (allows for broadcasting)

    Returns
    -------
    ndarray
        Area weighted data.
    """
    assert(data.shape[-1] == lats.shape[-1])
    scale = np.sqrt(np.cos(np.radians(lats)))
    return data * scale


def build_trial_fcast(fcast_trials, eofs):
    """
    Build forecast dataset from trials and EOFs.  This stacks the forecast
    trials sequentially along the temporal dimension.  I.e. a forecast_trials
    with dimensions of  trials x time samples x spatial (if it's been converted
    out of eof space)  would be built with dimensions of
    (trials * time samples) x spatial.

    Parameters
    ----------
    fcast_trials: ndarray
        LIM forecast output in EOF space.
        Dimensions of trials x num_eigs x num_samples
    eofs: ndarray
        Empirical orthogonal functions corresponding to each trial.
        Dimensions of trials x spatial x num_eigs

    Returns
    -------
    ndarray
        Forecast trials stacked along the temporal dimensions and converted to
        physical space.
    """

    t_shp = fcast_trials.shape
    dat_shp = [t_shp[0]*t_shp[-1], eofs.shape[1]]
    phys_fcast = np.zeros(dat_shp, dtype=fcast_trials.dtype)

    for i, (trial, eof) in enumerate(izip(fcast_trials, eofs)):
        i0 = i*t_shp[-1]  # i * (test time dimension)
        ie = i*t_shp[-1] + t_shp[-1]

        phys_fcast[i0:ie] = np.dot(trial.T, eof.T)

    return phys_fcast


def build_trial_fcast_from_h5(h5file, tau):
    """
    Build forecast dataset from trials and EOFs that are read
    from the input HDF5 LIM forecast file.

    Parameters
    ----------
    h5file: tables.File
        Pytables HDF5 file holding LIM forecast output.
    tau: int
        Forecast lead time to build forecast dataset from

    Returns
    ndarray
        Forecast trials stacked along the temporal dimensions and coverted to
        physical space.
    """
    assert(h5file is not None and type(h5file) == tb.File)
    try:
        fcast_trials = h5file.list_nodes(h5file.root.data.fcast_bin)[tau].read()
        eofs = h5file.root.data.eofs.read()
    except tb.NodeError as e:
        raise type(e)(e.message + ' Returning without finishing operation...')

    return build_trial_fcast(fcast_trials, eofs)


def build_trial_obs(obs, start_idxs, tau, test_tdim):
    """
    Build observation dataset to compare to a forecast dataset built by
    the build_trial_fcast...  methods.

    Parameters
    ----------
    obs: ndaray
        Observations to build from.
        Dimensions of time x space
    start_idxs: list
        List of indices corresponding to trial start times in observations.
    tau: int
        Lead time of the forecast to which the observations are being
        compared.
    test_tdim: int
        Length of time sample for each trial.

    Returns
    -------
    ndarray
        Observations corresponding to each forecast trial stacked along the
        temporal dimension.
    """
    dat_shp = [len(start_idxs)*test_tdim, obs.shape[-1]]
    obs_data = np.zeros(dat_shp, dtype=obs.dtype)

    for i, idx in enumerate(start_idxs):
        i0 = i*test_tdim
        ie = i*test_tdim + test_tdim

        obs_data[i0:ie] = obs[(idx+tau):(idx+tau+test_tdim)]

    return obs_data


def build_trial_obs_from_h5(h5file, tau):
    """
    Build observation dataset from HDF5 file to compare to a forecast
    datset built by the build_trial_fcast... methods.

    Parameters
    ----------
    h5file: tables.File
        Pytables HDF5 file holding LIM observation data.
    tau: int
        Lead time of the forecast to which the observations are being
        compared.

    Returns
    -------
    ndarray
        Observations corresponding to each forecast tiral stacked along the
        temporal dimension.
    """
    assert(h5file is not None and type(h5file) == tb.File)

    try:
        obs = h5file.root.data.anomaly_srs[:]
        start_idxs = h5file.root.data.test_start_idxs[:]
        yrsize = h5file.root.data._v_attrs.yrsize
        test_tdim = h5file.root.data._v_attrs.test_tdim
    except tb.NodeError as e:
        raise type(e)(e.message + ' Returning without finishing operation...')

    tau_months = tau*yrsize

    return build_trial_obs(obs, start_idxs, tau_months, test_tdim)


# TODO: Implement correct significance testing
def calc_corr_signif(fcast, obs):
    """
    Calculate local anomaly correlation along with 95% significance.
    """
    assert(fcast.shape == obs.shape)

    corr_neff = St.calc_n_eff(fcast, obs)
    corr = St.calc_lac(fcast, obs)

    signif = np.empty_like(corr, dtype=np.bool)

    if True in (abs(corr) < 0.5):
        g_idx = np.where(abs(corr) < 0.5)
        gen_2std = 2./np.sqrt(corr_neff[g_idx])
        signif[g_idx] = (abs(corr[g_idx]) - gen_2std) > 0

    if True in (abs(corr) >= 0.5):
        z_idx = np.where(abs(corr) >= 0.5)
        z = 1./2 * np.log((1 + corr[z_idx]) / (1 - corr[z_idx]))
        z_2std = 2. / np.sqrt(corr_neff[z_idx] - 3)
        signif[z_idx] = (abs(z) - z_2std) > 0

    # if True in ((corr_neff <= 3) & (abs(corr) >= 0.5)):
    #     assert(False) # I have to figure out how to implement T_Test
    #     trow = np.where((corr_neff <= 20) & (corr >= 0.5))

    return corr, signif


# TODO: Fix CE calculation for comparisons and add reference
def fcast_ce(h5file):
    """
    Calculate the coefficient of efficiency for a LIM forecast at every point.

    Parameters
    ----------
    h5file: tables.File
        PyTables HDF5 file containing LIM forecast data.  All necessary
        variables are loaded from this file.

    Returns
    -------
    ndarray
        Coefficient of efficiency for each forecast lead time
        (compared against observations)

    References
    ----------
    """
    node_name = 'ce'
    parent = '/stats'

    assert(h5file is not None and type(h5file) == tb.File)

    # Load necessary data
    try:
        obs = h5file.root.data.anomaly_srs[:]
        test_start_idxs = h5file.root.data.test_start_idxs[:]
        fcast_times = h5file.root.data.fcast_times[:]
        fcasts = h5file.list_nodes(h5file.root.data.fcast_bin)
        eofs = h5file.root.data.eofs[:]
        yrsize = h5file.root.data._v_attrs.yrsize
        test_tdim = h5file.root.data._v_attrs.test_tdim
    except tb.NodeError as e:
        raise type(e)(e.message + ' Returning without finishing operation...')

    # Create output location in h5file
    atom = tb.Atom.from_dtype(obs.dtype)
    ce_shp = [len(fcast_times), obs.shape[1]]
    try:
        ce_out = Dt.empty_hdf5_carray(h5file, parent, node_name, atom, ce_shp,
                                      title="Spatial Coefficient of Efficiency",
                                      createparents=True)
    except tb.FileModeError:
        ce_out = np.zeros(ce_shp)

    # Calculate CE
    for i, lead in enumerate(fcast_times):
        print 'Calculating CE: %i yr fcast' % lead
        compiled_obs = build_trial_obs(obs, test_start_idxs, lead*yrsize, test_tdim)
        data = fcasts[i].read()
        for j, trial in enumerate(data):
            phys_fcast = np.dot(trial.T, eofs[j].T)
            ce_out[i] += St.calc_ce(phys_fcast, compiled_obs[j], obs)

        ce_out[i] /= float(len(data))

    return ce_out


def fcast_corr(h5file):
    """
    Calculate the local anomaly correlation for a LIM forecast at every point.

    Parameters
    ----------
    h5file: tables.File
        PyTables HDF5 file containing LIM forecast data.  All necessary
        variables are loaded from this file.

    Returns
    -------
    ndarray
        Local anomaly correlation for each forecast lead time at all points.
        (compared against observations)
    """
    node_name = 'corr'
    parent = '/stats'

    assert(h5file is not None and type(h5file) == tb.File)

    # Load necessary data
    try:
        obs = h5file.root.data.anomaly_srs[:]
        test_start_idxs = h5file.root.data.test_start_idxs[:]
        fcast_times = h5file.root.data.fcast_times[:]
        fcasts = h5file.list_nodes(h5file.root.data.fcast_bin)
        eofs = h5file.root.data.eofs[:]
        yrsize = h5file.root.data._v_attrs.yrsize
        test_tdim = h5file.root.data._v_attrs.test_tdim
    except tb.NodeError as e:
        raise type(e)(e.message + ' Returning without finishing operation...')

    # Create output location in h5file
    atom = tb.Atom.from_dtype(obs.dtype)
    corr_shp = [len(fcast_times), obs.shape[1]]

    try:
        corr_out = Dt.empty_hdf5_carray(h5file, parent, node_name, atom,
                                        corr_shp,
                                        title="Spatial Correlation",
                                        createparents=True)
    except tb.FileModeError:
        corr_out = np.zeros(corr_shp)

    # Calculate LAC
    for i, lead in enumerate(fcast_times):
        print 'Calculating Correlation: %i yr fcast' % lead
        compiled_obs = build_trial_obs(obs, test_start_idxs, lead*yrsize, test_tdim)
        data = fcasts[i].read()
        phys_fcast = build_trial_fcast(data, eofs)

        # for j, trial in enumerate(data):
        #     phys_fcast = np.dot(trial.T, eofs[j].T)
        #     corr_out[i] += St.calc_ce(phys_fcast, compiled_obs[j], obs)

        corr_out[i] = St.calc_lac(phys_fcast, compiled_obs)

    return corr_out
    
####  PLOTTING FUNCTIONS  ####


def plot_corrdata(lats, lons, data, title, outfile=None, signif=None):
    """
    Plot local anomaly correlation data.  Lats, lons, and data should have the
    same temporal dimensions.  Lats and lons should denote gridbox edges.
    e.g. data has spatial dimensions of M (lats) x N (lons) then lats and lons
    should have dimensions of M+1 x N+1.  Only uses Newman colarmap.

    Parameters
    ----------
    lats: ndarray
        Latitude array (gridbox edges)
    lons: ndarray
        Longitude array (gridbox edges)
    data: ndarray
        Correlation data
    title: str
        Plot title
    signif: ndarray, optional
        Boolean ndarray denoting locations that do not pass 95% significance.
    """
    plt.clf()

    # Colorbar ticks
    contourlev = np.concatenate(([-1], np.linspace(0, 1, 11)))
    cbticks = np.linspace(0, 1, 11)

    m = Basemap(projection='gall', llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=0, urcrnrlon=360, resolution='c')
    m.drawcoastlines()
    color = newm
    color.set_under('#9acce5')

    m.contourf(lons, lats, data, latlon=True, cmap=color,
               vmin=0, levels=contourlev)
    m.colorbar(ticks=cbticks)

    if signif is not None:
        ridx, cidx = np.where(np.logical_not(signif))
        lons = lons[ridx, cidx].flatten()
        lats = lats[ridx, cidx].flatten()
        x, y = m(lons, lats)
        m.scatter(x, y, s=2, c='k', marker='o', alpha=0.5)

    plt.title(title)
    plt.show()


def plot_cedata(lats, lons, data, title):
    """
    Plot coefficient of efficiency data.  Lats, lons, and data should have the
    same temporal dimensions.  Lats and lons should denote gridbox edges.
    e.g. data has spatial dimensions of M (lats) x N (lons) then lats and lons
    should have dimensions of M+1 x N+1.

    Parameters
    ----------
    lats: ndarray
        Latitude array (gridbox edges)
    lons: ndarray
        Longitude array (gridbox edges)
    data: ndarray
        CE data
    title: str
        Plot title
    """
    plt.close('all')
    m = Basemap(projection='gall', llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=0, urcrnrlon=360, resolution='c')
    m.drawcoastlines()
    
    if data.min() < 0:
        color = cm.bwr
    else:
        color = cm.OrRd
        
    m.pcolor(lons, lats, data, latlon=True, cmap=color, vmin=-1, vmax=1)
    m.colorbar()
    plt.title(title)
    plt.show()


def plot_spatial(lats, lons, data, title):
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

    if data.min() >= 0:
        color = cm.OrRd
        plt_max = plt_range
        plt_min = 0
    else:
        color = cm.bwr
        plt_max = plt_range
        plt_min = -plt_range

    m.pcolor(lons, lats, data, latlon=True, cmap=color, vmin=plt_min,
             vmax=plt_max)
    m.colorbar()
    
    plt.title(title)
    plt.show()
