import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as patches
import matplotlib.colors as colors
import cartopy.crs as ccrs
import numpy as np

import lim_utils as lutils
import misc_utils as mutils
import data_utils as dutils


INTERACTIVE_PLOT = False


def init_projection(projection):
    if projection == 'rotated':
        projection = ccrs.RotatedPole(pole_latitude=72.18, 
                                        pole_longitude=-39.5, 
                                        central_rotated_longitude=140)
    elif projection == 'plate':
        projection = ccrs.PlateCarree()
    elif projection == 'robinson':
        projection = ccrs.Robinson()
        
    return projection
    
def spatial_plotter(lon, lat, data, title, ax=None, do_colorbar=True, 
                    data_bound=None, projection='robinson', gridlines=True,
                    extend='neither',
                    **kwargs):
    
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError('Lat/Lon fields must be 2D')

    projection = init_projection(projection)
    
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        spec = gs.GridSpec(2, 1, height_ratios=[20, 1])
        ax = plt.subplot(spec[0], projection=projection)
    else:
        spec = None
        
    fmax = np.nanmax(data)
    fmin = np.nanmin(data)
    if data_bound is None:
        bnd = max(abs(fmax), abs(fmin))
        bnd1 = -bnd
        bnd2 = bnd
    else:
        bnd1, bnd2 = data_bound

    cf = ax.pcolormesh(lon, lat, data, vmin=bnd1, vmax=bnd2,
                      transform=ccrs.PlateCarree(), **kwargs)
    ax.set_aspect('auto')
    ax.coastlines(alpha=0.7)
    if gridlines:
        ax.gridlines()
    ax.set_title(title)
    ax.set_global()
    
    if do_colorbar and spec is not None:
        plt.colorbar(cf, cax=plt.subplot(spec[1]), orientation='horizontal',
                     extend=extend)
        
    return cf


def plot_multiple_fields(nrows, ncols, plot_arg_tuples, cbar_type='single',
                         projection='robinson', gridlines=True, save_file=None):
    projection = init_projection(projection)
    fig = plt.figure(figsize=(6*ncols, 4.5*nrows))
    height_ratios = [20]*nrows
    width_ratios = [20]*ncols
    total_nrows = nrows
    total_ncols = ncols

    if cbar_type == 'single' or cbar_type == 'col':
        height_ratios += [1]
        total_nrows += 1
    elif cbar_type == 'row':
        width_ratios += [1]
        total_ncols += 1
    elif cbar_type is not None:
        raise ValueError('Unrecognized colorbar type: {}'.format(cbar_type))
    
    spec = gs.GridSpec(total_nrows, total_ncols, height_ratios=height_ratios,
                       width_ratios=width_ratios)
    
    cf_arr = []
    for i in range(nrows):
        cf_arr.insert(i, [])
        for j in range(ncols):
            curr_spec = spec[i, j]
            plot_args, plot_kwargs = plot_arg_tuples[i][j]
            ax = plt.subplot(curr_spec, projection=projection)
            ax.background_patch.set_facecolor('#020b36')
            plot_kwargs.update({'ax': ax})
            cf = spatial_plotter(*plot_args, gridlines=gridlines, **plot_kwargs)
            cf_arr[i].append(cf)
    
    if cbar_type == 'single':
        cax = plt.subplot(spec[-1, :])
        cf = cf_arr[-1][0]
        plt.colorbar(cf, cax=cax, orientation='horizontal')
    elif cbar_type == 'row':
        for i in range(nrows):
            cbar_spec = spec[i, -1]
            cf = cf_arr[i][-1]
            cax = plt.subplot(cbar_spec)
            plt.colorbar(cf, cax=cax, orientation='vertical')
    elif cbar_type == 'col':
        for i in range(ncols):
            cbar_spec = spec[-1, i]
            cf = cf_arr[-1][i]
            cax = plt.subplot(cbar_spec)
            plt.colorbar(cf, cax=cax, orientation='horizontal')
            
    if save_file is not None:
        plt.savefig(save_file, dpi=150)

    if INTERACTIVE_PLOT:
        plt.show()

    plt.close(fig)


def plot_exp_eofs(dobjs, eofs_by_var, filename=None, title=None):
    ncols = len(dobjs)

    var_names = list(dobjs.keys())

    var_pargs = []
    for var in var_names:

        curr_pargs = get_dobj_eof_plot_args(dobjs[var], eofs_by_var[var],
                                            var, title=title)
        var_pargs.append(curr_pargs)

    full_pargs = list(zip(*var_pargs))
    nrows = len(full_pargs)

    plot_multiple_fields(nrows, ncols, full_pargs,
                         cbar_type='col', gridlines=False,
                         save_file=filename)


def plot_single_spatial_field(dobj, field, title, data_bnds=None,
                              savefile=None, cmap='RdBu_r', midpoint=None,
                              **kwargs):
    lat, lon = dutils.get_lat_lon_grids(dobj, compressed=False, flat=False)
    lat_bnd, lon_bnd = mutils.calculate_latlon_bnds(lat, lon)

    if midpoint is not None:
        norm = _MidpointNormalize(midpoint=midpoint)
    else:
        norm = None

    spatial_plotter(lon_bnd, lat_bnd, field, title, data_bound=data_bnds,
                    cmap=cmap, norm=norm, **kwargs)

    if savefile is not None:
        plt.savefig(savefile, fmt='png', dpi=150)

    if INTERACTIVE_PLOT:
        plt.show()


def get_var_plot(dobj, name, data, partial_title=None, orig_basis=True):
    if orig_basis and dobj._eofs is not None:
        data = data @ dobj._eofs.T

    var_title = 'Field: {} '.format(name)
    if partial_title is not None:
        title = var_title + partial_title
    else:
        title = var_title

    grids = dobj.get_coordinate_grids(['lat', 'lon'], compressed=False)
    lat = grids['lat']
    lon = grids['lon']
    lat_bnds, lon_bnds = mutils.calculate_latlon_bnds(lat, lon, lat_ax=0)

    if dobj.is_masked:
        data = dobj.inflate_full_grid(data=data, reshape_orig=True)
        data = np.ma.masked_invalid(data)
    else:
        data = data.reshape(dobj._spatial_shp)

    plt_args = (lon_bnds, lat_bnds, data, title)
    plt_kwargs = {'cmap': 'RdBu'}

    return (plt_args, plt_kwargs)


def get_single_mode_plot(mode_num, l_eval, l_evect, state, partial_title=None):
    l_eval = l_eval.real
    l_evect = l_evect.real

    mode_title = 'Mode: {}  EFT: {:1.2f} yr'.format(mode_num, -1 / l_eval)

    if partial_title is not None:
        mode_title = mode_title + partial_title

    var_keys = list(state.dobjs.keys())
    plot_args = []
    for vkey in var_keys:
        dobj = state.dobjs[vkey]
        data = state.get_var_from_state(vkey, data=l_evect)
        parg_tup = get_var_plot(dobj, vkey, data, partial_title=mode_title,
                                orig_basis=True)
        plot_args.append(parg_tup)

    return plot_args


def plot_multi_lim_modes(lim_obj, state, row_limit=20, save_file=None):

    # Get eigenvectors/values for G1
    g_evals, g_evects = np.linalg.eig(lim_obj.G_1)

    # Convert to eigenvalues for L and sort (L and G have same eigenvectors)
    l_evals = np.log(g_evals)
    sort_idx = l_evals.argsort()
    l_evals = l_evals[sort_idx][::-1]
    l_evects = g_evects[:, sort_idx][:, ::-1]

    l_evects = np.squeeze(np.array(l_evects))

    # Iterate over modes looking for complex conjugate pairs
    i = 0
    modes = []
    while i < (len(l_evals) - 1):
        if l_evals[i].real == l_evals[i + 1].real:
            modes.append((i, i + 1))
            i = i + 1
        else:
            modes.append(i)
        i += 1
    else:
        # Account for final single decaying mode, if last mode was pair
        # i = len(l_evals)
        if i < len(l_evals):
            modes.append(i)

    plot_arg_tups = []
    i = 0
    for mode_num in modes:
        if i + 1 > row_limit:
            break

        if isinstance(mode_num, tuple):
            mode1, mode2 = mode_num
            curr_evect = l_evects[:, mode1]
            curr_eval = l_evals[mode1]
            new_evect = lutils.get_ortho_complex_basis(curr_evect)
            cos_phase = new_evect.real
            cos_phase = state.proj_data_into_orig_basis(cos_phase)
            sin_phase = new_evect.imag
            sin_phase = state.proj_data_into_orig_basis(sin_phase)

            period = 2 * np.pi / curr_eval.imag
            dual_title = ' {} T: {:1.2f} yr'
            mode_str = '{}/{}'.format(mode1, mode2)

            pargs1 = get_single_mode_plot(mode_str,
                                          curr_eval,
                                          cos_phase,
                                          state,
                                          partial_title=dual_title.format('COS',
                                                                          period))
            pargs2 = get_single_mode_plot(mode_str,
                                          curr_eval,
                                          sin_phase,
                                          state,
                                          partial_title=dual_title.format('SIN',
                                                                          period))
            pargs = [pargs1, pargs2]
            plot_arg_tups += pargs
            i += 2

        else:
            curr_evect = l_evects[:, mode_num]
            curr_evect_basis = state.proj_data_into_orig_basis(curr_evect)
            curr_eval = l_evals[mode_num]

            pargs = get_single_mode_plot(mode_num, curr_eval,
                                         curr_evect_basis, state)
            plot_arg_tups.append(pargs)
            i += 1

    nrows = len(plot_arg_tups)
    ncols = len(state.dobjs.keys())
    plot_multiple_fields(nrows, ncols, plot_arg_tups,
                         cbar_type='col', gridlines=False,
                         save_file=save_file)


def get_bound_and_cmap(field):
    minval = np.nanmin(field)
    maxval = np.nanmax(field)
    if minval >= 0:
        bound = (0, maxval)
        cmap = 'PuBu'
    else:
        abs_maxval = max(abs(minval), abs(maxval))
        bound = (-abs_maxval, abs_maxval)
        cmap = 'BrBG'
        
    return bound, cmap

# Shift results for skill with annual means
def shift_results_plot_args(shift_spatial_res, dobjs, gm_values):
    
    plot_params = {}
    field_grids = {}
    for field_key, field_results in shift_spatial_res.items():
        plot_params[field_key] = []
        dobj = dobjs[field_key]
        grids = dobj.get_coordinate_grids(['lat', 'lon'], compressed=False)
        lat = grids['lat']
        lon = grids['lon']
        field_grids[field_key] = (lon, lat)
        
        
    for field_key, field_results in shift_spatial_res.items():
        for j, (metr_key, metric) in enumerate(field_results.items()):
            bnd, cmap = get_bound_and_cmap(metric)
            avg = gm_values[field_key][metr_key]
            
            for i in range(metric.shape[0]):
                if j == 0:
                    plot_params[field_key].append([])

                title = 'Field: {} Metric: {} Avg: {:.2f}'.format(field_key,
                                                                  metr_key,
                                                                  avg[i])
                lon_lat = field_grids[field_key]
                plot_args = (*lon_lat, metric[i], title)
                plot_kwargs = {'cmap': cmap, 'data_bound': bnd}
                plot_params[field_key][i].append((plot_args, plot_kwargs))
                
                
    return plot_params


def get_dobj_eof_plot_args(dobj, eofs, name, title=None):

    lat, lon = dutils.get_lat_lon_grids(dobj, compressed=False)
    lat_bnds, lon_bnds = mutils.calculate_latlon_bnds(lat, lon, lat_ax=0)

    plot_args = []
    for i in range(eofs.shape[1]):
        eof = eofs[:, i]

        eof = dutils.reinflate_field(dobj, eof)

        if title is None:
            stats = dobj.get_eof_stats()
            var_expl = stats['var_expl_by_mode'][i]*100
            curr_title = 'Field: {} EOF_{:d} '.format(name, i + 1)
            curr_title += 'Var Expl: {:2.1f}%'.format(var_expl)
        else:
            curr_title = title.format(i+1, name)

        curr_plt_args = (lon_bnds, lat_bnds, eof, curr_title)
        curr_plt_kwargs = {'cmap': 'RdBu'}

        plot_args.append((curr_plt_args, curr_plt_kwargs))

    return plot_args


def plot_scalar_verification(years, fcast, reference, r, r_conf95, auto1_r,
                             auto1_r_conf95, ce, ce_conf95, auto1_ce,
                             auto1_ce_conf95, title, ref_name, ylabel,
                             savefile=None):

    capsize = 5
    mew = 2
    tseries_title = '1-year LIM Forecast vs. Ref:  ' + title

    r_conf95 = _convert_bnds_for_errorplot(r_conf95, r)
    auto1_r_conf95 = _convert_bnds_for_errorplot(auto1_r_conf95, auto1_r)
    ce_conf95 = _convert_bnds_for_errorplot(ce_conf95, ce)
    auto1_ce_conf95 = _convert_bnds_for_errorplot(auto1_ce_conf95, auto1_ce)

    spec = gs.GridSpec(1, 3, width_ratios=[6, 1, 1])
    fig = plt.figure(figsize=(10, 4))
    tseries_ax = plt.subplot(spec[0])
    r_ax = plt.subplot(spec[1])
    ce_ax = plt.subplot(spec[2])

    # Time Series Plot
    tseries_ax.plot(years, reference, color='C0', label='CCSM4: ' + ref_name)
    tseries_ax.plot(years, fcast, color='C1', label='LIM 1-yr fcast')
    tseries_ax.set_xlabel('Year')
    tseries_ax.set_ylabel(ylabel)
    tseries_ax.set_title(tseries_title)
    tseries_ax.legend(loc='best')

    # Correlation w/ 95% Conf
    r_ax.errorbar(-0.1, r, fmt='o', yerr=r_conf95, color='C3',
                  label='Fcast/Ref', capsize=capsize, mew=mew)
    r_ax.errorbar(0.1, auto1_r, fmt='o', yerr=auto1_r_conf95, color='grey',
                  label='Ref 1yr Persist', capsize=capsize, mew=mew)
    r_ax.set_title('Correlation')
    r_ax.tick_params(axis='x', which='both', bottom='off', top='off',
                     labelbottom='off')
    r_ax.set_xlim(-0.8, 0.8)
    lim_patch = patches.Patch(color='C3', label='LIM Fcast vs. Ref')
    auto_patch = patches.Patch(color='grey', label='Ref 1-yr Persistence')
    r_ax.legend(handles=[lim_patch, auto_patch],
                bbox_to_anchor=(2, 0))

    # CE w/ 95% Conf
    ce_ax.errorbar(-0.1, ce, fmt='^', yerr=ce_conf95, color='C3',
                   label='Fcast/Ref', capsize=capsize, mew=mew)
    ce_ax.errorbar(0.1, auto1_ce, fmt='^', yerr=auto1_ce_conf95, color='grey',
                   label='Ref 1yr Persist', capsize=capsize, mew=mew)
    ce_ax.set_title('CE')
    ce_ax.tick_params(axis='x', which='both', bottom='off', top='off',
                      labelbottom='off')
    ce_ax.set_xlim(-0.8, 0.8)

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, fmt='png', dpi=150)

    if INTERACTIVE_PLOT:
        plt.show()

    plt.close(fig)


def _convert_bnds_for_errorplot(bnds, res_mean):

    upper_bnd, lower_bnd = bnds
    lower_bnd = abs(lower_bnd - res_mean)
    upper_bnd = abs(upper_bnd - res_mean)

    # Create a 2x1 array to be interpreted by error bar as upper/lower
    error_bar_bnds = np.array([lower_bnd, upper_bnd])[:, None]
    return error_bar_bnds


def plot_rank_histogram(test_ens, ref_data, title, savefile=None):

    # TODO: Doesn't account for ties
    nens = test_ens.shape[0]
    nbins = nens + 1

    # Count number of ensemble members less than or equal to reference
    # Equivalent to a rank assuming there aren't tied values
    less = np.less_equal(test_ens, ref_data).sum(axis=0).astype(np.int)

    # Adjust ranks for start at 1 instead of 0
    less += 1

    # Find count of each rank (i.e. how many times had each rank)
    unique, counts = np.unique(less, return_counts=True)
    hist_dict = dict(zip(unique, counts))

    bins = np.linspace(0.5, nbins+0.5, nbins+1)
    y_upper_bnd = (counts / counts.sum()).max() + 0.1

    fig = plt.figure(figsize=(5, 5))
    plt.hist(less, bins=bins, density=True)
    plt.title(title)
    plt.xticks([1, nbins])
    plt.xlim([0.5, nbins+0.5])
    plt.xlabel('Rank')
    plt.ylim([0, y_upper_bnd])
    plt.ylabel('P(Rank)')
    plt.axhline(1/nbins, linestyle='--', color='k')

    if savefile is not None:
        plt.savefig(savefile, fmt='png', dpi=150)

    if INTERACTIVE_PLOT:
        plt.show()

    plt.close(fig)

    return hist_dict


def plot_reliability(obs_freq, fcast_bin_loc, errors, title, savefile=None):

    use_idx = np.isfinite(obs_freq)
    use_idx &= np.isfinite(fcast_bin_loc)

    obs_freq = obs_freq[use_idx]
    fcast_bin_loc = fcast_bin_loc[use_idx]
    errors = errors[:, use_idx]

    fig = plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linewidth=0.5, linestyle='--', color='k')
    plt.errorbar(fcast_bin_loc, fcast_bin_loc, yerr=errors, color='k',
                 capsize=5, mew=2, fmt='|', linewidth=0, elinewidth=1)
    plt.plot(fcast_bin_loc, obs_freq, linewidth=2, marker='x',
             markersize=9, mew=3, mec='k')
    plt.xlabel('forecast probabilities')
    plt.ylabel('observed frequencies')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(title, fontsize=11)
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, fmt='png', dpi=150)

    if INTERACTIVE_PLOT:
        plt.show()

    plt.close(fig)


class _MidpointNormalize(colors.Normalize):
    """
    Copied simple example from
    https://matplotlib.org/users/colormapnorms.html
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


