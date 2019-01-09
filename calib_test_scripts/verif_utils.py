import os
import pandas as pd
import numpy as np
import dask.array as da
from multiprocessing import Pool
from itertools import product


import lim_utils as lutils
import plot_tools as ptools
import misc_utils as mutils
import data_utils as dutils

import pylim.Stats as ST


def get_scalar_outputs(dobj, nelem_in_yr, var_fcast, verif_data_attr,
                       out_types, use_dask=False, ):
    
    if use_dask:
        truth_data = dobj.reset_data(verif_data_attr)
    else:
        truth_data = getattr(dobj, verif_data_attr)
        
    truth_1yr = truth_data[nelem_in_yr:]
    truth_init = truth_data[:-nelem_in_yr]

    curr_var_output = {}

    for out_type in out_types:

        fcast_factor, verif_factor = get_scalar_factor(dobj, out_type,
                                                       verif_data_attr)

        var_out = var_fcast @ fcast_factor
        truth_init_out = truth_init @ verif_factor
        truth_1yr_out = truth_1yr @ verif_factor

        # Standardize PDO Index relative to truth output
        if out_type == 'pdo':
            truth_1yr_out, std_dev = _standardize_series(truth_1yr_out)
            var_out, _ = _standardize_series(var_out, std_dev=std_dev)
            truth_init_out, _ = _standardize_series(truth_init_out,
                                                    std_dev=std_dev)
            
        if use_dask:
            t_truth_1yr_out = np.empty(truth_1yr_out.shape)
            t_truth_init_out = np.empty(truth_init_out.shape)

            dask_vars = [truth_1yr_out, truth_init_out]
            dask_outs = [t_truth_1yr_out, t_truth_init_out]

            if ST.is_dask_array(var_out):
                t_var_out = np.empty(var_out.shape)
                dask_vars.append(var_out)
                dask_outs.append(t_var_out)

            da.store(dask_vars, dask_outs)

            truth_1yr_out = t_truth_1yr_out
            truth_init_out = t_truth_init_out

            if ST.is_dask_array(var_out):
                var_out = t_var_out

        curr_var_output[out_type] = {'fcast': var_out,
                                     't0': truth_init_out,
                                     '1yr': truth_1yr_out}
    return curr_var_output


def get_scalar_factor(dobj, out_type, verif_data_attr):

        eofs = dobj._eofs
        cell_area = dobj.cell_area

        lat, lon = dutils.get_lat_lon_grids(dobj, compressed=True, flat=True)

        if verif_data_attr == 'eof_proj' and out_type == 'glob_mean':
            fcast_factor, _ = lutils.get_glob_mean_factor(eofs, cell_area)
            verif_factor = fcast_factor
        elif out_type == 'glob_mean':
            fcast_factor, verif_factor = lutils.get_glob_mean_factor(eofs,
                                                                     cell_area)
        elif out_type == 'enso':
            fcast_factor, verif_factor = lutils.get_enso_factor(eofs, lat, lon)
        elif out_type == 'pdo':
            pdo_calc_data = dobj.area_weighted
            fcast_factor, verif_factor = lutils.get_pdo_factor(eofs,
                                                               pdo_calc_data,
                                                               lat, lon)
        else:
            raise KeyError('Unrecognized output key: {}'.format(out_type))

        return fcast_factor, verif_factor


def calc_scalar_ce_r(fcast, reference, init_t0):
    [r, r_conf95] = lutils.conf_bound95(fcast, reference, metric='r')
    [auto1_r, auto1_r_conf95] = lutils.conf_bound95(reference, init_t0,
                                                    metric='r')

    [ce, ce_conf95] = lutils.conf_bound95(fcast, reference, metric='ce')
    [auto1_ce, auto1_ce_conf95] = lutils.conf_bound95(reference, init_t0,
                                                      metric='ce')

    return (r, r_conf95, auto1_r, auto1_r_conf95, ce, ce_conf95,
            auto1_ce, auto1_ce_conf95)


def get_yrs_from_dobj(dobj, nelem_in_yr):
    yrs = dobj.get_dim_coords([dobj.TIME])
    yrs = yrs['time'][1]
    yrs = [d.year for d in yrs][nelem_in_yr:]

    return yrs


def perfect_fcast_verification(fcast_1yr, fcast_outputs, dobjs, state,
                               verif_spec, nelem_in_yr, experiment_name,
                               avg_key,
                               var_name_map, out_name_map, units_map,
                               fig_dir, do_scalar_plot=True,
                               do_spatial_plot=True):

    output = {}
    scalar_verif = []

    for var_key, out_types in fcast_outputs.items():
        dobj = dobjs[var_key]
        var_fcast = state.get_var_from_state(var_key, data=fcast_1yr)
        verif_data_attr = verif_spec.get(var_key, 'detrended')

        curr_var_output = get_scalar_outputs(dobj, nelem_in_yr, var_fcast,
                                             verif_data_attr, out_types,
                                             use_dask=True)

        # Run scalar verification
        for out_type, scalar_output in curr_var_output.items():
            fcast = scalar_output['fcast']
            ref = scalar_output['1yr']
            init_t0 = scalar_output['t0']

            r_ce_results = calc_scalar_ce_r(fcast, ref, init_t0)
            verif_df = mutils.ce_r_results_to_dataframe(var_key, avg_key,
                                                        out_type,
                                                        *r_ce_results)
            scalar_verif.append(verif_df)

            title = '{}, {}'.format(var_name_map[var_key],
                                    out_name_map[out_type])
            label = experiment_name + ' ' + avg_key
            yrs = get_yrs_from_dobj(dobj, nelem_in_yr)
            filename = 'scalar_plot_{}_{}_{}_{}.png'.format(experiment_name,
                                                            avg_key,
                                                            var_key,
                                                            out_type)
            filepath = os.path.join(fig_dir, filename)

            if out_type == 'enso':
                ylabel = 'ENSO 3.4 Index'
            elif out_type == 'pdo':
                ylabel = 'PDO Index'
            else:
                ylabel = 'Anomaly ({})'.format([units_map[var_key]])

            if do_scalar_plot:
                ptools.plot_scalar_verification(yrs, fcast, ref, *r_ce_results,
                                                title,
                                                label,
                                                ylabel,
                                                savefile=filepath)

        # Run Field verification

        if do_spatial_plot:
            fcast, ref_data, wgts = _get_spatial_field_and_wgts(dobj,
                                                                var_fcast,
                                                                var_key,
                                                                verif_spec,
                                                                get_dask=True)

            ref = ref_data[nelem_in_yr:]
            ref_init = ref_data[:-nelem_in_yr]

            lac = ST.calc_lac(fcast, ref)
            ce = ST.calc_ce(fcast, ref)

            # Persistence fcast metrics
            auto1_lac = ST.calc_lac(ref_init, ref)
            auto1_ce = ST.calc_ce(ref_init, ref)

            lac_out = np.empty(lac.shape)
            ce_out = np.empty(ce.shape)
            auto1_lac_out = np.empty(auto1_lac.shape)
            auto1_ce_out = np.empty(auto1_ce.shape)

            da.store([lac, ce, auto1_lac, auto1_ce],
                     [lac_out, ce_out, auto1_lac_out, auto1_ce_out])

            # spatial averages
            lac_gm = lac_out @ wgts
            ce_gm = ce_out @ wgts
            auto1_lac_gm = auto1_lac_out @ wgts
            auto1_ce_gm = auto1_ce_out @ wgts

            spatial_gm_df = mutils.ce_r_results_to_dataframe(var_key, avg_key,
                                                             'spatial_gm',
                                                             lac_gm,
                                                             None,
                                                             auto1_lac_gm,
                                                             None,
                                                             ce_gm,
                                                             None,
                                                             auto1_ce_gm,
                                                             None)

            scalar_verif.append(spatial_gm_df)

            curr_var_output['spatial_metr'] = {'lac': lac_out,
                                               'ce': ce_out,
                                               'auto1_lac': auto1_lac_out,
                                               'auto1_ce': auto1_ce_out}
            output[var_key] = curr_var_output

            _plot_spatial(lac_out, 'LAC', experiment_name, avg_key, var_key,
                          dobj, fig_dir)
            _plot_spatial(auto1_lac_out, 'Auto1_LAC', experiment_name, avg_key,
                          var_key, dobj, fig_dir)
            _plot_spatial(ce_out, 'CE', experiment_name, avg_key, var_key, dobj,
                          fig_dir)
            _plot_spatial(auto1_ce_out, 'Auto1_CE', experiment_name, avg_key,
                          var_key, dobj, fig_dir)

    scalar_verif = pd.concat(scalar_verif)

    return output, scalar_verif


def ens_fcast_verification(ens_fcast, fcast_outputs, dobjs, state,
                           verif_spec, nelem_in_yr, experiment_name, avg_key,
                           var_name_map, out_name_map, fig_dir,
                           do_hist=True, do_reliability=True):

    ens_metr_by_var = {}
    ens_scalar_out = {}
    for var_key, dobj in dobjs.items():

        var_fcast = state.get_var_from_state(var_key, data=ens_fcast)
        verif_data_attr = verif_spec.get(var_key, 'detrended')
        out_types = fcast_outputs[var_key]

        curr_var_output = get_scalar_outputs(dobj, nelem_in_yr, var_fcast,
                                             verif_data_attr, out_types,
                                             use_dask=True)

        ens_scalar_out[var_key] = curr_var_output

        curr_var_ens_scalar = {}
        for out_type, scalar_output in curr_var_output.items():
            print('Ens. Scalar Verification: {}, {}'.format(var_key, out_type))

            fcast_ens = scalar_output['fcast']
            ref = scalar_output['1yr']
            title = ('Exp: {}, {}  '
                     'Field: {} Measure: {}'.format(experiment_name,
                                                    avg_key,
                                                    var_key,
                                                    out_type))
            fig_fname = 'rank_hist_{}_{}_{}_{}.png'.format(experiment_name,
                                                           avg_key,
                                                           var_key,
                                                           out_type)
            fig_fpath = os.path.join(fig_dir, fig_fname)

            ens_calib = calc_ens_calib_ratio(fcast_ens, ref)
            curr_var_ens_scalar[out_type] = {'calib': ens_calib}

            if do_hist:
                rank_data = ptools.plot_rank_histogram(fcast_ens, ref, title,
                                                       savefile=fig_fpath)
                curr_var_ens_scalar[out_type]['rank'] = rank_data

            if (out_type == 'enso' or out_type == 'pdo') and do_reliability:

                title_temp = ('Exp: {}, {}  Metr: {} Reliability '
                              '(index {})')
                savefile_temp = 'reliability_{}_{}_{}_{}_{}.png'

                pct_map = {'upper': '>0.5',
                           'lower': '<-0.5'}

                reliab_dict = {}
                for event_type in ['upper', 'lower']:
                    obs_freq, bin_fcast_avg, errors = \
                        calc_reliability_with_bounds(fcast_ens, ref,
                                                     event_type=event_type)

                    title = title_temp.format(experiment_name, avg_key,
                                              out_type, pct_map[event_type])
                    fname = savefile_temp.format(experiment_name, avg_key,
                                                 var_key, out_type,
                                                 event_type)
                    savefile = os.path.join(fig_dir, fname)

                    ptools.plot_reliability(obs_freq, bin_fcast_avg, errors,
                                            title, savefile=savefile)
                    reliab_dict[event_type] = (obs_freq, bin_fcast_avg,
                                               errors)

                curr_var_ens_scalar[out_type]['reliability'] = reliab_dict

        # Run field ens_calibration measure

        # eofs = dobj._eofs
        #
        # wgts = dobj.cell_area / dobj.cell_area.sum()
        #
        # ref_data_attr = verif_spec.get(var_key, 'detrended')
        # ref_data = getattr(dobj, ref_data_attr)
        #
        # ntimes = var_fcast.shape[1]
        #
        # args = _stl_ens_calib_arg_generator(ntimes, var_fcast,
        #                                     ref_data, ref_data_attr, eofs)
        #
        # with Pool(processes=8) as proc_pool:
        #     calib_res = proc_pool.map(_stl_ens_calib_func, args)
        #
        # ens_calib_avg = np.array(calib_res).mean(axis=0)
        # spatl_avg_ens_calib = ens_calib_avg @ wgts
        # curr_var_ens_calib['spatial_avg'] = spatl_avg_ens_calib
        #
        #
        # if do_spatial_plot:
        #     _plot_spatial(ens_calib_avg, 'ens_calib', experiment_name,
        #                   avg_key, var_key, dobj, fig_dir)

        ens_metr_by_var[var_key] = curr_var_ens_scalar

    return ens_metr_by_var, ens_scalar_out


def long_output_to_scalar(output, dobjs, output_map, state, verif_spec,
                          use_dask=False):
    output_orig = state.proj_data_into_orig_basis(output, unstandardize=True)

    var_scalar_out = {}
    for var_key, dobj in dobjs.items():
        verif_data_attr = verif_spec.get(var_key, 'detrended')
        data = state.get_var_from_state(var_key, data=output_orig)

        if use_dask:
            truth_data = dobj.reset_data(verif_data_attr)
        else:
            truth_data = getattr(dobj, verif_data_attr)
            truth_data = truth_data[:]

        curr_var_output = {}
        for out_type in output_map[var_key]:

            fcast_factor, verif_factor = get_scalar_factor(dobj,
                                                           out_type,
                                                           verif_data_attr)
            scalar_out = data @ fcast_factor
            compare_scalar_out = truth_data @ verif_factor

            if out_type == 'pdo':
                [compare_scalar_out,
                 std_dev] = _standardize_series(compare_scalar_out)
                scalar_out, _ = _standardize_series(scalar_out, std_dev=std_dev)

            if use_dask:
                tmp_compare = np.empty(compare_scalar_out.shape)
                da.store(compare_scalar_out, tmp_compare)
                compare_scalar_out = tmp_compare

                if ST.is_dask_array(scalar_out):
                    tmp_scalar = np.empty(scalar_out.shape)
                    da.store(scalar_out, tmp_scalar)
                    scalar_out = tmp_scalar

            curr_var_output[out_type] = {'fcast': scalar_out,
                                         'source': compare_scalar_out}

        var_scalar_out[var_key] = curr_var_output

    return var_scalar_out


def _plot_spatial(field, metric, experiment_name, avg_key, var_key, dobj,
                  fig_dir):

    fname_template = 'spatial_verif_{}_{}_{}_{}.png'
    title_template = 'Exp: {}, {} Field: {} Metric: {}'

    field = dutils.reinflate_field(dobj, field)

    fname = fname_template.format(metric, experiment_name, avg_key, var_key)
    title = title_template.format(experiment_name, avg_key, var_key, metric)
    fpath = os.path.join(fig_dir, fname)

    if 'ce' in metric.lower():
        extend = 'min'
    else:
        extend = 'neither'

    if 'ens_calib' in metric.lower():
        cmap = 'inferno'
        bnds = [0, 10]
        midpoint = 1
    else:
        midpoint = None
        bnds = [-1, 1]
        cmap = 'RdBu_r'

    ptools.plot_single_spatial_field(dobj, field, title, data_bnds=bnds,
                                     savefile=fpath, gridlines=False,
                                     extend=extend, cmap=cmap,
                                     midpoint=midpoint)


def _get_spatial_field_and_wgts(dobj, var_fcast, var_key, verif_spec,
                                get_dask=False):

    ref_data_attr = verif_spec.get(var_key, 'detrended')
    if get_dask:
        ref_data = dobj.reset_data(ref_data_attr)
        var_fcast = da.from_array(var_fcast, (151, var_fcast.shape[-1]))
    else:
        ref_data = getattr(dobj, ref_data_attr)
        ref_data = ref_data[:]

    eofs = dobj._eofs
    fcast = var_fcast @ eofs.T

    if ref_data_attr == 'eof_proj':
        ref_data = ref_data @ eofs.T

    wgts = dobj.cell_area / dobj.cell_area.sum()

    return fcast, ref_data, wgts


def calc_ens_calib_ratio(fcast, ref):

    sq_err = (fcast.mean(axis=0) - ref)**2
    mse = sq_err.mean(axis=0)

    mean_ens_var = fcast.var(ddof=1, axis=0).mean(axis=0)

    ens_calib_ratio = mse / mean_ens_var

    return ens_calib_ratio


#  This is for spatial ensemble calibration
def _stl_ens_calib_func(args):

    i, var_fcast, ref_data, ref_data_attr, eofs = args
    curr_fcast_t = var_fcast[:, i:i+1] @ eofs.T
    curr_ref_t = ref_data

    if ref_data_attr == 'eof_proj':
        curr_ref_t = curr_ref_t @ eofs.T

    curr_ens_calib = calc_ens_calib_ratio(curr_fcast_t, curr_ref_t)

    return curr_ens_calib


def _stl_ens_calib_arg_generator(ntimes, var_fcast,
                                 ref_data, ref_data_attr, eofs):

    for i in range(ntimes):
        yield (i, var_fcast, ref_data[i+1:i+2], ref_data_attr, eofs)


def calc_ens_reliability(fcast_probs, occurences):

    # Get the number of counts for each bin
    bin_counts, bin_edges = np.histogram(fcast_probs, bins=10, range=(0, 1))

    # Map each forecast to a bin
    fcast_bin_map = np.digitize(fcast_probs, bin_edges[:-1])

    bin_num_hits = []
    bin_fcast_mean = []
    for i, bin_count in enumerate(bin_counts, start=1):
        idxs, = np.where(fcast_bin_map == i)
        hits = occurences[idxs].sum()
        mean = fcast_probs[idxs].mean()
        bin_num_hits.append(hits)
        bin_fcast_mean.append(mean)

    bin_num_hits = np.array(bin_num_hits)
    bin_counts = np.array(bin_counts)
    bin_fcast_mean = np.array(bin_fcast_mean)

    obs_rel_freq = bin_num_hits / bin_counts

    return obs_rel_freq, bin_fcast_mean


def _get_event_func(ref, event_type):

    if event_type == 'upper':
        def event(x):
            return x >= 0.5
    elif event_type == 'lower':
        def event(x):
            return x <= -0.5
    else:
        raise ValueError('Unrecognized event designation key: '
                         '{}'.format(event_type))

    return event


def _mc_obs_rel_freq(args):

    i, fcast_probs = args

    np.random.seed(i)

    x_hat = np.random.choice(fcast_probs, size=len(fcast_probs),
                             replace=True)
    y_hat = np.random.random(size=len(x_hat))
    y_hat = np.less(y_hat, x_hat)
    obs_rel_freq, _ = calc_ens_reliability(x_hat, y_hat)

    return obs_rel_freq


def calc_reliability_with_bounds(fcast, ref, event_type='upper'):

    # Resampling operation to bound reliable forecast region
    # Brocker and Smith 2007

    event = _get_event_func(ref, event_type)

    nens = fcast.shape[0]
    fcast_probs = event(fcast).sum(axis=0) / nens
    occurrences = event(ref)

    num_mc_iter = 1000
    seeds = np.random.choice(num_mc_iter*10, size=num_mc_iter, replace=False)
    args = product(seeds, (fcast_probs, ))

    with Pool(processes=8) as reliable_pool:
        res_obs_freq = reliable_pool.map(_mc_obs_rel_freq, args)

    res_obs_freq = np.array(res_obs_freq)
    upper_bnd = np.percentile(res_obs_freq, 97.5, axis=0)
    lower_bnd = np.percentile(res_obs_freq, 2.5, axis=0)

    [obs_rel_freq,
     bin_fcast_mean] = calc_ens_reliability(fcast_probs, occurrences)

    mean_rof = res_obs_freq.mean(axis=0)
    upper_bnd = upper_bnd - mean_rof
    lower_bnd = abs(lower_bnd - mean_rof)

    errors = np.vstack((lower_bnd, upper_bnd))

    return obs_rel_freq, bin_fcast_mean, errors


def _standardize_series(data, std_dev=None):

    data = data - data.mean(axis=-1, keepdims=True)

    if std_dev is None:
        std_dev = np.std(data, axis=-1, ddof=1)

    data = data / std_dev

    return data, std_dev

