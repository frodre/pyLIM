import data_utils as dutils
import lim_utils as lutils
import plot_tools as ptools
import verif_utils as vutils
import misc_utils as mutils
import pylim.LIM as LIM

import os
import logging
import pickle
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Main')

# Which experiment to use
experiment_name = 'past1000'
# experiment_name = 'piControl'
# experiment_name = 'montlhly_past1000'
# experiment_name = 'monthly_piControl'

base_sim = 'past1000'
# base_sim = 'piControl'

# What variables in LIM
var_names = ['tas', 'tos', 'zos']

# What averaging
# avg_key = 'ann_std'
avg_key = 'ann_spr'
# avg_key = 'ann_fall'
# avg_key = 'none'

# Number of elements in a year
yr_len = 1

# Data to forecast on other than self
# fcast_against = {'piControl': 'ann_spr'}
# fcast_against = {'past1000': 'ann_spr'}
# fcast_against = {'piControl': 'ann_fall',
#                  'past1000': 'ann_fall'}
fcast_against = {}
comp_yr_len = 1

# Use pre-loaded files
use_preproc_files = False

neof_var = 400
neof_mulivar = 30

# Locations for data files
ncf_dir = '/home/disk/katabatic/wperkins/data/LMR/data/model/'
h5_dir = '/home/disk/katabatic/wperkins/data/pyLIM'
lim_obj_dir = '/home/disk/katabatic/wperkins/data/pyLIM_output'
fig_dir = os.path.join('/home/disk/p/wperkins/ipynb/ens_pred/production_online_lims',
                       experiment_name, avg_key)
long_integ_output = os.path.join(
    '/home/disk/katabatic/wperkins/data/long_integ_out',
    experiment_name,
    avg_key)

# Fig output
plot_neofs = 10
plot_eofs = False
plot_state_eofs = False

plot_lim_modes = False
plot_num_lim_modes = 10

plot_lim_noise_eofs = False
plot_num_noise_modes = 10

# Do the self forecasting, or only forecast against?
fcast_against_only = False

# Perfect Forecast Experiments
do_perfect_fcast = False
fcast_outputs = {'tas': ['glob_mean'],
                 'tos': ['glob_mean',
                         'enso',
                         'pdo'],
                 'zos': ['glob_mean']}
verif_spec = {'zos': 'eof_proj'}
plot_scalar_verif = True
plot_spatial_verif = False

# Ensemble noise integration forecast experiments
do_ens_fcast = False
do_hist = True
do_reliability = True

# Long integration forecast experiments
do_long_integration = True
integration_len_yr = 500
integration_iters = 100

# Definitions
# TODO: consistency w/ LMR definitions
avg_key_map = {'ann_std': {'args': ('ann_std', 12),
                           'kwargs': {'shift': 0}},
               'ann_spr': {'args': ('ann_spr', 12),
                           'kwargs': {'shift': 4}},
               'ann_fall': {'args': ('ann_fall', 12),
                            'kwargs': {'shift': 10}},
               'none': None}

scalar_type_name = {'glob_mean': 'Global Avg',
                    'enso': 'ENSO 3.4 Index',
                    'pdo': 'PDO Index'}

var_long_name = {'tas': 'Sfc Air Temp',
                 'tos': 'Sea Sfc Temp',
                 'zos': 'Dynamic Sea Sfc Height'}

units = {'tas': 'K',
         'zos': 'm',
         'tos': 'K'}

var_mapping = {'tas': 'atmos',
               'tos': 'ocean',
               'zos': 'ocean'}

cell_area_by_realm = {'atmos': 'areacella_fx_CCSM4_{}_r0i0p0.nc',
                      'ocean': 'areacello_fx_CCSM4_{}_r0i0p0.nc'}

exp_folder = {'past1000': 'ccsm4_last_millenium',
              'piControl': 'ccsm4_piControl'}
exp_files = {'past1000': '{}_CCSM4_past1000_085001-185012.nc',
             'piControl': '{}_CCSM4_piControl_025001-130012.nc'}


def get_cell_area_map(base_sim):
    cell_area_by_var = {}
    for var_key in var_mapping.keys():
        realm = var_mapping[var_key]
        cell_area_file = cell_area_by_realm[realm]
        cell_area_file = cell_area_file.format(base_sim)
        cell_area_path = os.path.join(ncf_dir, exp_folder[base_sim],
                                      cell_area_file)
        cell_area_by_var[var_key] = cell_area_path

    return cell_area_by_var


def get_fnames(base_sim):

    fname_exp = 'ccsm4_{exp}'.format(exp=base_sim)
    h5_fname_var = '_{}.h5'
    dobj_fname_var = '_{}_dobj'

    preproc = os.path.join(lim_obj_dir, fname_exp + dobj_fname_var)
    h5 = os.path.join(h5_dir, fname_exp + h5_fname_var)
    ncf_f = os.path.join(ncf_dir, exp_folder[base_sim], exp_files[base_sim])

    return preproc, h5, ncf_f


def load_n_eof_em(name_base_sim, average_key,
                  nelem_in_yr=1, num_eofs=500, print_stats=True,
                  dobjs_for_eof_basis=None):

    preproc_fname, h5_fname, ncf_fname = get_fnames(name_base_sim)
    cell_area_by_var = get_cell_area_map(name_base_sim)
    curr_dobjs = dutils.load_dobjs(var_names, h5_fname, cell_area_by_var,
                                   ncf_fname, preproc_fname, use_preproc_files)

    dutils.check_proper_average(curr_dobjs, average_key,
                                avg_key_map[average_key])
    do_save = dutils.prep_dobjs_for_eof(nelem_in_yr, curr_dobjs)
    if do_save:
        dutils.save_dobjs(curr_dobjs, preproc_fname + '.pkl')

    do_save = dutils.calc_eofs_dobjs(curr_dobjs, num_eofs=num_eofs,
                                     print_stats=print_stats,
                                     eof_dobjs=dobjs_for_eof_basis)
    do_save |= dutils.standardize_dobjs(curr_dobjs,
                                        std_dobjs=dobjs_for_eof_basis)
    
    if do_save:
        dutils.save_dobjs(curr_dobjs, preproc_fname + '.pkl')

    return curr_dobjs


# Load main data objects
dobjs = load_n_eof_em(base_sim, avg_key,
                      nelem_in_yr=yr_len)
os.makedirs(fig_dir, exist_ok=True)

if plot_eofs:
    fig_fname = os.path.join(fig_dir,
                             '{}_{}_basis_eofs.png'.format(experiment_name,
                                                           avg_key))
    dobj_eofs = {var_key: dobjs[var_key]._eofs[:, :plot_neofs]
                 for var_key in var_names}
    ptools.plot_exp_eofs(dobjs, dobj_eofs, filename=fig_fname)

state = lutils.LIMState(dobjs, dobj_key='standardized')
multivar_eof_stats = state.calc_state_eofs(neof_mulivar)
state.proj_state_onto_eofs()

logger.info('\nMultivar EOF Stats:\n\tVariance Retained: {:1.2f}'
            '\n\tVar Retained (by mode){}'
            ''.format(multivar_eof_stats['var_expl_by_ret'],
                      multivar_eof_stats['var_expl_by_mode'][:10]))

if plot_state_eofs:
    fig_fname = os.path.join(fig_dir,
                             '{}_{}_multivar_eofs.png'.format(experiment_name,
                                                              avg_key))

    multivar_eofs = {}
    for var_key in var_names:
        dobj = dobjs[var_key]
        state_eofs = state.get_var_eofs(var_key)
        dobj_eofs = dobj._eofs

        multivar_eofs[var_key] = dobj_eofs @ state_eofs[:, :plot_neofs]

    title = 'Multivar EOF_{:d}  Field: {}'

    ptools.plot_exp_eofs(dobjs, multivar_eofs, filename=fig_fname, title=title)

t0 = state.data[:-yr_len]
t1 = state.data[yr_len:]
lim = LIM.LIM(tau0_data=t0, tau1_data=t1, fit_noise=True)

if plot_lim_noise_eofs:
    fig_fname = os.path.join(fig_dir,
                             '{}_{}_noise_eofs.png'.format(experiment_name,
                                                           avg_key))
    Q_evect = lim.Q_evects[:, :plot_num_noise_modes]
    Q_evals = lim.Q_evals[:plot_num_noise_modes]
    noise_eofs = {}
    for var_key in var_names:
        dobj = dobjs[var_key]
        state_eofs = state.get_var_eofs(var_key)
        dobj_eofs = dobj._eofs
        real_Q_evects = (Q_evect @ np.diag(Q_evals)).real
        noise_eofs[var_key] = dobj_eofs @ state_eofs @ real_Q_evects

    title = 'Noise EOF_{:d}  Field: {}'

    ptools.plot_exp_eofs(dobjs, noise_eofs, filename=fig_fname, title=title)

# Load test comparison data objects
fcast_test_dobjs = {}
fcast_test_state = {}
for comparison_exp, compare_avg_key in fcast_against.items():
    if comparison_exp == base_sim:
        dutils.close_hdf(dobjs)

    # TODO: Fix year length description
    comp_dobjs = load_n_eof_em(comparison_exp, compare_avg_key,
                               nelem_in_yr=comp_yr_len,
                               dobjs_for_eof_basis=dobjs)
    test_name = '_'.join([comparison_exp, compare_avg_key])
    fcast_test_dobjs[test_name] = comp_dobjs
    comp_state = lutils.LIMState(comp_dobjs, dobj_key='standardized')
    fcast_test_state[test_name] = comp_state

if plot_lim_modes:
    fig_fname = os.path.join(fig_dir,
                             '{}_{}_lim_fcast_modes.png'.format(experiment_name,
                                                                avg_key))
    ptools.plot_multi_lim_modes(lim, state, row_limit=plot_num_lim_modes,
                                save_file=fig_fname)

if do_perfect_fcast:

    fcast_on_str = 'fcast_on-' + '_'.join([experiment_name, avg_key])
    df_fname = 'scalar_df_{}_{}.h5'.format(experiment_name, avg_key)
    df_path = os.path.join(fig_dir, df_fname)

    output = {}

    if not fcast_against_only:
        init = state.data[:-yr_len]
        fcast_1yr = lim.forecast(init, [1])[0]

        # Put back in original state basis
        state.proj_state_into_phys()
        fcast_1yr_basis = state.proj_data_into_orig_basis(fcast_1yr, unstandardize=True)

        output['fcast'] = {'data': fcast_1yr, 'state_eofs': state.eofs,
                            'var_span': state.var_span, 'avg_key': avg_key}

        # Forecast verification on self
        [verif_outputs,
         verif_df] = \
            vutils.perfect_fcast_verification(fcast_1yr_basis, fcast_outputs, dobjs,
                                              state, verif_spec,
                                              yr_len, experiment_name,
                                              avg_key, var_long_name,
                                              scalar_type_name, units, fig_dir,
                                              do_scalar_plot=plot_scalar_verif,
                                              do_spatial_plot=plot_spatial_verif)

        output['verif'] = verif_outputs
        verif_df.to_hdf(df_path, fcast_on_str, mode='a')

    # Get other states to test LIM forecast on.
    fcast_test_outputs = {}
    for test_name, comp_state in fcast_test_state.items():
        test_1yr_fcast, test_1yr_init = lutils.fcast_1yr_state(lim,
                                                               state,
                                                               comp_state,
                                                               comp_yr_len)
        fcast_test_outputs[test_name] = (test_1yr_fcast, test_1yr_init)

    # Forecast verification on each forecast against specified
    for exp_name, (test_fcast, test_init) in fcast_test_outputs.items():
        test_dobjs = fcast_test_dobjs[exp_name]
        curr_fig_dir = os.path.join(fig_dir, 'fcast_on-' + exp_name)
        os.makedirs(curr_fig_dir, exist_ok=True)
        [test_outputs,
         test_df] = \
            vutils.perfect_fcast_verification(test_fcast, fcast_outputs,
                                              test_dobjs, state, verif_spec,
                                              comp_yr_len,
                                              experiment_name+'X', avg_key,
                                              var_long_name,
                                              scalar_type_name, units,
                                              curr_fig_dir,
                                              do_scalar_plot=plot_scalar_verif,
                                              do_spatial_plot=plot_spatial_verif)

        fcast_on_str = 'fcast_on-'+exp_name
        test_df.to_hdf(df_path, fcast_on_str, mode='a')

        output[fcast_on_str] = test_outputs

    output_fname = 'fcast_output_{}_{}.pkl'.format(experiment_name, avg_key)
    output_fpath = os.path.join(fig_dir, output_fname)
    with open(output_fpath, 'wb') as f:
        pickle.dump(output, f)

if do_ens_fcast:
    nens = 100

    if not fcast_against_only:
        state.proj_state_onto_eofs()
        t0 = state.data[:-yr_len]

        ens_output = lutils.ens_1yr_fcast(nens, lim, t0, timesteps=1440)

        ens_output_full = state.proj_data_into_orig_basis(ens_output, unstandardize=True)

        # calc rank histogram
        [ens_metr_out,
         ens_scalar_out] = \
            vutils.ens_fcast_verification(ens_output_full,
                                          fcast_outputs, dobjs, state,
                                          verif_spec, yr_len, experiment_name,
                                          avg_key, var_long_name,
                                          scalar_type_name, fig_dir,
                                          do_hist=do_hist,
                                          do_reliability=do_reliability)

        scalar_out_fname = ('fcast_ens_scalar_output_'
                            '{}_{}.pkl'.format(experiment_name, avg_key))
        metr_out_fname = 'fcast_ens_metr_output_{}_{}.pkl'.format(experiment_name,
                                                                  avg_key)
        output_fpath = os.path.join(fig_dir, scalar_out_fname)
        output_fpath2 = os.path.join(fig_dir, metr_out_fname)
        with open(output_fpath, 'wb') as f, open(output_fpath2, 'wb') as f2:
            pickle.dump(ens_scalar_out, f)
            pickle.dump(ens_metr_out, f2)

    for test_name, test_dobjs in fcast_test_dobjs.items():
        logger.info('Test ensemble forecast: {}'.format(test_name))
        test_state = fcast_test_state[test_name]
        test_data = state.proj_data_into_eof_basis(test_state.data)

        curr_fig_dir = os.path.join(fig_dir, 'fcast_on-'+test_name)
        os.makedirs(curr_fig_dir, exist_ok=True)

        t0 = test_data[:-comp_yr_len]

        ens_output = lutils.ens_1yr_fcast(nens, lim, t0, timesteps=1440)
        ens_output_full = state.proj_data_into_orig_basis(ens_output, unstandardize=True)

        [test_ens_scalar_out,
         test_ens_metr_out] = \
            vutils.ens_fcast_verification(ens_output_full, fcast_outputs,
                                          test_dobjs, test_state, verif_spec,
                                          comp_yr_len,
                                          experiment_name+'X', avg_key,
                                          var_long_name, scalar_type_name,
                                          curr_fig_dir,
                                          do_hist=do_hist,
                                          do_reliability=do_reliability)

        scalar_out_fname = ('fcast_ens_scalar_output'
                            '_{}_{}.pkl'.format(experiment_name + 'X', avg_key))
        metr_out_fname = ('fcast_ens_metr_output'
                          '_{}_{}.pkl'.format(experiment_name + 'X', avg_key))

        output_fpath = os.path.join(curr_fig_dir, scalar_out_fname)
        output_fpath2 = os.path.join(curr_fig_dir, metr_out_fname)
        with open(output_fpath, 'wb') as f, open(output_fpath2, 'wb') as f2:
            pickle.dump(test_ens_scalar_out, f)
            pickle.dump(test_ens_metr_out, f2)

if do_long_integration:
#     state.proj_state_onto_eofs()
    t0 = state.data[0:1, :]

    last, avg = lutils.ens_long_integration(integration_iters,
                                            integration_len_yr,
                                            lim, t0)

    fname = 'long_integration_output_{}_{}.npz'.format(experiment_name,
                                                       avg_key)
    path = os.path.join(fig_dir, fname)
    np.savez(path, last=last, avg=avg)

    scalar_outs = vutils.long_output_to_scalar(last, dobjs, fcast_outputs,
                                               state, verif_spec,
                                               use_dask=True)
    fname = 'long_integration_{}_{}_{}.pkl'
    curr_path = os.path.join(fig_dir,
                             fname.format('last', experiment_name, avg_key))
    with open(curr_path, 'wb') as f:
        pickle.dump(scalar_outs, f)

    scalar_outs = vutils.long_output_to_scalar(avg, dobjs, fcast_outputs,
                                               state, verif_spec,
                                               use_dask=True)
    curr_path = os.path.join(fig_dir,
                             fname.format('avg', experiment_name, avg_key))
    with open(curr_path, 'wb') as f:
        pickle.dump(scalar_outs, f)

# dutils.close_hdf(dobjs)










