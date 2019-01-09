import pylim.DataTools as DT
import pylim.Stats as ST

import os
import tables as tb

RECALC_EOFS = True

def netcdf_to_hdf5(var_fpath_dict, out_dir, out_fname):
    for key, fname in var_fpath_dict.items():
        print('Converting {} file to HDF5...'.format(key))
        out_fpath = os.path.join(out_dir, out_fname.format(key))
        DT.netcdf_to_hdf5_container(fname, key, out_fpath)


def load_dobjs(var_names, h5_fname, cell_area_map, ncf_fname,
               preproc_fname, use_preproc_files):
    # if we're doing pre-calib load that, if not then look for h5
    if use_preproc_files:
        try:
            dobjs = load_dobjs_pkl(var_names, preproc_fname + '.pkl')
        except IOError as e:
            print('Pre-processed file not found: ', e)
            dobjs = load_dobjs_hdf5(var_names, h5_fname,
                                    preproc_fname + '.h5',
                                    ncf_fname,
                                    cell_area_map=cell_area_map)
    else:
        dobjs = load_dobjs_hdf5(var_names, h5_fname,
                                preproc_fname + '.h5',
                                ncf_fname, cell_area_map=cell_area_map)

    return dobjs

# Loading data objects
def load_dobjs_hdf5(load_keys, in_fpath, out_h5_fpath,
                    ncf_path, cell_area_map=None):

    var_long_post = {'tas': 'tas_sfc_Amon',
                     'tos': 'tos_sfc_Omon',
                     'zos': 'zos_sfc_Omon'}
    dobjs = {}
    
    for key in load_keys:
        infile = in_fpath.format(key)
        outfile = out_h5_fpath.format(key)
        
        if os.path.exists(outfile):
            os.remove(outfile)
            
        dobj_h5 = tb.open_file(outfile,
                               mode='w',
                               title='Data object for var: {}'.format(key),
                               filters=tb.Filters(complib='blosc',
                                                  complevel=2))
        
        if cell_area_map is not None:
            cell_area_path = cell_area_map.get(key, None)
        else:
            cell_area_path = None

        if not os.path.exists(infile):
            curr_var_ncf = ncf_path.format(var_long_post[key])
            DT.netcdf_to_hdf5_container(curr_var_ncf, key, infile)

        dobjs[key] = DT.Hdf5DataObject.from_hdf5(infile, key, dobj_h5,
                                                 cell_area_path=cell_area_path)
        
    return dobjs

def load_dobjs_pkl(load_keys, in_fpath):
    dobjs = {}
    
    for key in load_keys:
        dobjs[key] = DT.Hdf5DataObject.from_pickle(in_fpath.format(key))

    return dobjs

def prep_dobjs_for_eof(nelem_in_yr, dobjs):
    did_op = False

    for key, dobj in dobjs.items():
        if dobj._EOFPROJ in dobj._ops_performed[dobj._curr_data_key]:
            continue
        else:
            print(f'Prepping data object: {key}')
            if dobj._ANOMALY not in dobj._ops_performed[dobj._curr_data_key]:
                print('Changing to anomaly...')
                dobj.calc_anomaly(nelem_in_yr)
                did_op |= True
            if dobj._DETRENDED not in dobj._ops_performed[dobj._curr_data_key]:
                dobj.detrend_data()
                print('Detrending data...')
                did_op |= True
            if dobj._AWGHT not in dobj._ops_performed[dobj._curr_data_key]:
                print('Area weighting for EOF calculation...')
                dobj.area_weight_data()
                did_op |= True

    return did_op


def standardize_dobjs(dobjs, std_dobjs=None):

    did_op = False
    for key, dobj in dobjs.items():
        if std_dobjs is not None:
            std_factor = std_dobjs[key]._std_scaling
            dobj.standardize_data(std_factor=std_factor)
            did_op |= True
        elif not dobj._STD in dobj._ops_performed[dobj._curr_data_key]:
            dobj.standardize_data()
            did_op |= True

    return did_op


def check_proper_average(dobjs, avg_key, avg_func_args):
    # Check to see that consistently averaged
    for var_key, dobj in dobjs.items():
        if avg_key == 'none':
            for avg in ['ann_std', 'ann_fall', 'ann_spr']:
                if avg in dobj._ops_performed[dobj._curr_data_key]:
                    _reset_orig(dobj)

        elif avg_key not in dobj._ops_performed[dobj._curr_data_key]:
            try:
                dobj.reset_data(avg_key)
            except KeyError:
                _reset_orig(dobj)
                dobj.time_average_resample(*avg_func_args['args'],
                                           **avg_func_args['kwargs'])


def _reset_orig(dobj):
    if dobj.is_masked:
        dobj.reset_data(dobj._COMPRESSED)
    else:
        dobj.reset_data(dobj._ORIGDATA)

        
def print_eof_var_stats(dobj, name='VAR'):
    stats = dobj.get_eof_stats()
    
    print('Field EOFS: {}'.format(name))
    print('\tNum EOFS: {}'.format(stats['num_ret_modes']))
    print('\tVar by mode: ', stats['var_expl_by_mode'][0:10])
    print('\tTot Var Expl: {:2.1f}%'.format(stats['var_expl_by_ret']*100))
    
def calc_eofs_dobjs(dobjs, num_eofs=10, print_stats=False, eof_dobjs=None):
    did_op = False
    for key, dobj in dobjs.items():
        if eof_dobjs is not None:
            dobj.eof_proj_data(eof_in=eof_dobjs[key]._eofs,
                               proj_key=dobj._DETRENDED)
            print_stats = False
            did_op = False
        elif (dobj._EOFPROJ not in dobj._ops_performed[dobj._curr_data_key]
              or RECALC_EOFS):
            dobj.eof_proj_data(num_eofs=num_eofs, proj_key=dobj._DETRENDED)
            did_op |= True
        
        if print_stats:
            print_eof_var_stats(dobj, name=key)

    return did_op

def save_dobjs(dobjs, out_pkl_fname):
    for var, dobj in dobjs.items():
        fname = out_pkl_fname.format(var)
        dobj.save_dataobj_pckl(fname)
            
def close_hdf(dobjs):
    for dobj in dobjs.values():
        dobj.h5f.close()
        
def reset_dobjs_to_key(key, dobjs):
    for dobj in dobjs.values():
        dobj.reset_data(key)

def get_lat_lon_grids(dobj, compressed=False, flat=False):

    grids = dobj.get_coordinate_grids(['lat', 'lon'], compressed=compressed,
                                      flat=flat)
    lat = grids['lat']
    lon = grids['lon']

    return lat, lon

def reinflate_field(dobj, field):

    if dobj.is_masked:
        field = dobj.inflate_full_grid(data=field, reshape_orig=True)
    else:
        field = field.reshape(dobj._spatial_shp)

    return field

def eofs_state_basis_to_orig_basis(field, dobjs, state):

    for var, dobj in dobjs.items():
        state_eofs = state.get_var_eofs(var)
        dobj_eofs = dobj._eofs

        full_field = dobj_eofs @ state_eofs @ field

