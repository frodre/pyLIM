import os
import numpy as np
import pandas as pd

def build_ncf_fpaths(parent_dir, exp_names, exp_folders, exp_files,
                     var_names):
    var_mapping = {'tas': 'atmos',
                   'tos': 'ocean',
                   'zos': 'ocean'}
    var_long_post = {'atmos': '_sfc_Amon',
                     'ocean': '_sfc_Omon'}
    netcdf_paths = {}
    for exp in exp_names:
        exp_paths = {}
        for var in var_names:
            file_dir = os.path.join(parent_dir, exp_folders[exp])
            var_realm = var_mapping[var]
            var_name = var + var_long_post[var_realm]
            file_name = exp_files[exp].format(var_name, exp)
            path = os.path.join(file_dir, file_name)

            exp_paths[var] = path
        netcdf_paths[exp] = exp_paths


def calculate_latlon_bnds(lats, lons, lat_ax=0):
    """
    Calculate the bounds for gridded lats and lons.

    Parameters
    ----------
    lats: ndarray
        Latitude array. Can be either 1D or 2D.
    lons:  ndarray
        Longitude array.  Can be either 1D or 2D.

    Returns
    -------
    lat_bnds:
        Array of latitude boundaries for each input latitude.
    lon_bnds:
        Array of longitude boundaries for each input longitude.

    Notes
    -----
    This function was originally built with regularly gridded coordinate
    dimensions in mind.  It was extended to be useful for conservative
    regridding of irregular grids, but only tested on Rotated Pole grids.
    When using funky grids it is best to make sure output makes sense.

    """
    if not 1 <= lats.ndim <= 2 or not 1 <= lons.ndim <= 2:
        raise ValueError('Input lats and lons must be 1D or 2D')
    if lats.ndim != lons.ndim:
        raise ValueError(
            'Input lats and lons must have the same dimensions')
    if lats.ndim > 1 and lats.shape != lons.shape:
        raise ValueError('2D lats and lons must have same array shape.')

    if lats.ndim == 2:
        lon_ax = int(np.logical_not(lat_ax))
        bnd_2d = True
    else:
        lat_ax = lon_ax = 0
        bnd_2d = False

    dim_order = sorted([lat_ax, lon_ax])

    lat_space = np.diff(lats, axis=lat_ax)
    lon_space = np.diff(lons, axis=lon_ax)

    nlat = lats.shape[lat_ax]
    nlon = lons.shape[lon_ax]

    lat_bnd_shp = [dim_len + 1 for dim_len in lats.shape]
    lon_bnd_shp = [dim_len + 1 for dim_len in lons.shape]

    lat_bnds = np.zeros(lat_bnd_shp)
    lon_bnds = np.zeros(lon_bnd_shp)

    # Handle cyclic point if necessary
    if bnd_2d:
        if np.any(lon_space > 300):
            i_idx, j_idx = np.where(lon_space > 300)
            lon_space[i_idx, j_idx] = lon_space[i_idx, j_idx + 1]

        if np.any(lon_space < -300):
            i_idx, j_idx = np.where(lon_space < -300)
            lon_space[i_idx, j_idx] = lon_space[i_idx, j_idx - 1]
    else:
        if np.any(lon_space > 300):
            i_idx, = np.where(lon_space > 300)
            lon_space[i_idx] = lon_space[i_idx + 1]

        if np.any(lon_space < -300):
            i_idx, = np.where(lon_space < -300)
            lon_space[i_idx] = lon_space[i_idx - 1]

    # Handle out of bounds latitudes
    lat_space[lat_space > 90] = 90
    lat_space[lat_space < -90] = -90

    lon_sl = slice(0, nlon - 1)
    lat_sl = slice(0, nlat - 1)
    all_but_last = slice(0, -1)
    last_two = slice(-2, None)
    all_vals = slice(None)

    # TODO: Not an elegant solution for variable dimension order but I think it
    # works...
    if bnd_2d:
        # Create slices to be used for general dimension order
        bnd_slice = (all_but_last, lon_sl)
        coord_slice = (all_vals, all_but_last)
        bnd_end_slice = (all_but_last, last_two)
        coord_end_slice = (all_vals, last_two)
        diff_end_slice = (all_vals, last_two)
        cyclic_bnd_dst = (-1, all_vals)
        cyclic_bnd_src = (-2, all_vals)

        # If lon changes over first dimension we want to reverse the slice
        # tuples
        if lon_ax == 0:
            rev = -1
        else:
            rev = 1

        lon_bnds[bnd_slice[::rev]] = lons[coord_slice[
                                          ::rev]] - lon_space / 2
        lon_bnds[bnd_end_slice[::rev]] = (lons[coord_end_slice[::rev]] +
                                          lon_space[diff_end_slice[
                                                    ::rev]] / 2)
        lon_bnds[cyclic_bnd_dst[::rev]] = lon_bnds[
            cyclic_bnd_src[::rev]]

        # Adjust the bnd slice for latitude dimension
        bnd_slice = (all_but_last, lat_sl)

        # If lat changes over first dimension we want to reverse the slice
        # tuples
        if lat_ax == 0:
            rev = -1
        else:
            rev = 1

        lat_bnds[bnd_slice[::rev]] = lats[coord_slice[
                                          ::rev]] - lat_space / 2
        lat_bnds[bnd_end_slice[::rev]] = (lats[coord_end_slice[::rev]] +
                                          lat_space[diff_end_slice[
                                                    ::rev]] / 2)
        lat_bnds[cyclic_bnd_dst[::rev]] = lat_bnds[
            cyclic_bnd_src[::rev]]

    else:
        lon_bnds[lon_sl] = lons[all_but_last] - lon_space / 2
        lon_bnds[last_two] = lons[last_two] + lon_space[last_two] / 2

        lat_bnds[lat_sl] = lats[all_but_last] - lat_space / 2
        lat_bnds[last_two] = lats[last_two] + lat_space[last_two] / 2

    return lat_bnds, lon_bnds


def ce_r_results_to_dataframe(var_key, avg_key, output_type,
                              r, r_conf95, auto1_r, auto1_r_conf95,
                              ce, ce_conf95, auto1_ce, auto1_ce_conf95):

    dat_list = [r, r_conf95, auto1_r, auto1_r_conf95, ce, ce_conf95,
                auto1_ce, auto1_ce_conf95]

    new_dat_list = []
    for item in dat_list:
        if isinstance(item, tuple):
            new_dat_list.append(item[1])
            new_dat_list.append(item[0])
        elif item is None:
            new_dat_list.append(None)
            new_dat_list.append(None)
        else:
            new_dat_list.append(item)

    columns = ['r', 'r(2.5%)', 'r(97.5%)',
               'auto1_r', 'auto1_r(2.5%)', 'auto1_r(97.5%)',
               'ce', 'ce(2.5%)', 'ce(97.5%)',
               'auto1_ce', 'auto1_ce(2.5%)', 'auto1_ce(97.5%)']
    index = pd.MultiIndex.from_tuples(((var_key, avg_key, output_type),),
                                      names=['Variable', 'Average',
                                             'ScalarType'])
    df = pd.DataFrame(index=index,
                      columns=columns,
                      data=np.array([new_dat_list, ]))

    return df
