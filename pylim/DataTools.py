"""
Module for data interaction tools for the LIM package.

Author: Andre Perkins
"""

import tables as tb
import numpy as np
import os.path as path
import warnings
import netCDF4 as ncf
import numexpr as ne
import multiprocessing as mp

from Stats import run_mean, calc_anomaly
from scipy.signal import detrend


class BaseDataObject(object):
    """Data Input Object

    This class is for handling data which may be in a masked format. This
    class can also be used to expand previously compressed data if an
    original mask is provided.


    Notes
    -----
    Right now it is writen to work with 2D spatial data. It assumes that
    the leading dimension is temporal. In the future it might change to
    incorporate 3D spatial fields or just general data.

    Also might incorporate IRIS DataCubes to store data in the future.
    """

    # Static names
    TIME = 'time'
    LEVEL = 'level'
    LAT = 'lat'
    LON = 'lon'

    # Static databin keys
    _COMPRESSED = 'compressed'
    _ORIGDATA = 'orig'
    _DETRENDED = 'detrended'
    _AWGHT = 'area_weighted'
    _RUNMEAN = 'run_mean'
    _ANOMALY = 'anomaly'
    _CLIMO = 'climo'

    @staticmethod
    def _match_dims(shape, dim_coords):
        return {key: value[0] for key, value in dim_coords.items()
                if shape[value[0]] == len(value[1])}

    def __init__(self, data, dim_coords=None, valid_data=None, force_flat=False,
                 save_none=False,
                 is_anomaly=False, is_run_mean=False, is_detrended=False,
                 is_area_weighted=False):
        """
        Construction of a DataObject from input data.  If nan or
        infinite values are present, a compressed version of the data
        is also stored.

        Parameters
        ----------
        data: ndarray
            Input dataset to be used.
        dim_coords: dict(str:ndarray)
            Coordinate vector dictionary for supplied data.  Please use
            DataObject attributes (e.g. DataObject.TIME) for dictionary
            keys.
        valid_data: ndarray (np.bool), optional
            Array corresponding to valid data in the of the input dataset
            or uncompressed version of the input dataset.  Should have the same
            number of dimensions as the data and each  dimension should be
            greater than or equal to the spatial dimensions of data.
        force_flat: bool
            Force spatial dimensions to be flattened (1D array)
        is_anomaly: bool
            Data is already in anomaly form.
        is_run_mean: bool
            Data has been smoothed with a running mean.
        is_detrended: bool
            Data has been detrended.
        """

        assert data.ndim <= 4, 'Maximum of 4 dimensions are allowed.'
        self._full_shp = data.shape
        self.data_dtype = data.dtype
        self.forced_flat = force_flat
        self._save_none = save_none
        self._data_bins = {}
        self._curr_data_key = None

        # Future possible data manipulation functionality
        self.anomaly = None
        self.climo = None
        self.is_anomaly = is_anomaly
        self.running_mean = None
        self.is_run_mean = is_run_mean
        self.detrended = None
        self.is_detrended = is_detrended
        self.area_weighted = None
        self.is_area_weighted = is_area_weighted

        # Match dimension coordinate vectors
        if dim_coords is not None:
            if self.TIME in dim_coords.keys():
                # Assert leading dimension is sampling dimension
                assert dim_coords[self.TIME][0] == 0, 'Sampling dimension must'\
                    ' always be the leading dimension if provided.'
                self._leading_time = True
                self._time_shp = [data.shape[0]]
                self._spatial_shp = data.shape[1:]
            else:
                self._leading_time = False
                self._time_shp = []
                self._spatial_shp = self._full_shp
            self._dim_idx = self._match_dims(data.shape, dim_coords)
            self._dim_coords = dim_coords
        else:
            self._leading_time = False
            self._time_shp = []
            self._spatial_shp = self._full_shp
            self._dim_dix = None

        self._flat_spatial_shp = [np.product(self._spatial_shp)]

        # Check to see if data input is a compressed version
        self.is_masked = False
        compressed = False
        if valid_data is not None:
            dim_lim = valid_data.ndim

            assert dim_lim <= 3,\
                'mask should not have more than 3 dimensions.'
            assert dim_lim == len(self._spatial_shp),\
                'mask should have same number of spatial dimensions as data'

            # Check the dimensions of the mask and data to se if compressed
            for dat_dim, mask_dim in zip(self._spatial_shp, valid_data.shape):
                assert dat_dim <= mask_dim,\
                    'Valid data array provided should have larger ' +\
                    'spatial dimension than the masked input data.'
                compressed |= dat_dim < mask_dim

            # Apply input mask if its spatial dimensions match data
            if not compressed:
                # multplication broadcasts across leading sampling dimension if
                # applicable
                full_valid = np.ones(data.shape, dtype=np.bool) * valid_data
                data[~full_valid] = np.nan
            else:
                assert np.all(np.isfinite(data)),\
                    'Previously compressed data should not contain NaN data.'
                self._full_shp = self._time_shp + list(valid_data.shape)

            self.valid_data = valid_data.flatten()
            self.is_masked = True
        else:
            # Check to see if non-finite data requires use of mask
            if not np.alltrue(np.isfinite(data)):
                if self._leading_time:
                    valid_data = np.isfinite(data[0])
                    for time in data:
                        valid_data &= np.isfinite(time)
                    full_valid = np.ones(data.shape, dtype=np.bool) * valid_data
                    data[~full_valid] = np.nan
                else:
                    valid_data = np.isfinite(data)

                self.is_masked = True
                self.valid_data = valid_data.flatten()

        self.data = data
        self._curr_data_key = self._ORIGDATA
        # Flatten Spatial Dimension if applicable
        if force_flat or self.is_masked:
            if self._leading_time:
                self.data = data.reshape(self._time_shp +
                                         self._flat_spatial_shp)
            else:
                self.data = data.reshape(self._flat_spatial_shp)

        # Initialized here for flattening purposes
        if not save_none:
            self.orig_data = self._new_databin(self.data, self._ORIGDATA)

        # Compress the data if mask is present
        if compressed or self.is_masked:
            if compressed:
                self.compressed_data = self.orig_data
            elif self.is_masked:
                if self._leading_time:
                    self.data = self.data[:, self.valid_data]
                    if not save_none:
                        self.compressed_data = self._new_databin(
                            self.data,
                            self._COMPRESSED)
                else:
                    self.data = self.data[self.valid_data]
                    if not save_none:
                        self.compressed_data = self._new_databin(
                            self.data,
                            self._COMPRESSED)
            self._curr_data_key = self._COMPRESSED
        else:
            self.compressed_data = None

    # Create data backend container
    def _new_databin(self, data, name):
        new = np.empty_like(data)
        new[:] = data
        self._data_bins[name] = new
        return new

    def inflate_full_grid(self, data=None, reshape_orig=False):
        """
        Returns previously compressed data to its full grid filled with np.NaN
        values.

        Parameters
        ----------
        data: ndarray like, optional
            Data to inflate to its original grid size.
        reshape_orig: bool, optional
            If true use self._full_shp (shape of data used to initialize the
            DataObject) to reshape the inflated grid output

        Returns
        -------
        ndarray
            Full decompressed grid filled with NaN values in masked locations.
        """
        assert self.is_masked, 'Can only inflate compressed data.'

        if data is None:
            data = self.data

        shp = self._time_shp + list(self.valid_data.shape)
        full = np.empty(shp) * np.nan
        if self._leading_time:
            full[:, self.valid_data] = data
        else:
            full[self.valid_data] = data

        if reshape_orig:
            return full.reshape(self._full_shp)

        return full

    def calc_running_mean(self, window_size, save=True, **kwargs):
        assert self._leading_time, 'Can only perform running mean with a '\
            'specified leading sampling dimension'
        self.data, bedge, tedge = run_mean(self.data, window_size,
                                           **kwargs)
        if save and not self._save_none:
            self.running_mean = self._new_databin(self.data, self._RUNMEAN)
        self._curr_data_key = self._RUNMEAN
        self.is_run_mean = True
        # Running mean smooths data, no longer an anomaly or detrended
        self.is_anomaly = False
        self.is_detrended = False
        return self.data, bedge, tedge

    # TODO: Use provided time coordinates to determine year size
    # TODO: Determine if climo needs to be tied to object
    def calc_anomaly(self, yr_size, save=True, climo=None):
        assert self._leading_time, 'Can only perform anomaly calculation with '\
            'a specified leading sampling dimension'
        self.data, climo = calc_anomaly(self.data, yr_size, climo=climo)

        if save and not self._save_none:
            self.anomaly = self._new_databin(self.data, self._ANOMALY)
        self.climo = climo
        self._curr_data_key = self._ANOMALY
        self.is_anomaly = True
        return self.anomaly

    def detrend_data(self, save=True):
        assert self._leading_time, 'Can only perform anomaly calculation with '\
            'a specified leading sampling dimension'
        self.data = detrend(self.data, axis=0, type='linear')
        if save and not self._save_none:
            self.detrended = self._new_databin(self.data, self._DETRENDED)
        self._curr_data_key = self._DETRENDED
        self.is_detrended = True
        return self.detrended

    def area_weight_data(self, save=True):
        if self.LAT not in self._dim_idx.keys():
            warnings.warn('No latitude dimension specified. No area weighting'
                          'was performed.')
            return self.data
        lats = self.get_coordinate_grids([self.LAT])[self.LAT]
        scale = np.sqrt(np.cos(np.radians(lats)))
        awgt = self.data
        self.data = ne.evaluate('awgt * scale')
        if save and not self._save_none:
            self.area_weighted = self._new_databin(self.data,
                                                   self._AWGHT)
        self._curr_data_key = self._AWGHT
        self.is_area_weighted = True

    def get_dim_coords(self, keys):
        dim_coords = {}

        for key in keys:
            if key in self._dim_coords.keys():
                dim_coords[key] = self._dim_coords[key]

        return dim_coords

    def get_coordinate_grids(self, keys, compressed=True):
        grids = {}

        for key in keys:
            assert key in self._dim_idx.keys(), 'No matching dimension for key'\
                '({}) was found in data shape'.format(key)

            idx = self._dim_idx[key]
            if self._leading_time:
                idx -= 1
            coords = self._dim_coords[key][1]
            grid = np.copy(coords)
            for dim, _ in enumerate(self._spatial_shp):
                if dim != idx:
                    grid = np.expand_dims(grid, dim)

            grid = np.ones(self._spatial_shp) * grid
            if self.is_masked or self.forced_flat:
                grid = grid.flatten()

                if self.is_masked and compressed:
                    grid = grid[self.valid_data]
                elif self.is_masked:
                    grid[self.valid_data] = np.nan

            grids[key] = grid

        return grids

    # TODO: figure out consisntent state booleans is_detrended, etc.
    def reset_data(self, key):
        try:
            self.data = self._data_bins[key][:]
        except KeyError:
            raise KeyError('Key {} not saved.  Could not reset self.data.')

        return self.data


class Hdf5DataObject(BaseDataObject):

    def __init__(self, data, h5file, dim_coords=None, valid_data=None,
                 force_flat=False, is_anomaly=False, is_run_mean=False,
                 is_detrended=False, default_grp='/data'):
        """
        Construction of a Hdf5DataObject from input data.  If nan or
        infinite values are present, a compressed version of the data
        is also stored.

        Parameters
        ----------
        data: ndarray
            Input dataset to be used.
        h5file: tables.File
            HDF5 Pytables file to use as a data storage backend
        dim_coords: dict(str:ndarray), optional
            Coordinate vector dictionary for supplied data.  Please use
            DataObject attributes (e.g. DataObject.TIME) for dictionary
            keys.
        valid_data: ndarray (np.bool), optional
            Array corresponding to valid data in the of the input dataset
            or uncompressed version of the input dataset.  Should have the same
            number of dimensions as the data and each  dimension should be
            greater than or equal to the spatial dimensions of data.
        force_flat: bool
            Force spatial dimensions to be flattened (1D array)
        is_anomaly: bool
            Data is already in anomaly form.
        is_run_mean: bool
            Data has been smoothed with a running mean.
        is_detrended: bool
            Data has been detrended.
        default_grp: tables.Group or str, optional
            Group to store all created databins under in the hdf5 file.

        Notes
        -----
        If NaN values are present I do not suggest
        using the orig_data variable when reloading from a file.  Currently
        PyTables Carrays have no method of storing np.NaN so the values in those
        locations will be random.  Please only read the compressed data or make
        sure you apply the mask on the data if you think self.orig_data is being
        read from disk.
        """

        assert type(h5file) == tb.File,\
            'Hdf5DataObject only works with PyTables.File types'
        assert h5file.mode in ['a', 'w'], \
            'h5file is not write enabled'
        assert h5file.isopen,\
            'h5file is closed and therefore not write enabled'

        self.h5f = h5file
        self._default_grp = None
        self.set_databin_grp(default_grp)

        super(Hdf5DataObject, self).__init__(data,
                                             dim_coords=dim_coords,
                                             valid_data=valid_data,
                                             force_flat=force_flat,
                                             is_anomaly=is_anomaly,
                                             is_run_mean=is_run_mean,
                                             is_detrended=is_detrended)

    # Create backend data container
    def _new_databin(self, data, name):
        new = var_to_hdf5_carray(self.h5f,
                                 self._default_grp,
                                 name,
                                 data)
        self._data_bins[name] = new
        return new

    def set_databin_grp(self, group):
        """
        Set the default PyTables group for databins to be created under in the
        HDF5 File.  This overwrites existing nodes with the same name and will
        create the full path necessary to reach the desired node.

        Parameters
        ----------
        group: tables.Group or str
            A PyTables group object or string path to set as the default group
            for the HDF5 backend to store databins.
        """
        assert type(group) == tb.Group or type(group) == str, \
            'default_grp must be of type tb.Group or a path string'
        try:
            self._default_grp = self.h5f.get_node(group)
            try:
                assert type(self._default_grp) == tb.Group
            except AssertionError:
                self.h5f.remove_node(self._default_grp)
                raise tb.NoSuchNodeError
        except tb.NoSuchNodeError:
            if type(group) == tb.Group:
                grp_path = path.split(group._v_pathname)
            else:
                grp_path = path.split(group)

            self._default_grp = self.h5f.create_group(grp_path[0], grp_path[1],
                                                      createparents=True)


def var_to_hdf5_carray(h5file, group, node, data, **kwargs):
    """
    Take an input data and insert into a PyTables carray in an HDF5 file.

    Parameters
    ----------
    h5file: tables.File
        Writeable HDF5 file to insert the carray into.
    group: str, tables.Group
        PyTables group to insert the data node into
    node: str, tables.Node
        PyTables node of the carray.  If it already exists it will remove
        the existing node and create a new one.
    data: ndarray
        Data to be inserted into the node carray
    kwargs:
        Extra keyword arguments to be passed to the
        tables.File.create_carray method.

    Returns
    -------
    tables.carray
        Pointer to the created carray object.
    """
    assert(type(h5file) == tb.File)

    # Switch to string
    if type(group) != str:
        group = group._v_pathname

    # Join path for node existence check
    if group[-1] == '/':
        node_path = group + node
    else:
        node_path = '/'.join((group, node))

    # Check existence and remove if necessary
    if h5file.__contains__(node_path):
        h5file.remove_node(node_path)

    out_arr = h5file.create_carray(group,
                                   node,
                                   atom=tb.Atom.from_dtype(data.dtype),
                                   shape=data.shape,
                                   **kwargs)
    out_arr[:] = data
    return out_arr


def empty_hdf5_carray(h5file, group, node, in_atom, shape, **kwargs):
    """
    Create an empty PyTables carray.  Replaces node if it already exists.

    Parameters
    ----------
    h5file: tables.File
        Writeable HDF5 file to insert the carray into.
    group: str, tables.Group
        PyTables group to insert the data node into
    node: str, tables.Node
        PyTables node of the carray.  If it already exists it will remove
        the existing node and create a new one.
    in_atom: tables.Atom
        Atomic datatype and chunk size for the carray.
    shape: tuple, list
        Shape of empty carray to be created.
    kwargs:
        Extra keyword arguments to be passed to the
        tables.File.create_carray method.

    Returns
    -------
    tables.carray
        Pointer to the created carray object.
    """
    assert(type(h5file) == tb.File)

    # Switch to string
    if type(group) == tb.Group:
        group = group._v_pathname

    # Join path for node existence check
    if group[-1] == '/':
        node_path = group + node
    else:
        node_path = '/'.join((group, node))

    # Check existence and remove if necessary
    if h5file.__contains__(node_path):
        h5file.remove_node(node_path)

    out_arr = h5file.create_carray(group,
                                   node,
                                   atom=in_atom,
                                   shape=shape,
                                   **kwargs)
    return out_arr


def netcdf_to_data_obj(filename, var_name, h5file=None):

    f = ncf.Dataset(filename, 'r')

    try:
        data = f.variables[var_name][:]
        coords = {BaseDataObject.LAT: f.variables['lat'][:],
                  BaseDataObject.LON: f.variables['lon'][:]}
        times = f.variables['time']
        coords[BaseDataObject.TIME] = ncf.num2date(times[:], times.units)

        for i, key in enumerate(f.dimensions.iterkeys()):
            if key in coords.keys():
                coords[key] = (i, coords[key])

        if h5file is not None:
            return Hdf5DataObject(data, h5file, dim_coords=coords, force_flat=True)

        else:
            return BaseDataObject(data, dim_coords=coords, force_flat=True)

    finally:
        f.close()

def netcdf_to_hdf5_container(infile, var_name, outfile, data_dir='/'):
    f = ncf.Dataset(infile, 'r')
    outf = tb.open_file(outfile, 'w', filters=tb.Filters(complib='blosc',
                                                         complevel=5))

    try:
        data = f.variables[var_name]
        atom = tb.Atom.from_dtype(data.datatype)
        shape = data.shape
        out = empty_hdf5_carray(outf, data_dir, var_name, atom, shape)
        for i, chunk in enumerate(data):
                out[i] = chunk

        lat = var_to_hdf5_carray(outf, data_dir, 'lat',
                                 f.variables['lat'][:])
        lon = var_to_hdf5_carray(outf, data_dir, 'lon',
                                 f.variables['lon'][:])
        times = f.variables['time']
        time_out = var_to_hdf5_carray(outf, data_dir, 'time',
                                      times[:])
        time_out.attrs.units = times.units

        coord_dims = {'lat': lat.attrs, 'lon': lon.attrs,
                      'time': time_out.attrs}

        for i, key in enumerate(f.dimensions.iterkeys()):
            if key in coord_dims.keys():
                coord_dims[key].index = i
    finally:
        f.close()
        outf.close()

def hdf5_to_data_obj(filename, var_name, h5file=None, data_dir='/'):

    f = tb.open_file(filename, 'r')

    try:
        data = f.get_node(data_dir, name=var_name)[:]
        lat = f.get_node(data_dir+'lat')
        lon = f.get_node(data_dir+'lon')
        coords = {BaseDataObject.LAT: (lat.attrs.index, lat[:]),
                  BaseDataObject.LON: (lon.attrs.index, lon[:])}
        times = f.get_node(data_dir+'time')
        coords[BaseDataObject.TIME] = (times.attrs.index,
                                       ncf.num2date(times[:],
                                                    times.attrs.units))

        if h5file is not None:
            return Hdf5DataObject(data, h5file, dim_coords=coords,
                                  force_flat=True)

        else:
            return BaseDataObject(data, dim_coords=coords, force_flat=True)

    finally:
        f.close()
