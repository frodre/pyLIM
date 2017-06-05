"""
Module for data interaction tools for the LIM package.

Author: Andre Perkins
"""

import tables as tb
import dask.array as da
import numpy as np
import os.path as path
import warnings
import netCDF4 as ncf
import numexpr as ne
import multiprocessing as mp
import cPickle as cpk

from Stats import run_mean, calc_anomaly
from scipy.signal import detrend

tb.parameters.NODE_CACHE_SLOTS = 0


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
                 save_none=False, time_units=None, time_cal=None,
                 fill_value=None):
        """
        Construction of a DataObject from input data.  If nan or
        infinite values are present, a compressed version of the data
        is also stored.

        Parameters
        ----------
        data: ndarray
            Input dataset to be used.
        dim_coords: dict(str:(int, ndarray)
            Demension position and oordinate vector dictionary for supplied
            data.  Please use DataObject attributes (e.g. DataObject.TIME)
            for dictionary keys.
        valid_data: ndarray (np.bool), optional
            Array corresponding to valid data in the of the input dataset
            or uncompressed version of the input dataset.  Should have the same
            number of dimensions as the data and each  dimension should be
            greater than or equal to the spatial dimensions of data.
        force_flat: bool
            Force spatial dimensions to be flattened (1D array)
        fill_value: float
            Value to be considered invalid data during the mask and 
            compression. Only considered when data is not masked.
        """

        assert data.ndim <= 4, 'Maximum of 4 dimensions are allowed.'
        self._full_shp = data.shape
        self.data_dtype = data.dtype
        self.forced_flat = force_flat
        self.time_units = time_units
        self.time_cal = time_cal
        self._fill_value = fill_value
        self._save_none = save_none
        self._data_bins = {}
        self._curr_data_key = None
        self._ops_performed = []

        # Future possible data manipulation functionality
        self.anomaly = None
        self.climo = None
        self.compressed_data = None
        self.running_mean = None
        self.detrended = None
        self.area_weighted = None

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

        # Masked array valid handling
        self.is_masked, self.valid_data = self._data_masking(data)
        if self.valid_data is not None:
            self.valid_data = self.valid_data.flatten()
            self.mask = np.logical_not(self.valid_data)

        # Flatten Spatial Dimension if applicable
        if force_flat or self.is_masked:
            if self._leading_time:
                self.data = data.reshape(self._time_shp +
                                         self._flat_spatial_shp)
            else:
                self.data = data.reshape(self._flat_spatial_shp)
        else:
            self.data = data

        # Initialized here for flattening purposes
        if not save_none:
            self.orig_data = self._new_databin(self.data, self._ORIGDATA)
        self._set_curr_data_key(self._ORIGDATA)

        # Compress the data if mask is present
        if compressed:
            self.compressed_data = self.orig_data
        elif self.is_masked:
            if not save_none:
                if self._leading_time:
                    new_shp = (self._time_shp[0], self.valid_data.sum())
                else:
                    new_shp = (self.valid_data.sum(),)
                self.compressed_data = self._new_empty_databin(new_shp,
                                                               self.data.dtype,
                                                               self._COMPRESSED)

            self.data = self._compress_masked_data(self.data,
                                                   self.mask,
                                                   out_arr=self.compressed_data)
            self._set_curr_data_key(self._COMPRESSED)

    def _set_curr_data_key(self, key):
        self._curr_data_key = key

    def _new_empty_databin(self, shape, dtype, name):
        """
        Create an empty backend data container.
        """
        new = np.empty(shape, dtype=dtype)
        self._data_bins[name] = new
        return new

    def _new_databin(self, data, name):
        """
        Create and copy data into a new backend data container.
        """
        new = np.empty_like(data)
        new[:] = data
        self._data_bins[name] = new
        return new

    def _gen_composite_mask(self, data):
        if self._leading_time:
            composite_mask = data.mask.sum(axis=0) > 0
        else:
            composite_mask = data.mask

        return composite_mask

    def _check_invalid_data(self, data):
        full_valid = np.isfinite(data)
        if self._fill_value is not None:
            full_valid &= data != self._fill_value

        if not np.all(full_valid):
            masked = True
            if self._leading_time:
                valid_data = full_valid.sum() < self._time_shp[0]
            else:
                valid_data = full_valid
        else:
            masked = False
            valid_data = None

        return masked, valid_data

    def _data_masking(self, data):
        if np.ma.is_masked(data[0]):
            masked = True
            composite_mask = self._gen_composite_mask(data)
            valid_data = np.logical_not(composite_mask)
        else:
            masked, valid_data = self._check_invalid_data(data)

        return masked, valid_data

    def _compress_masked_data(self, data, mask, out_arr=None):
        if self._leading_time:
            compress_axis=1
        else:
            compress_axis=None

        out_arr = np.compress(mask, data, axis=compress_axis, out=out_arr)
        return out_arr

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

    def calc_running_mean(self, window_size, year_len, save=True):

        # TODO: year_len should eventually be a property determined during init
        if not self._leading_time:
            raise ValueError('Can only perform a running mean when data has a '
                             'leading sampling dimension.')

        edge_pad = window_size // 2
        edge_trim = np.ceil(edge_pad / float(year_len)) * year_len

        if save and not self._save_none:
            new_time = self.data.shape[0] - edge_trim * 2
            new_shape = list(self.data.shape)
            new_shape[0] = new_time
            new_shape = tuple(new_shape)
            self.running_mean = self._new_empty_databin(new_shape,
                                                        self.data.dtype,
                                                        self._RUNMEAN)

        self.data = run_mean(self.data, window_size, trim_edge=edge_trim,
                             output_arr=self.running_mean)
        self._set_curr_data_key(self._RUNMEAN)

        return self.data

    # TODO: Use provided time coordinates to determine year size
    # TODO: Determine if climo needs to be tied to object
    def calc_anomaly(self, yr_size, save=True, climo=None):
        assert self._leading_time, 'Can only perform anomaly calculation with '\
            'a specified leading sampling dimension'

        if save and not self._save_none:
            self.anomaly = self._new_empty_databin(self.data.shape,
                                                   self.data.dtype,
                                                   self._ANOMALY)

        self.data, self.climo = calc_anomaly(self.data, yr_size,
                                             climo=climo,
                                             output_arr=self.anomaly)

        self._set_curr_data_key(self._ANOMALY)
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
        scale = np.sqrt(abs(np.cos(np.radians(lats))))
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
            if self._leading_time:
                # running mean alters the time TODO: check that this works
                self._time_shp = [self.data.shape[0]]
        except KeyError:
            raise KeyError('Key {} not saved.  Could not reset self.data.')

        return self.data

    def is_leading_time(self):
        return self._leading_time

    def save_dataobj_pckl(self, filename):

        tmp_dimcoord = self._dim_coords[self.TIME]
        tmp_time = tmp_dimcoord[1]
        topckl_time = ncf.date2num(tmp_time, units=self.time_units,
                                   calendar=self.time_cal)
        self._dim_coords[self.TIME] = (tmp_dimcoord[0], topckl_time)

        with open(filename, 'w') as f:
            cpk.dump(self, f)

        self._dim_coords[self.TIME] = (tmp_dimcoord[0], tmp_time)

    @classmethod
    def from_netcdf(cls, filename, var_name, **kwargs):

        with ncf.Dataset(filename, 'r') as f:
            data = f.variables[var_name][:]
            coords = {BaseDataObject.LAT: f.variables['lat'][:],
                      BaseDataObject.LON: f.variables['lon'][:]}
            times = f.variables['time']

            try:
                cal = times.calendar
                coords[BaseDataObject.TIME] = ncf.num2date(times[:], times.units,
                                                           calendar=cal)
            except AttributeError:
                coords[BaseDataObject.TIME] = ncf.num2date(times[:], times.units)
                cal=None

            for i, key in enumerate(data.dimensions):
                if key in coords.keys():
                    coords[key] = (i, coords[key])

            force_flat = kwargs.pop('force_flat', True)
            return cls(data, dim_coords=coords, force_flat=force_flat,
                       time_units=times.units, time_cal=cal, **kwargs)

    @classmethod
    def from_hdf5(cls, filename, var_name, data_dir='/'):

        with tb.open_file(filename, 'r') as f:

            data = f.get_node(data_dir, name=var_name)
            if data.attrs.masked:
                fill_val = data.attrs.fill_value
                data = data[:]
                mask = data == fill_val
                data = np.ma.array(data,
                                   mask=mask,
                                   fill_value=data.attrs.fill_value)
            else:
                data = data[:]
            lat = f.get_node(data_dir+'lat')
            lon = f.get_node(data_dir+'lon')
            coords = {BaseDataObject.LAT: (lat.attrs.index, lat[:]),
                      BaseDataObject.LON: (lon.attrs.index, lon[:])}
            times = f.get_node(data_dir+'time')
            coords[BaseDataObject.TIME] = (times.attrs.index,
                                           ncf.num2date(times[:],
                                                        times.attrs.units))
            return cls(data, dim_coords=coords, force_flat=True)

    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'r') as f:
            dobj = cpk.load(f)

        tmp_dimcoord = dobj._dim_coords[dobj.TIME]
        tmp_time = tmp_dimcoord[1]
        topckl_time = ncf.num2date(tmp_time, units=dobj.time_units,
                                   calendar=dobj.time_cal)
        dobj._dim_coords[dobj.TIME] = (tmp_dimcoord[0], topckl_time)

        return dobj


class Hdf5DataObject(BaseDataObject):

    def __init__(self, data, h5file, dim_coords=None, valid_data=None,
                 force_flat=False, fill_value=None, chunk_shape=None,
                 default_grp='/data'):
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
            Data has been detrended.
        fill_value: float
            Value to be considered invalid data during the mask and 
            compression. Only considered when data is not masked.
            
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

        if chunk_shape is None:
            leading_time = BaseDataObject.TIME in dim_coords
            self._chunk_shape = self._determine_chunk(leading_time,
                                                      data.shape,
                                                      data.dtype)
        else:
            self._chunk_shape = chunk_shape

        data = da.from_array(data, chunks=self._chunk_shape)

        super(Hdf5DataObject, self).__init__(data,
                                             dim_coords=dim_coords,
                                             valid_data=valid_data,
                                             force_flat=force_flat,
                                             fill_value=fill_value)

    def _set_curr_data_key(self, key):
        if not hasattr(self.data, 'dask'):
            chunk_shp = self._determine_chunk(self._leading_time,
                                              self.data.shape,
                                              self.data.dtype)
            self._chunk_shape = chunk_shp
            self.data = da.from_array(self.data, chunks=self._chunk_shape)
        super(Hdf5DataObject, self)._set_curr_data_key(key)

    # Create backend data container
    def _new_empty_databin(self, shape, dtype, name):
        new = empty_hdf5_carray(self.h5f,
                                self._default_grp,
                                name,
                                tb.Atom.from_dtype(dtype),
                                shape
                                )
        self._data_bins[name] = new
        return new

    def _new_databin(self, data, name):
        new = self._new_empty_databin(data.shape, data.dtype, name)
        da.store(data, new)
        self._data_bins[name] = new
        return new

    def _determine_chunk(self, leading_time, shape, dtype, size=10):
        """
        Determine default chunk size for dask array operations.
        
        shape: tuple<int>
            Shape of the data to be chunked.
        dype: numpy.dtype
            Datatype of the data to be chunked
        size: int
            Size (in MB) of the desired chunk
        """
        if leading_time:
            sptl_size = np.product(shape[1:]) * dtype.itemsize
            rows_in_chunk = size*1024**2 // sptl_size
            rows_in_chunk = int(rows_in_chunk)
            chunk = tuple([rows_in_chunk] + list(shape[1:]))
        else:
            nelem = np.product(shape)
            elem_in_chunk = nelem*dtype.itemsize // (size * 1024**2)

            if elem_in_chunk == 0:
                chunk = shape
            else:
                dim_len = elem_in_chunk **(1./len(shape))
                dim_len = int(dim_len)
                chunk = tuple([dim_len for item in shape])
        return chunk

    def _check_invalid_data(self, data):

        finite_data = da.isfinite(data)
        not_filled_data = data != self._fill_value
        valid_data = da.logical_and(finite_data, not_filled_data)

        if self._leading_time:
            time_len = data.shape[0]
            valid_data = valid_data.sum(axis=0) == time_len

        valid_data = valid_data.compute()
        masked = True

        if np.all(valid_data):
            valid_data = None
            masked = False

        return masked, valid_data

    def _compress_masked_data(self, data, mask, out_arr):
        if self._leading_time:
            compress_axis = 1
        else:
            compress_axis = None

        compressed_data = da.compress(mask, data, axis=compress_axis)
        da.store(compressed_data, out_arr)
        return out_arr

    def calc_running_mean(self, window_size, year_len, save=True):

        if self._leading_time:
            orig = self._chunk_shape
            new_chunk = tuple([window_size*50] + list(orig[1:]))
            self.data.rechunk(new_chunk)

        res = super(Hdf5DataObject, self).calc_running_mean(window_size,
                                                            year_len,
                                                            save=save)

        if self._leading_time:
            res = res.rechunk(orig)
        
        return res

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

    @classmethod
    def from_netcdf(cls, filename, var_name, h5file):

        with ncf.Dataset(filename, 'r') as f:
            # TODO: Figure out why multiprocessing dask arrays fail on netcdf
            data = f.variables[var_name][:]
            coords = {BaseDataObject.LAT: f.variables['lat'][:],
                      BaseDataObject.LON: f.variables['lon'][:]}
            times = f.variables['time']
            coords[BaseDataObject.TIME] = ncf.num2date(times[:], times.units)

            for i, key in enumerate(f.dimensions.iterkeys()):
                if key in coords.keys():
                    coords[key] = (i, coords[key])

            return cls(data, h5file, dim_coords=coords, force_flat=True)

    @classmethod
    def from_hdf5(cls, filename, var_name, h5file, data_dir='/'):

        with tb.open_file(filename, 'r') as f:

            data = f.get_node(data_dir, name=var_name)
            try:
                fill_val = data.attrs.fill_value
            except AttributeError:
                fill_val = None

            lat = f.get_node(data_dir+'lat')
            lon = f.get_node(data_dir+'lon')
            coords = {BaseDataObject.LAT: (lat.attrs.index, lat[:]),
                      BaseDataObject.LON: (lon.attrs.index, lon[:])}
            times = f.get_node(data_dir+'time')
            coords[BaseDataObject.TIME] = (times.attrs.index,
                                           ncf.num2date(times[:],
                                                        times.attrs.units))
            return cls(data, h5file, dim_coords=coords, force_flat=True,
                       fill_value=fill_val)


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


def netcdf_to_data_obj(filename, var_name, h5file=None, force_flat=True):

    f = ncf.Dataset(filename, 'r')

    try:
        data = f.variables[var_name]
        coords = {BaseDataObject.LAT: f.variables['lat'][:],
                  BaseDataObject.LON: f.variables['lon'][:]}
        times = f.variables['time']
        coords[BaseDataObject.TIME] = ncf.num2date(times[:], times.units)

        for i, key in enumerate(data.dimensions):
            if key in coords.keys():
                coords[key] = (i, coords[key])

        if h5file is not None:
            return Hdf5DataObject(data[:], h5file, dim_coords=coords,
                                  force_flat=force_flat, time_units=times.units)

        else:
            return BaseDataObject(data[:], dim_coords=coords,
                                  force_flat=force_flat, time_units=times.units)

    finally:
        f.close()


def posterior_ncf_to_data_obj(filename, var_name, h5file=None):

    f = ncf.Dataset(filename, 'r')

    try:
        data = f.variables[var_name][:]
        coords = {BaseDataObject.LAT: f.variables['lat'][:],
                  BaseDataObject.LON: f.variables['lon'][:]}
        times = (0, f.variables['time'][:])

        coords['time'] = times
        coords['lat'] = (1, coords['lat'])
        coords['lon'] = (1, coords['lon'])

        if h5file is not None:
            return Hdf5DataObject(data, h5file, dim_coords=coords,
                                  force_flat=True,
                                  is_run_mean=True)

        else:
            return BaseDataObject(data, dim_coords=coords, force_flat=True,
                                  is_run_mean=True)

    finally:
        f.close()


def posterior_npz_to_data_obj(filename):
    f = np.load(filename)

    data = f['values'][:]
    lat = f['lat'][:, 0]
    lon = f['lon'][0, :]
    coords = {BaseDataObject.LAT: (1, lat),
              BaseDataObject.LON: (1, lon),
              BaseDataObject.TIME: (0, f['years'])}

    return BaseDataObject(data, dim_coords=coords, force_flat=True,
                          is_run_mean=True, is_anomaly=True)


def netcdf_to_hdf5_container(infile, var_name, outfile, data_dir='/'):
    f = ncf.Dataset(infile, 'r')
    outf = tb.open_file(outfile, 'w', filters=tb.Filters(complib='blosc',
                                                         complevel=5))

    try:
        data = f.variables[var_name]
        atom = tb.Atom.from_dtype(data.datatype)
        shape = data.shape
        out = empty_hdf5_carray(outf, data_dir, var_name, atom, shape)

        spatial_nbytes = np.product(data.shape[1:])*data.dtype.itemsize
        tchunk_60mb = 60*1024**2 // spatial_nbytes
        fill_value = 1.0e20

        masked = False
        for k in xrange(0, shape[0], tchunk_60mb):
            if k == 0:
                data_chunk = data[k:k+tchunk_60mb]
                masked = np.ma.is_masked(data_chunk)
                if masked:
                    out.attrs.masked = True
                    out.attrs.fill_value = fill_value
            elif masked:
                data_chunk = data[k:k+tchunk_60mb].filled(1.0e20)
            else:
                data_chunk = data[k:k+tchunk_60mb]

            out[k:k+tchunk_60mb] = data_chunk

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
        if data.attrs.masked:
            mask = data == data.attrs.fill_value
            data = np.ma.array(data,
                               mask=mask,
                               fill_value=data.attrs.fill_value)

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
