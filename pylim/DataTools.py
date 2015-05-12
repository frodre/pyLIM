"""
Module for data interaction tools for the LIM package.

Author: Andre Perkins
"""

import tables as tb
import numpy as np


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

    @staticmethod
    def _match_dims(shape, dim_coords):
        return {key: value[0] for key, value in dim_coords.items()
                if shape[value[0]] == len(value[1])}

    @staticmethod
    def _new_databin(data):
        new = np.empty_like(data)
        new[:] = data
        return new

    def __init__(self, data, dim_coords=None, valid_data=None, force_flat=False,
                 is_anomaly=False, is_run_mean=False, is_detrended=False):
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
        mask: ndarray (np.bool), optional
            Masked array corresponding to input dataset or uncompressed
            version of the input dataset.  Should have spatial dimensions
            greater than or equal to the spatial dimensions of data.
        """

        assert data.ndim <= 4, 'Maximum of 4 dimensions are allowed.'
        self._full_shp = data.shape

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

        self.orig_data = self._new_databin(data)
        self.data = self.orig_data

        # Flatten Spatial Dimension if applicable
        if force_flat or self.is_masked:
            if self._leading_time:
                self.data = data.reshape(self._time_shp + self._flat_spatial_shp)
            else:
                self.data = data.reshape(self._flat_spatial_shp)

        # Compress the data if mask is present
        if compressed or self.is_masked:
            if compressed:
                self.compressed_data = self.orig_data
            elif self.is_masked:
                if self._leading_time:
                    self.compressed_data = self._new_databin(self.data[:, self.valid_data])
                else:
                    self.compressed_data = self._new_databin(self.data[self.valid_data])
            self.data = self.compressed_data
        else:
            self.compressed_data = None

        # Future possible data manipulation functionality
        self.is_anomaly = is_anomaly
        self.is_run_mean = is_run_mean
        self.is_detrended = is_detrended

    def inflate_full_grid(self, data=None, reshape_orig=False):
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