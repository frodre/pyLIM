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

    # Trying out use of static attributes
    _TIME = 'time'
    _LEVEL = 'level'
    _LAT = 'lat'
    _LON = 'lon'

    @property
    def time(self):
        return type(self)._TIME

    @property
    def level(self):
        return type(self)._LEVEL

    @property
    def lat(self):
        return type(self)._LAT

    @property
    def lon(self):
        return type(self)._LON

    @staticmethod
    def _match_dims(shape, dim_coords):
        return {key: shape.index(len(value)) for key, value in dim_coords.items()}


    def __init__(self, data, dim_coords=None, mask=None):
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
        if self._TIME in dim_coords:
            assert data.shape[0] == len(dim_coords[self._TIME]), \
                'Temporal dimension must be the leading dimension.'
            self._leading_time = True
            self._spatial_shp = data.shape[1:]
        else:
            self._leading_time = False
            self._spatial_shp = self._full_shp

        try:
            self._dim_idx = self._match_dims(data, dim_coords)
        except ValueError as e:
            # Could not fit spatial dimensions, no regridding
            self._dim_idx = None


        # Check to see if data input is a compressed version
        compressed = False
        if mask is not None:
            dim_lim = mask.ndim

            assert dim_lim <= 3,\
                'valid_input should not have more than 2 dimensions.'

            # Check the dimensions of the mask and data to se if compressed
            for dat_dim, mask_dim in zip(data.shape[::-1][:dim_lim],
                                         mask.shape[::-1][:dim_lim]):
                assert dat_dim <= mask_dim,\
                    'Valid data array provided should have larger ' +\
                    'spatial dimension than the masked input data.'
                if dat_dim < mask_dim:
                    compressed |= True

            # Apply input mask if its spatial dimensions match data
            if not compressed:
                full_valid = np.ones(data.shape, dtype=np.bool) * mask
                data[~full_valid] = np.nan
        else:
            compressed = False

        # Create full grid from compressed data if compressed
        if compressed:
            self.data = data
            full_shp = list(data.shape[:(data.ndim-dim_lim)]) + list(mask.shape)
            full_valid = np.ones(full_shp, dtype=np.bool) * mask
            self.full_data = np.empty(full_shp)*np.nan
            self.full_data[full_valid] = self.data
            self.full_shp = self.full_data.shape
            self.have_data = mask
            self.is_masked = True
        else:
            self.full_data = data
            self.full_shp = data.shape

            # Flatten spatial dimension
            if data.ndim == 3:
                new_shp = (self.full_shp[0], self.full_shp[1]*self.full_shp[2])
                self.full_data = self.full_data.reshape(new_shp)

            # Create mask if data contains non-finite elements (nans, infs)
            if not np.alltrue(np.isfinite(data)):
                self.is_masked = True
                self.have_data = np.isfinite(self.full_data[0])

                #Find locations we have data for at all times
                for time in self.full_data:
                    self.have_data &= np.isfinite(time)

                self.data = self.full_data[:, self.have_data]
            else:
                self.is_masked = False
                self.data = self.full_data

        # Future possible data manipulation functionality
        self.is_anomaly = False
        self.is_runmean = False
        self.is_detrended = False


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