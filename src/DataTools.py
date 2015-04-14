import scipy.io.netcdf as ncf #should use netcdf library
import tables as tb
import numpy as np
from Stats import calc_anomaly


class DataObject(object):
    """Data Input Object

    This class is for handling data which may be in a masked format.
    """

    def __init__(self, data, valid_data=None):
        assert type(data) == np.ndarray
        assert (data.ndim <= 3) and (data.ndim >= 1),\
            'Expected ndim from 1 to 3'

        # Check to see if compressed array was given
        compressed = False
        if valid_data is not None:
            dim_lim = valid_data.ndim
            assert dim_lim < 3,\
                'valid_input should not have more than 2 dimensions.'
            for dat_dim, mask_dim in zip(data.shape[::-1][:dim_lim],
                                         valid_data.shape[::-1][:dim_lim]):
                assert dat_dim <= mask_dim,\
                    'Valid data array provided should have larger ' +\
                    'spatial dimension than the masked input data.'
                if dat_dim < mask_dim:
                    compressed |= True

            if not compressed:
                full_valid = np.ones(data.shape, dtype=np.bool) * valid_data
                data[~full_valid] = np.nan
        else:
            compressed = False

        if compressed:
            self.data = data
            full_shp = list(data.shape[:(data.ndim-dim_lim)]) + list(valid_data.shape)
            full_valid = np.ones(full_shp, dtype=np.bool) * valid_data
            self.full_data = np.empty(full_shp)*np.nan
            self.full_data[full_valid] = self.data
            self.full_shp = self.full_data.shape
            self.have_data = valid_data
            self.is_masked = True
        else:
            self.full_data = data
            self.full_shp = data.shape

            if data.ndim == 3:
                new_shp = (self.full_shp[0], self.full_shp[1]*self.full_shp[2])
                self.full_data = self.full_data.reshape(new_shp)

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

        self.is_anomaly = False
        self.is_runmean = False
        self.is_detrended = False


def var_to_hdf5_carray(h5file, group, node, data, **kwargs):
    assert(type(h5file) == tb.File)

    try:
        out_arr = h5file.create_carray(group,
                                       node,
                                       atom=tb.Atom.from_dtype(data.dtype),
                                       shape=data.shape,
                                       **kwargs)
        out_arr[:] = data
    except tb.NodeError:
        if type(group) == tb.Group:
            node_path = '/'.join((group._v_pathname, node))
        elif type(group) == str:
            node_path = '/'.join((group, node))
        else:
            raise TypeError('Expected group type of tables.Group or str.')

        h5file.remove_node(node_path)
        out_arr = h5file.create_carray(group,
                                       node,
                                       atom=tb.Atom.from_dtype(data.dtype),
                                       shape=data.shape,
                                       **kwargs)
        out_arr[:] = data

    return out_arr


def empty_hdf5_carray(h5file, group, node, in_atom, shape, kwargs):
    assert(type(h5file) == tb.File)

    try:
        out_arr = h5file.create_carray(group,
                                       node,
                                       atom=in_atom,
                                       shape=shape,
                                       **kwargs)

    except tb.NodeError:
        if type(group) == tb.Group:
            node_path = '/'.join((group._v_pathname, node))
        elif type(group) == str:
            node_path = '/'.join((group, node))
        else:
            raise TypeError('Expected group type of tables.Group or str.')

        h5file.remove_node(node_path)
        out_arr = h5file.create_carray(group,
                                       node,
                                       atom=in_atom,
                                       shape=shape,
                                       **kwargs)
    return out_arr