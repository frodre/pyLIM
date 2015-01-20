__author__ = 'wperkins'

import scipy.io.netcdf as ncf #should use netcdf library
import tables as tb


def unpack_netcdf_data(ncvar):
    assert(type(ncvar) == ncf.netcdf_variable)

    try:
        data = ncvar.data*ncvar.scale_factor + ncvar.add_offset
    except AttributeError:
        data = ncvar.data

    return data


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


def empty_hdf5_carray(h5file, group, node, in_atom, shape, **kwargs):
    assert(type(h5file) == tb.File)

    # kwargs['atom'] = in_atom
    # kwargs['shape'] = shape

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