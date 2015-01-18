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
                                       createparents=True,
                                       **kwargs)
        out_arr[:] = data
    except tb.NodeError:
        node_path = '/'.join(group._v_pathname, node)
        h5file.remove_node(node_path)
        out_arr = h5file.create_carray(group,
                                       node,
                                       atom=tb.Atom.from_dtype(data.dtype),
                                       shape=data.shape,
                                       createparents=True,
                                       **kwargs)
        out_arr[:] = data

    return out_arr


def empty_hdf5_carray(h5file, group, node, atom, shape, **kwargs):
    assert(type(h5file) == tb.File)

    try:
        out_arr = h5file.create_carray(group,
                                       node,
                                       atom=atom,
                                       shape=shape,
                                       createparents=True,
                                       **kwargs)

    except tb.NodeError:
        node_path = '/'.join(group._v_pathname, node)
        h5file.remove_node(node_path)
        out_arr = h5file.create_carray(group,
                                       node,
                                       atom=atom,
                                       shape=shape,
                                       createparents=True,
                                       **kwargs)
    return out_arr