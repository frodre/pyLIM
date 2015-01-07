__author__ = 'wperkins'

import scipy.io.netcdf as ncf
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
    except tb.NodeError:
        pass