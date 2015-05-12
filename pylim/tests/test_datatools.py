from pylim.DataTools import BaseDataObject

__author__ = 'wperkins'

import tables as tb
import numpy as np
import pytest
import os
from pylim import DataTools as Dt
from pylim.DataTools import BaseDataObject as BDO


@pytest.fixture(scope='module')
def tb_file(request):
    tbf = tb.open_file('test.h5', 'w',
                       filters=tb.Filters(complevel=0, complib='blosc'))

    def fin():
        tbf.close()
        os.system('rm -f test.h5')

    request.addfinalizer(fin)
    return tbf


def test_empty_carray_node(tb_file):
    shape = (5, 10)
    Dt.empty_hdf5_carray(tb_file, '/', 'empty',
                         tb.Atom.from_dtype(np.dtype(np.float64)),
                         shape)
    assert tb_file.__contains__('/empty')
    assert tb_file.root.empty.shape == shape


def test_empty_carray_node_createparents(tb_file):
    shape = (5, 10)
    Dt.empty_hdf5_carray(tb_file, '/path/to/node', 'empty',
                         tb.Atom.from_dtype(np.dtype(np.float64)),
                         shape,
                         createparents=True)
    assert tb_file.__contains__('/path/to/node/empty')
    assert tb_file.get_node('/path/to/node/empty').shape == shape


def test_empty_carray_node_already_exists(tb_file):
    shape = (5, 10)
    Dt.empty_hdf5_carray(tb_file, '/', 'empty',
                         tb.Atom.from_dtype(np.dtype(np.float64)),
                         shape)
    Dt.empty_hdf5_carray(tb_file, '/', 'empty',
                         tb.Atom.from_dtype(np.dtype(np.float64)),
                         shape)
    assert tb_file.__contains__('/empty')


def test_var_to_carray_node(tb_file):
    shape = (5, 10)
    data = np.arange(50).reshape(shape)
    Dt.var_to_hdf5_carray(tb_file, '/', 'data', data)
    assert tb_file.__contains__('/data')
    assert tb_file.root.data.shape == shape
    assert np.array_equal(data, tb_file.root.data[:])


def test_var_to_carray_node_createparents(tb_file):
    shape = (5, 10)
    data = np.arange(50).reshape(shape)
    Dt.var_to_hdf5_carray(tb_file, '/path/to/node', 'data', data,
                          createparents=True)
    assert tb_file.__contains__('/path/to/node/data')
    assert tb_file.get_node('/path/to/node/data').shape == shape
    assert np.array_equal(data, tb_file.root.path.to.node.data[:])


def test_var_to_carray_node_already_exists(tb_file):
    shape = (5, 10)
    data1 = np.arange(50).reshape(shape)
    data2 = np.arange(50, 100).reshape(shape)

    Dt.var_to_hdf5_carray(tb_file, '/', 'data', data1)
    Dt.var_to_hdf5_carray(tb_file, '/', 'data', data2)
    assert tb_file.__contains__('/data')
    assert np.array_equal(tb_file.root.data[:], data2)


#### DataObject Tests ####

## When no dimensions are provided 1D-3D data could have spatial and temporal
## components.  Should do not change shape or data.  4D data we do not know
## the time dimension so it should also be left alone by default.
@pytest.mark.parametrize("shape, data_obj",
    [(24, BDO),
    ((4, 6), BDO),
    ((4, 3, 2), BDO),
    ((2, 2, 3, 2),BDO),
    pytest.mark.xfail(((1, 2, 2, 3, 2), Dt.BaseDataObject)),
    ])
def test_basedataobj_data_nodim(shape, data_obj):
    data = np.arange(np.product(np.array(shape))).reshape(shape)
    obj = data_obj(data)
    assert np.array_equal(obj.data, data)
    assert obj.data.shape == data.shape
    assert obj._full_shp == data.shape


def test_basedataobj_data_force_flat():
    data = np.arange(20).reshape(2, 2, 5)
    obj = BDO(data, force_flat=True)
    assert np.array_equal(data.flatten(), obj.data[:])
    assert obj.data.shape == data.flatten().shape


def test_baseddataobj_dim_matching():
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {BDO.TIME: (0, [1, 2]),
              BDO.LEVEL: (1, [1000, 900]),
              BDO.LAT: (2, [45, 50, 55]),
              BDO.LON: (3, [-80, -90])}
    obj = BDO(data, dim_coords=coords)
    assert obj._leading_time
    assert obj._dim_idx == {key: value[0] for key, value in coords.items()}
    assert data.shape[0] == obj._time_shp
    assert data.shape[1:] == obj._spatial_shp
    assert data.shape == obj.data.shape

@pytest.mark.xfail
def test_baseddataobj_dim_noleadsample():
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {BDO.TIME: (2, [1, 2, 3]),
              BDO.LEVEL: (1, [1000, 900]),
              BDO.LAT: (0, [50, 55]),
              BDO.LON: (3, [-80, -90])}
    obj = BDO(data, dim_coords=coords)


def test_baseddataobj_dim_mismatched():
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {BDO.TIME: (0, [1, 2]),
              BDO.LEVEL: (1, [1000, 900, 800]),
              BDO.LAT: (2, [45, 50, 55]),
              BDO.LON: (3, [-80, -90, 100])}
    obj = BDO(data, dim_coords=coords)
    assert len(obj._dim_idx.keys()) == 2
    assert BDO.TIME in obj._dim_idx
    assert BDO.LAT in obj._dim_idx


def test_baseddataobj_dim_notime():
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {BDO.LEVEL: (1, [1000, 900]),
              BDO.LAT: (2, [45, 50, 55]),
              BDO.LON: (3, [-80, -90])}
    obj = BDO(data, dim_coords=coords)
    assert len(obj._dim_idx.keys()) == 3
    assert obj._full_shp == data.shape


if __name__ == '__main__':
    # try:
    #     f = tb.open_file('test.h5', 'w')
    #     test_var_to_carray_node_already_exists(f)
    # finally:
    #     f.close()
    #     os.system('rm -f test.h5')
    test_baseddataobj_dim_matching()



