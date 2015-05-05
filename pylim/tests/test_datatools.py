__author__ = 'wperkins'

import tables as tb
import numpy as np
import pytest
import os
from pylim import DataTools as Dt


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


if __name__ == '__main__':
    try:
        f = tb.open_file('test.h5', 'w')
        test_var_to_carray_node_already_exists(f)
    finally:
        f.close()
        os.system('rm -f test.h5')



