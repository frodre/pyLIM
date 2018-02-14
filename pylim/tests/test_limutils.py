import pytest
import numpy as np
import pylim.LIMUtils as limutil



@pytest.fixture()
def data_key_list(request):
    keys = ['one', 'two', 'three']
    return keys


@pytest.fixture()
def data_arr_list(request):
    data = [np.ones((4, i))*i for i in range(1, 4)]
    return data


def test_state_vector_wrong_length_args(data_key_list, data_arr_list):
    keys = data_key_list[:-1]

    with pytest.raises(ValueError):
        limutil.create_state_vector(keys, data_arr_list)


def test_state_vector_length(data_key_list, data_arr_list):

    pos, state = limutil.create_state_vector(data_key_list, data_arr_list)

    concat_dim_len = 0
    for arr in data_arr_list:
        concat_dim_len += arr.shape[-1]

    assert len(data_key_list) == len(list(pos.keys()))
    assert state.shape[-1] == concat_dim_len


def test_state_vector_retrieve(data_key_list, data_arr_list):

    pos, state = limutil.create_state_vector(data_key_list, data_arr_list)

    for key, orig_data in zip(data_key_list, data_arr_list):

        start, end = pos[key]
        data = state[..., start:end]
        np.testing.assert_array_equal(data, orig_data)
