"""
Helper for running LIM Experiments
"""

import numpy as np


def create_state_vector(datakeys, state_members):
    """
    Creates a state vector to be used by LIM object.
    
    Parameters
    ----------
    datakeys: list of str
        Keys for each member of the state vector
    state_members: list of array-like
        List of arrays to concatenate.  Assumes leading dimension is sampling
        and that order matches datakeys.

    Returns
    -------
    state_positions: dict of {str: tuple(int)}
        Relative position of each data array in the state vector along trailing
        axis
    state: array-like
        Concatenated array of all state members along trailing axis
    """

    if not len(datakeys) == len(state_members):
        raise ValueError('The length of datakeys and state_members must match')

    start = end = 0
    state_positions = {}
    for key, member in zip(datakeys, state_members):
        end += member.shape[-1]
        state_positions[key] = (start, end)
        start = end

    state = np.concatenate(state_members, axis=-1)

    return state_positions, state



