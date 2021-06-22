import torch
import numpy as np


def get_windows(arr, window_size=200, step_size=50):
    """Transforms time series to a matrix of windows

    Parameters
    ----------
    arr : array-like of shape (n, )
        Array-like object.
    window_size : int, optional (default=200)
        Size of int in term of measurements.
    step_size : int, optional (default=50)
        Size of shift of the window.

    Returns
    -------
    np.ndarray of shape (n // step, window_size)
        Slices of the passed `arr`.

    """
    indexes = (np.arange(0, arr.shape[0], step_size).astype(int)[:, None]
               + np.arange(0, window_size, 1).astype(int).reshape(1, -1))
    cutoff = indexes[:, -1] < arr.shape[0]
    ids = indexes[cutoff]
    if type(arr) == torch.Tensor:
        ids = torch.LongTensor(ids)
    windows = arr[ids]

    return windows


def interpolate(x, xp, fp):
    if fp.ndim == 1:
        interp_x = np.interp(x, xp, fp)
    else:
        axis = fp.shape[1]
        len_x = x.shape[0]
        size = (len_x, axis)
        interp_x = np.zeros(size)
        for i in range(axis):
            signal = fp[:, i]
            interp_x[:, i] = np.interp(x, xp, signal)
    return interp_x

def train_test_valid_split(exps_ids, valid_size=0.2, test_size=0, random_state=23):
    """Gets train, test, and valid separation."""

    if random_state is not None:
        np.random.seed(random_state)
        np.random.shuffle(exps_ids)
    sep1 = int(len(exps_ids) * (1 - valid_size - test_size))
    sep2 = int(len(exps_ids) * (1 - test_size))
    train_exps = exps_ids[:sep1]
    valid_exps = exps_ids[sep1:sep2]
    test_exps = exps_ids[sep2:]
    split = {"train": train_exps,
             "valid": valid_exps,
             "test": test_exps}

    return split