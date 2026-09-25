import numpy as np


def round_to_int_32(data):
    """
    Takes a Numpy array of float values between -1 and 1,
    and rounds them to significant 32-bit integer values,
    to be used in the morton code computation.
    """
    min_data = np.abs(np.min(data) - 0.5)
    data = 256 * (data + min_data)
    data = np.round(2 ** 21 - data).astype(dtype=np.int32)
    return data


def split_by_3(x):
    """
    Separate bits of a 32-bit integer by 3 positions apart.
    """
    x &= 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def get_z_order(x, y, z):
    """
    Given 3 arrays of corresponding x, y, z coordinates,
    compute the morton (or z) code for each point.
    """
    res = 0
    res |= split_by_3(x) | split_by_3(y) << 1 | split_by_3(z) << 2
    return res


def get_z_values(data):
    """
    Computes the z values for a point array.
    :param data: Nx3 array of x, y, and z location
    :return: Nx1 array of z values
    """
    points_round = round_to_int_32(data)
    z = get_z_order(points_round[:, 0], points_round[:, 1], points_round[:, 2])
    return z
