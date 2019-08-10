import tensorflow as tf
"""
Methods useful when building models.

This file is included into the micronet package in __init__.py.

"""


def normalize_image(image_tensor, mean_rgb, stddev_rgb,
                    data_format='channels_last'):
    """Normalize the image given the means and stddevs."""
    # TODO: support GPU by using shape=[3, 1, 1]
    if data_format == 'channels_first':
        shape = [3, 1, 1]
    elif data_format == 'channels_last':
        shape = [1, 1, 3]
    else:
        raise Exception('Unexpected data_format: {}'.format(data_format))
    image_tensor -= tf.constant(mean_rgb, shape=shape,
                            dtype=image_tensor.dtype)
    image_tensor /= tf.constant(stddev_rgb, shape=shape,
                            dtype=image_tensor.dtype)
    return image_tensor
