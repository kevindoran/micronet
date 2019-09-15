import numpy as np
import tensorflow as tf
import micronet.dataset.imagenet as imagenet_ds
import micronet.experiments.sequential_pool.efficientnet_builder as efnet_builder
import micronet.experiments.sequential_pool.efficientnet_model as efnet_model

MASK_WIDTH = 7
MASK_HEIGHT = 7
NUM_PIXELS = MASK_WIDTH * MASK_HEIGHT
NUM_ACTIONS = NUM_PIXELS + 1
STOP_ACTION = NUM_ACTIONS - 1
IMAGE_SHAPE = (224, 224, 3)
ENCODED_STATE_SHAPE = [7] # 4 for MaskEncoding, 3 for PoolEncoding.


class MaskEncoding:

    buckets_flat = np.array([
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 4, 4, 5, 1, 1],
        [0, 0, 4, 5, 5, 1, 1],
        [2, 2, 6, 7, 7, 3, 3],
        [2, 2, 2, 3, 3, 3, 3],
        [2, 2, 2, 3, 3, 3, 3],
    ])

    buckets = np.array([
        [
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ], [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ], [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
    ])
    right_idx = 0
    bottom_idx = 1
    center_idx = 2

    max_right = 3*7 + 3
    max_bottom = 3*7
    max_center = 3*3
    max_count = 7*7

    def __init__(self, right_count, bottom_count, center_count, total):
        self.bottom_count = bottom_count
        self.right_count = right_count
        self.center_count = center_count
        self.total = total

    def __hash__(self):
        return hash((self.right_count, self.bottom_count, self.center_count,
                    self.total))

    def __eq__(self, other):
        return (self.right_count, self.bottom_count,
                self.center_count, self.total) == \
               (other.right_count, other.bottom_count,
                other.center_count, other.total)

    @classmethod
    def from_mask(cls, mask):
        """Encode a 7x7 mask into 4 integers.

        params:
            mask: 2D array, 7x7 mask for the global pool layer outputs.
        returns:


        The integers are:

            * bottom count
            * right count
            * center count
            * total

        Top, left and outer counts can be inferred by subtracting
        top, left and center from the total.

        This encoding was chosen so that:
          * it would be easy to find all states for a given total mask
            count.
          * coarse coding is used instead of state aggregation. I have seen
            claims that coarse coding is typically more effective than
            state aggregation. This, however, does warrant some investigation.

        Only 5-bits of each integer are used.

        Coarse code
        -----------

             0     1     2     3     4     5     6
          +-----+-----+-----+-----+-----+-----+-----+
        0 | 000 | 000 | 000 | 000 | 001 | 001 | 001 |
          +-----+-----+-----+-----+-----+-----+-----+
        1 | 000 | 000 | 000 | 000 | 001 | 001 | 001 |
          +-----+-----+-----+-----+-----+-----+-----+
        2 | 000 | 000 | 100 | 100 | 101 | 001 | 001 |
          +-----+-----+-----+-----+-----+-----+-----+
        3 | 000 | 000 | 100 | 111 | 101 | 001 | 001 |
          +-----+-----+-----+-----+-----+-----+-----+
        4 | 010 | 010 | 110 | 111 | 111 | 011 | 011 |
          +-----+-----+-----+-----+-----+-----+-----+
        5 | 010 | 010 | 010 | 011 | 011 | 011 | 011 |
          +-----+-----+-----+-----+-----+-----+-----+
        6 | 010 | 010 | 010 | 011 | 011 | 011 | 011 |
          +-----+-----+-----+-----+-----+-----+-----+

        As decimal
        ----------
        Note, this is not a state aggregation, but the decimal
        representation of the coarse code.

             0     1     2     3     4     5     6
          +-----+-----+-----+-----+-----+-----+-----+
        0 |  0  |  0  |  0  |  0  |  1  |  1  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+
        1 |  0  |  0  |  0  |  0  |  1  |  1  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+
        2 |  0  |  0  |  4  |  4  |  5  |  1  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+
        3 |  0  |  0  |  4  |  5  |  5  |  1  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+
        4 |  2  |  2  |  6  |  7  |  7  |  3  |  3  |
          +-----+-----+-----+-----+-----+-----+-----+
        5 |  2  |  2  |  2  |  3  |  3  |  3  |  3  |
          +-----+-----+-----+-----+-----+-----+-----+
        6 |  2  |  2  |  2  |  3  |  3  |  3  |  3  |
          +-----+-----+-----+-----+-----+-----+-----+
        """
        mask = np.array(mask)

        right_count = np.sum(np.multiply(np.bitwise_and(
            cls.buckets_flat, b'001'), mask))
        bottom_count = np.sum(np.multiply(np.right_shift(
            np.bitwise_and(cls.buckets_flat, b'0101'), 1), mask))
        center_count = np.sum(np.multiply(np.right_shift(
            np.bitwise_and(cls.buckets_flat, b'0101'), 1), mask))
        total = np.sum(mask)
        assert total <= NUM_PIXELS
        return MaskEncoding(right_count, bottom_count, center_count, total)

    @classmethod
    def encode_net(cls, mask):
        # square_mask = tf.reshape(mask, (-1, 7, 7))
        mask = tf.to_float(mask)
        flattened_buckets = np.reshape(cls.buckets, (3, NUM_PIXELS))
        right_count = tf.math.reduce_sum(
            mask * flattened_buckets[cls.right_idx], axis=[1])
        bottom_count = tf.math.reduce_sum(
            mask * flattened_buckets[cls.bottom_idx], axis=[1])
        center_count = tf.math.reduce_sum(
            mask * flattened_buckets[cls.center_idx], axis=[1])
        total = tf.math.reduce_sum(mask, axis=[1])
        return tf.stack([right_count, bottom_count, center_count, total],
                         axis=1)


class PoolEncoding:

    @staticmethod
    def encode_net(pool_output):
        """

        :param pool_output: bx1280 tensor.
        :return: 3xb tensor (average, total, SD).
        """
        abs_pool = tf.abs(pool_output)
        # axis 0 is the batch axis.
        sum = tf.math.reduce_sum(abs_pool, axis=[1])
        mean, variance = tf.nn.moments(abs_pool, axes=[1])
        sd = tf.sqrt(variance)
        # We use stack, as we wish there to be an extra dimension:
        # previously, all were (batch,), after we want (batch, 3)
        state = tf.stack((sum, mean, sd), axis=1)
        return state


def image_iterator(batch_size=1):
    image_fn = imagenet_ds.create_train_input(IMAGE_SHAPE[0]).input_fn
    params = {'batch_size': batch_size}
    dataset = image_fn(params)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def efficientnet_until_pool(image):
    def normalize_features(img, mean_rgb, stddev_rgb):
        """Normalize the image given the means and stddevs."""
        # TODO: support GPU by using shape=[3, 1, 1]
        img -= tf.constant(mean_rgb, shape=[1, 1, 3],
                           dtype=img.dtype)
        img /= tf.constant(stddev_rgb, shape=[1, 1, 3],
                           dtype=img.dtype)
        return img
    normalized_image = normalize_features(image,
                                          efnet_builder.MEAN_RGB,
                                          efnet_builder.STDDEV_RGB)
    blocks_args, global_params = efnet_builder.get_model_params(
        'efficientnet-b0', override_params=None)
    mask = None
    model = efnet_model.Model(mask, blocks_args, global_params)
    with tf.variable_scope('efficientnet-b0'):
        pool_inputs = model(normalized_image, training=False,
                            features_only=True)
    return pool_inputs
