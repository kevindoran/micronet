import os
import tensorflow as tf
import tensorflow.keras.layers as layers

# From briefly looking at the code, it seems like weight decay is only applied
# to the standard 2D convolution layers (not the separable convolution layers).
# I don't know if this means it applied to the expand layers or not, or if it
# is applied to the pointwise conv part of the depthwise separable layer.
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py)
CONV2D_WEIGHT_DECAY = 0.00004

def _argmax(l):
    m = max(zip(l, range(len(l))))[1]
    return m


def _argmin(l):
    m = min(zip(l, range(len(l))))[1]
    return m


def _closet_positive_multiple(v, divisor, min_value=None):
    """Round ``v`` to a positive integer divisible by ``divisor`` with a caveat.

    The caveat being, round up if rounding down would result in more than a 10%
    drop in ``v``.

    This function was copied from the original repo (then comments added):
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def rough_num_trainable_params(input_shape, alpha, classes):
    # TODO
    return 0


def create_model(input_shape, alpha=1.0, weights='imagenet',
                 classes=1000):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # Determine data_format.
    # Assume the smallest dimension is for channels (e.g. RGB).
    channel_index = _argmin(input_shape)
    if channel_index == 0:
        data_format = 'channels_first'
    elif channel_index == 2:
        data_format = 'channels_last'
    else:
        raise Exception("The input shape has an unexpected shape.")
    # Set the data-format globally. TODO: is this a good idea?
    tf.keras.backend.set_image_data_format(data_format)

    # Create the layers.
    input_layer = tf.keras.Input(shape=input_shape)
    # Block 0.
    first_block_filters = _closet_positive_multiple(32 * alpha, divisor=8)
    x = layers.Conv2D(
        filters=first_block_filters,
        kernel_size=3,
        strides=2,
        padding='same', # What is the effect?
        # why? Because batch normalization negates the
        # effect of a bias. There is a bias used within the
        # BatchNormalization layer that is applied after the
        # normalization (optionally).
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(CONV2D_WEIGHT_DECAY),
        name='Conv1')(input_layer)
    # FIXME 10: is the channel axis needed to be set?
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name='bn_Conv1')(x)
    x = layers.ReLU(max_value=6)(x)

    # Block 1
    x = _first_inverted_res_block(x, num_filters=16, alpha=alpha,
                                  strides=1, block_id=1)

    # Block 2, 3
    x = _inverted_res_block(x, num_filters=24, alpha=alpha, expansion=6,
                            strides=2, block_id=2)
    x = _inverted_res_block(x, num_filters=24, alpha=alpha, expansion=6,
                            strides=1, block_id=3)

    # Block 4, 5, 6
    x = _inverted_res_block(x, num_filters=32, alpha=alpha, expansion=6,
                            strides=2, block_id=4)
    x = _inverted_res_block(x, num_filters=32, alpha=alpha, expansion=6,
                            strides=1, block_id=5)
    x = _inverted_res_block(x, num_filters=32, alpha=alpha, expansion=6,
                            strides=1, block_id=6)

    # Block 7, 8, 9, 10
    x = _inverted_res_block(x, num_filters=64, alpha=alpha, expansion=6,
                            strides=2, block_id=7)
    x = _inverted_res_block(x, num_filters=64, alpha=alpha, expansion=6,
                            strides=1, block_id=8)
    x = _inverted_res_block(x, num_filters=64, alpha=alpha, expansion=6,
                            strides=1, block_id=9)
    x = _inverted_res_block(x, num_filters=64, alpha=alpha, expansion=6,
                            strides=1, block_id=10)

    # Block 11, 12, 13
    x = _inverted_res_block(x, num_filters=96, alpha=alpha, expansion=6,
                            strides=1, block_id=11)
    x = _inverted_res_block(x, num_filters=96, alpha=alpha, expansion=6,
                            strides=1, block_id=12)
    x = _inverted_res_block(x, num_filters=96, alpha=alpha, expansion=6,
                            strides=1, block_id=13)

    # Block 14
    x = _inverted_res_block(x, num_filters=160, alpha=alpha, expansion=6,
                            strides=2, block_id=14)
    x = _inverted_res_block(x, num_filters=160, alpha=alpha, expansion=6,
                            strides=1, block_id=15)
    x = _inverted_res_block(x, num_filters=160, alpha=alpha, expansion=6,
                            strides=1, block_id=16)

    # Block 15
    x = _inverted_res_block(x, num_filters=320, alpha=alpha, expansion=6,
                            strides=1, block_id=17)

    # Block 16
    feature_count = 1280 * alpha if alpha > 1.0 else 1280
    x = layers.Conv2D(
        filters=feature_count, kernel_size=1, use_bias=False, activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(CONV2D_WEIGHT_DECAY),
        name='b_16_conv_1d')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name='b_16_bn')(x)
    x = layers.ReLU(max_value=6, name='b_16_relu')(x)

    # Pooling and feature layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, activation='softmax', use_bias=True,
                     name='Logits')(x)

    # Create the model.
    model = tf.keras.Model(input_layer, x,
                           name='mobilenetv2_alpha_{}'.format(alpha))
    return model


def _inverted_res_block(previous_layer, num_filters, alpha, expansion, strides,
                        block_id):
    # Sadly, name_scope doesn't seem to work with Keras layers that well.
    # Either use name_scope and don't set any manual names, or set all names
    # and manually make them unique.
    with tf.name_scope('b_{}_'.format(block_id)):
        # Expand
        expand_to = expansion * _num_channels(previous_layer)
        # Padding shouldn't be needed, as the kernel size is 1.
        x = layers.Conv2D(filters=expand_to, kernel_size=1,
                          use_bias=False, activation=None)(previous_layer)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999)(x)
        x = layers.ReLU(max_value=6)(x)

        # Depthwise
        x = layers.DepthwiseConv2D(kernel_size=3, strides=strides,
                                   activation=None, use_bias=False,
                                   padding='same')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = layers.ReLU(max_value=6)(x)

        # Project
        num_filters = int(num_filters * alpha)
        num_filters = _closet_positive_multiple(num_filters, 8)
        x = layers.Conv2D(filters=num_filters, kernel_size=1, use_bias=False,
                          activation=None)(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        return x


def _first_inverted_res_block(previous_layer, num_filters, alpha, strides,
                              block_id):
    # Depthwise
    # TODO: use name_scope.
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=strides, activation=None, use_bias=False,
        padding='same',
        name='b_{}_depthwise_conv'.format(block_id))(previous_layer)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999,
        name='b_{}_depthwise_conv_bn'.format(block_id))(x)
    x = layers.ReLU(
        max_value=6,
        name='b_{}_depthwise_conv_relu'.format(block_id))(x)

    # Project
    num_filters = int(num_filters * alpha)
    num_filters = _closet_positive_multiple(num_filters, 8)
    x = layers.Conv2D(
        num_filters, kernel_size=1, use_bias=False,
        activation=None, name='b_{}_project_conv'.format(block_id))(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999,
        name='b_{}_project_conv_bn'.format(block_id))(x)
    return x


# Surely there is a common utility that does this?
def _num_channels(layer):
    # TODO: is it okay having a global data format?
    format = tf.keras.backend.image_data_format()
    # TODO: Is this the right way?
    shape = layer.shape.as_list()
    if format == 'channels_first':
        count = shape[0] if shape[0] is not None else shape[1]
    elif format == 'channels_last':
        count = shape[-1]
    else:
        raise Exception('Unexpected data format: {}'.format(format))
    return count

