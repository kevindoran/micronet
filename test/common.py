import numpy as np

# TODO: why isn't there an exposed way of doing this?
#       I suppose the Backend is exposed. The name is just off-putting. I'll
#       leave the code here for now, as it is enlightening.

# Copied from:https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
def _int_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        return tuple(x.get_shape().as_list())
    except ValueError:
        return None


def _count_params(tensor):
    # np.prod is an element-wise product.
    return np.prod(_int_shape(tensor))


# Copied from layer_utils.py: https://github.com/keras-team/keras/blob/master/keras/utils/layer_utils.py
def get_trainable_weights(keras_model):
    # TODO: Why is this done? Why not just the 'else' version?
    if hasattr(keras_model, '_collected_trainable_weights'):
        trainable_weights = keras_model._collected_trainable_weights
    else:
        trainable_weights = keras_model.trainable_weights
    return trainable_weights


def count_trainable_params(keras_model):
    trainable_weights = get_trainable_weights(keras_model)
    count = int(np.sum([_count_params(p) for p in set(trainable_weights)]))
    return count

# FIXME 5: finish.
#
# def _count_bytes(tensor):
#     params = _count_params(tensor)
#     size = params *
#
# def count_trainable_param_bytes(keras_model):
#    trainable_weights = get_trainable_weights(keras_model)
#    count = int(np.sum([_count_params(p) for p in set(trainable_weights)]))
