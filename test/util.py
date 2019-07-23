import numpy as np

"""
Everything here is imported into the __init__.py. It's sufficient to import
from micronet.test.

Note: is this a useful/correct practice?
"""
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


# FIXME 17: add checks for global_step/sec.
def check_train_and_eval(estimator, train_input_fn, eval_input_fn,
                         train_steps: int, eval_steps: int, num_classes: int,
                         expected_post_train_accuracy: float):
    """Check the eval and train behaviour of an estimator.

    Checks that:
        1. The untrained estimator can evaluate (with random performance).
        2. The estimator can be trained.
        3. After training, the estimator has the specified accuracy.
    """
    # 1. Evaluate using the untrained estimator.
    results = estimator.evaluate(eval_input_fn, steps=eval_steps)
    # TODO: make a reusable CDF_inverse function to easily calculate expected
    # random results.
    random_chance = 1/num_classes
    pre_train_bound_factor = 0.5
    assert random_chance*pre_train_bound_factor \
           < results['accuracy'] < \
           random_chance/pre_train_bound_factor

    # 2. Check that the model can be trained.
    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

    # 3. Check that the model accuracy has increased.
    results = estimator.evaluate(eval_input_fn, steps=eval_steps)
    post_train_bound_factor = 0.8
    assert expected_post_train_accuracy*post_train_bound_factor \
           < results['accuracy'] < \
           expected_post_train_accuracy/post_train_bound_factor


# FIXME 5: finish.
#
# def _count_bytes(tensor):
#     params = _count_params(tensor)
#     size = params *
#
# def count_trainable_param_bytes(keras_model):
#    trainable_weights = get_trainable_weights(keras_model)
#    count = int(np.sum([_count_params(p) for p in set(trainable_weights)]))
