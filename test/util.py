import numpy as np
import tensorflow as tf
import micronet

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
           < results[micronet.estimator.TOP_1_ACCURACY_KEY] < \
           random_chance/pre_train_bound_factor

    # 2. Check that the model can be trained.
    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

    # 3. Check that the model accuracy has increased.
    results = estimator.evaluate(eval_input_fn, steps=eval_steps)
    post_train_bound_factor = 0.8
    assert expected_post_train_accuracy*post_train_bound_factor \
           < results[micronet.estimator.TOP_1_ACCURACY_KEY] < \
           expected_post_train_accuracy/post_train_bound_factor


def example_keras_fn(num_classes):

    # This input function is an edited version of a the model function from:
    # https://github.com/tensorflow/tpu/blob/master/models/experimental/cifar_keras/cifar_keras.py
    def keras_model_fn():
        """Define a CIFAR model in Keras."""
        layers = tf.keras.layers
        # Pass our input tensor to initialize the Keras input layer.
        # Edited:
        # v = layers.Input(tensor=input_features)
        input_layer = layers.Input(shape=(32, 32, 3))
        first_layer = layers.Conv2D(filters=32, kernel_size=5,
                          activation="relu", padding="same")(input_layer)
                          # input_shape=(32, 32, 3))#(input_layer)
        v = layers.MaxPool2D(pool_size=2, name='maxPool1')(first_layer)
        v = layers.Conv2D(filters=64, kernel_size=5,
                          activation="relu", padding="same")(v)
        v = layers.MaxPool2D(pool_size=2, name='maxPool2')(v)
        v = layers.Flatten()(v)
        fc1 = layers.Dense(units=512, activation="relu")(v)
        # Edited:
        # logits = layers.Dense(units=10)(fc1)
        logits = layers.Dense(units=num_classes)(fc1)
        # Edited:
        # return logits
        model = tf.keras.Model(input_layer, logits)
        return model

    return keras_model_fn


def example_model_fn(num_classes):

    def model_fn(features, labels, mode, params):
        del params
        image = features

        def metric_fn(labels, logits):
            accuracy = tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(logits, axis=1))
            return {"top_1_accuracy": accuracy}

        model = example_keras_fn(10)()
        if mode == tf.estimator.ModeKeys.PREDICT:
            outputs = model(image, training=False)
            predictions = {
                'class_ids': tf.argmax(outputs, axis=1),
                'probabilities': tf.nn.softmax(outputs)
            }
            return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)
        elif mode == tf.estimator.ModeKeys.EVAL:
            outputs = model(image, training=False)
            loss = tf.losses.sparse_softmax_cross_entropy(logits=outputs,
                                                          labels=labels)
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss,
                eval_metrics=(metric_fn, [labels, outputs]))
        elif mode == tf.estimator.ModeKeys.TRAIN:
            outputs = model(image, training=True)
            loss = tf.losses.sparse_softmax_cross_entropy(logits=outputs,
                                                          labels=labels)
            learning_rate = tf.train.exponential_decay(
                0.05,
                tf.train.get_global_step(),
                decay_steps=100000,
                decay_rate=0.96)
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
            train_op = optimizer.minimize(loss, tf.train.get_global_step())
            estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss,
                                                        train_op=train_op)
            return estimator_spec

    return model_fn


    # FIXME 5: finish.
    #
    # def _count_bytes(tensor):
    #     params = _count_params(tensor)
    #     size = params *
    #
    # def count_trainable_param_bytes(keras_model):
    #    trainable_weights = get_trainable_weights(keras_model)
#    count = int(np.sum([_count_params(p) for p in set(trainable_weights)]))
