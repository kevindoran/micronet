import pytest
import micronet.dataset.cifar as cifar_ds
import tensorflow as tf
import numpy as np
import test.util
import functools


# This input function is an edited version of a the model function from:
# https://github.com/tensorflow/tpu/blob/master/models/experimental/cifar_keras/cifar_keras.py
# It has been edited as mentioned in the comments below. The tests are assuming
# that this model is functioning.
# Note: this function is also in test_estimator.py (with slightly different
# edits). If it is used in more places, consider factoring it out.
def _keras_model_fn():
    """Define a CIFAR model in Keras."""
    layers = tf.keras.layers
    # Pass our input tensor to initialize the Keras input layer.
    # Edited:
    # v = layers.Input(tensor=input_features)
    input_layer = layers.Input(shape=(32, 32, 3))
    v = layers.Conv2D(filters=32, kernel_size=5,
                      activation="relu", padding="same")(input_layer)
    v = layers.MaxPool2D(pool_size=2, name='maxPool1')(v)
    v = layers.Conv2D(filters=64, kernel_size=5,
                      activation="relu", padding="same")(v)
    v = layers.MaxPool2D(pool_size=2, name='maxPool2')(v)
    v = layers.Flatten()(v)
    fc1 = layers.Dense(units=512, activation="relu")(v)
    # Edited:
    # logits = layers.Dense(units=10)(fc1)
    logits = layers.Dense(units=100)(fc1)
    # Edited:
    # return logits
    model = tf.keras.Model(input_layer, logits)
    return model


# FIXME 2: It would be nice to be able to count the elements in a reasonably
# small dataset like cifar. The below doesn't seem to work (or takes too long).
def record_count(dataset):
    def init_fn(key):
        return 0

    def reduce_fn(value, state):
        return value + 1

    def finalize_fn(state):
        return state

    def key_fn(shape, type):
        return np.int64(0)

    reducer = tf.data.experimental.Reducer(init_fn, reduce_fn, finalize_fn)
    dataset = dataset.apply(tf.data.experimental.group_by_reducer(key_fn, reducer))
    count = tf.data.experimental.get_single_element(dataset).eval()
    return count


def _check_dataset(ds, crop_to):
    with tf.Session() as sess:
        ds_iter = ds.make_one_shot_iterator()
        # The records should be a (img, label) tuple.
        (img, label) = ds_iter.get_next()
        # FIXME 1: switch to uint8.
        assert img.dtype == cifar_ds.DTYPE
        assert img.shape.rank == 3 # W x H x D
        # Note: the batch dimension doesn't have a hard-coded size so as to allow
        # the network be flexible enough to use any batch sizes.
        # Default Tensorflow image format seems to be [depth, height, width). So
        # try to use that format (up until we need to switch channel to be first for
        # running on a GPU).
        # https://www.tensorflow.org/guide/performance/overview#use_nchw_imag
        assert (crop_to, crop_to,
                cifar_ds.COLOR_CHANNELS) == tuple(img.shape.as_list())
        assert cifar_ds.DEFAULT_DATA_SHAPE == tuple(img.shape.as_list())
        # FIXME 1: switch to uint8.
        # assert label.dtype == tf.uint8
        assert label.shape == ()
        label_val = label.eval()
        assert 0 <= label_val < 100
        # FIXME 2
        # assert record_count(ds) == cifar_ds.TRAIN_COUNT


def test_train_dataset():
    crop = cifar_ds.DEFAULT_IMAGE_SIZE
    ds = cifar_ds.train_dataset().map(cifar_ds.preprocess_fn(augment=False,
                                                             crop_to=crop))
    _check_dataset(ds, crop)


def test_eval_dataset():
    crop = cifar_ds.DEFAULT_IMAGE_SIZE
    ds = cifar_ds.eval_dataset().map(cifar_ds.preprocess_fn(augment=False,
                                                            crop_to=crop))
    _check_dataset(ds, crop)


def test_test_dataset():
    crop = cifar_ds.DEFAULT_IMAGE_SIZE
    ds = cifar_ds.test_dataset().map(cifar_ds.preprocess_fn(augment=False,
                                                            crop_to=crop))
    _check_dataset(ds, crop)


# FIXME 20: we should make this runs both for the standard and TPU estimator.
@pytest.mark.tpu_only
def test_with_estimator(estimator_fn):
    """Tests that the cifar dataset pipeline can generate data for a TPU.

    This is tested by running a train and evaluate job on a TPU using the
    datasets created by cifar.train_dataset(), cifar.eval_dataset() and
    cifar.test_dataset().

    Specific checks:
        1. estimator.evaluate() runs without errors when using
           cifar.eval_dataset().
        2. estimator.train()  runs without errors when using
           cifar.train_dataset().
        2. The model accuracy improves after training.
        3. estimator.evaluate() runs without errors when using
           cifar.test_dataset().
    """
    # Setup
    # These constants are similar to those chosen in test_estimator.py. Would
    # be nice to factor them out somewhere.
    batch_size = 128
    crop_to = 32 # Size expected by our test model.
    expected_accuracy = 0.99 # Expect overfitting.
    train_steps = 10000
    eval_steps = 5000 // batch_size # there are 5000 eval samples.
    test_steps = 10000 // batch_size # there are 1000 test samples.
    cifar100_classes = 100
    estimator = estimator_fn(keras_model_fn=_keras_model_fn,
                             train_batch_size=batch_size,
                             eval_batch_size=batch_size)

    # Create 3 input functions, for eval, train and test.
    def input_fn(ds_fn, params):
        del params
        # We must make any dataset call within the input_fn. Thus, this input_fn
        # cannot take a Dataset as a parameter, it must take a factory.
        # Otherwise, the dataset will be created outside of the training/eval
        # session.
        ds = ds_fn(cloud_storage=True)
        ds = ds.map(cifar_ds.preprocess_fn(augment=True, crop_to=crop_to))
        return ds.cache().repeat().batch(batch_size, drop_remainder=True)\
            .prefetch(1)

    train_input_fn = functools.partial(input_fn, cifar_ds.train_dataset)
    eval_input_fn = functools.partial(input_fn, cifar_ds.eval_dataset)
    test_input_fn = functools.partial(input_fn, cifar_ds.test_dataset)

    # Test
    # 1, 2, 3. Run estimator.evaluate() with eval_dataset(), estimator.train()
    # with train_dataset() and insure accuracy increases.
    test.util.check_train_and_eval(
        estimator, train_input_fn=train_input_fn, eval_input_fn=eval_input_fn,
        expected_post_train_accuracy=expected_accuracy, eval_steps=eval_steps,
    train_steps=train_steps, num_classes=cifar100_classes)

    # 4. Run estimator.evaluate() with train_dataset().
    results = estimator.evaluate(test_input_fn, steps=test_steps)
    # The reason for the poor test dataset performance (~30%) I *think* is due
    # to overfitting. Supporting this idea is that the eval step above is
    # getting ~99% accuracy. However, it's worth looking into more. (FIXME 23)
    # See gs://micronet_bucket1/pytest/test_with_estimator/20190723T173507/ for
    # results that seem to back this idea up.
    assert 0.2 < results['accuracy'] < 0.4
    # FIXME 22: create a propper assert near that takes into around the domain
    # restriction [0,1] and the resulting non-linear bounds.
