import pytest
import micronet.estimator
import tensorflow as tf
import functools
import test.util

# Model copied from:
# https://github.com/tensorflow/tpu/blob/master/models/experimental/cifar_keras/cifar_keras.py

TRAIN_FILE = 'gs://micronet_bucket1/cifar10_estimator_test/train.tfrecords'
EVAL_FILE = 'gs://micronet_bucket1/cifar10_estimator_test/eval.tfrecords'

# These step counts and accuracy are used by tests in this file. They are
# dependent on the keras_model_fn, which is fixed.
TRAIN_STEPS = 10000
EVAL_STEPS = 100
EXPECTED_ACCURACY = 0.7
NUM_CIFAR10_CLASSES = 10

# This input function is copied from the tensorflow/tpu repository, and thus,
# is assumed to be working. Although, there is two edits as mentioned in the
# comments below.
def keras_model_fn():
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
    logits = layers.Dense(units=10)(fc1)
    # Changed:
    #return logits
    model = tf.keras.Model(input_layer, logits)
    return model


# This input function is copied from the tensorflow/tpu repository, and thus,
# is assumed to be working.
def input_fn(input_file, params):
    """Read CIFAR input data from a TFRecord dataset."""
    def parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features["image"], tf.uint8)
        image.set_shape([3*32*32])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        # The output of this has shape (32, 32, 3).
        # FIXME 15: Storing the features in the (32, 32, 3) dimension will
        # avoid a reshape, which is potentially an expensive operation on
        # TPUs.
        image = tf.transpose(tf.reshape(image, [3, 32, 32]))
        label = tf.cast(features["label"], tf.int32)
        return image, label

    dataset = tf.data.TFRecordDataset([input_file])
    dataset = dataset.map(parser, num_parallel_calls=params['batch_size'])
    dataset = dataset.prefetch(4 * params['batch_size']).cache().repeat()
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset


train_input_fn = functools.partial(input_fn, TRAIN_FILE)
eval_input_fn = functools.partial(input_fn, EVAL_FILE)


@pytest.mark.tpu_only
def test_get_cluster_resolver(gcloud_settings):
    assert gcloud_settings
    # FIXME 12: add more checks.

    # Test
    # 1. Insure no exceptions are raised an a resolver is returned.
    resolver = micronet.estimator.get_cluster_resolver(gcloud_settings)
    assert resolver


@pytest.mark.tpu_only
def test_create_model_fn(gcloud_settings, gcloud_temp_path):
    """Tests that create_model_fn() creates an estimator compatible model_fn.

    Tests that:
        1. create_model_fn() runs without error.
        2. An estimator can be constructed with the created model_fn.
        3. The estimator is able to evaluate samples.
        4. The estimator can be trained.
        5. The estimator accuracy increases after training.
    """
    # Setup
    cluster_resolver = micronet.estimator.get_cluster_resolver(gcloud_settings)
    run_config = tf.contrib.tpu.RunConfig(
        cluster=cluster_resolver,
        model_dir=gcloud_temp_path,
        save_checkpoints_secs=3600,
        session_config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=100,
            num_shards=8))

    # Test
    # 1. Create the model_fn via create_model_fn().
    model_fn = micronet.estimator.create_model_fn(
        keras_model_fn, processor_type=micronet.estimator.ProcessorType.TPU)
    assert model_fn

    # 2. Construct a TPUEstimator.
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=True,
        config=run_config,
        train_batch_size=128,
        eval_batch_size=128,
        # We don't need to save the model.
        export_to_tpu=False)
        # export_to_cpu doesn't seem to be released as of tf 1.13.1.
        # export_to_cpu=False)
    assert estimator

    # 3, 4, 5. Test pre-train eval, training and post-train eval.
    test.util.check_train_and_eval(
        estimator, train_input_fn, eval_input_fn, train_steps=TRAIN_STEPS,
        eval_steps=EVAL_STEPS, num_classes=NUM_CIFAR10_CLASSES,
        expected_post_train_accuracy=EXPECTED_ACCURACY)


@pytest.mark.tpu_only
def test_create_tpu_estimator(gcloud_settings, gcloud_temp_path):
    """Tests that create_tpu_estimator() creates a usable TPUEstimator.

    Tests that:
        1. create_tpu_estimator() runs without error.
        2. The estimator is able to evaluate samples.
        3. The estimator can be trained.
        4. The estimator accuracy increases after training.
        5. FIXME 16: implement the test 'A warning is raised if batch size is
                     not divisible by 128'.
    """

    # Setup.
    # Create the estimator compatible model_fn from the Keras model_fn.
    # This method is tested elsewhere.
    model_fn = micronet.estimator.create_model_fn(
        keras_model_fn, processor_type=micronet.estimator.ProcessorType.TPU)
    batch_size = 128

    # Test
    # 1. Create an estimator without error.
    estimator = micronet.estimator.create_tpu_estimator(
        gcloud_settings, gcloud_temp_path, model_fn,
        train_batch_size=batch_size, eval_batch_size=batch_size)
    assert estimator

    # 2, 3, 4. Test pre-train eval, training and post-train eval.
    test.util.check_train_and_eval(
        estimator, train_input_fn, eval_input_fn, train_steps=TRAIN_STEPS,
        eval_steps=EVAL_STEPS, num_classes=NUM_CIFAR10_CLASSES,
        expected_post_train_accuracy=EXPECTED_ACCURACY)


# FIXME 20: we also need a test for the non-TPU Estimator case. We need some
#           sort of test parameterization.
@pytest.mark.tpu_only
def test_estimator_fixture(estimator_fn):
    """Tests that the estimator_fn fixture creates a working estimator factory.

    Tests that:
        1. estimator_fn is not None, and returns a non-None estimator.
        2. The estimator is able to evaluate samples.
        3. The estimator can be trained.
        4. The estimator accuracy increases after training.
    """
    # Setup
    batch_size = 128

    # Test
    # 1. estimator_fn shouldn't be None, and shouldn't return None.
    assert estimator_fn
    estimator = estimator_fn(keras_model_fn=keras_model_fn,
                             train_batch_size=batch_size,
                             eval_batch_size=batch_size)
    assert estimator

    # 2, 3, 4 Eval and train work.
    test.util.check_train_and_eval(
        estimator, train_input_fn, eval_input_fn, train_steps=TRAIN_STEPS,
        eval_steps=EVAL_STEPS, num_classes=NUM_CIFAR10_CLASSES,
        expected_post_train_accuracy=EXPECTED_ACCURACY)

