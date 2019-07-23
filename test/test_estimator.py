import pytest
import micronet.estimator
import tensorflow as tf
import functools

# Model copied from:
# https://github.com/tensorflow/tpu/blob/master/models/experimental/cifar_keras/cifar_keras.py

TRAIN_FILE = 'gs://micronet_bucket1/cifar10_estimator_test/train.tfrecords'
EVAL_FILE = 'gs://micronet_bucket1/cifar10_estimator_test/eval.tfrecords'


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


# Using the copied version for the
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


# def model_fn(features, labels, mode, params):
#     # Instead of constructing a Keras model for training, build our loss function
#     # and optimizer in Tensorflow.
#     #
#     # N.B.  This construction omits some features that are important for more
#     # complex models (e.g. regularization, batch-norm).  Once
#     # `model_to_estimator` support is added for TPUs, it should be used instead.
#     loss = tf.reduce_mean(
#         tf.nn.sparse_softmax_cross_entropy_with_logits(
#             logits=logits, labels=labels
#         )
#     )
#     optimizer = tf.train.AdamOptimizer()
#     optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
#     train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
#
#     return tf.contrib.tpu.TPUEstimatorSpec(
#         mode=mode,
#         loss=loss,
#         train_op=train_op,
#         predictions={
#             "classes": tf.argmax(input=logits, axis=1),
#             "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#         }
#     )


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
    eval_steps = 100
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

    # 3. Evaluate using the untrained estimator.
    results = estimator.evaluate(eval_input_fn, steps=eval_steps)
    # TODO: make a reusable CDF_inverse function to easily calculate expected
    # random results.
    classes = 10
    random_chance = 1/classes
    assert random_chance/2 < results['accuracy'] < random_chance*2

    # 4. Check that the model can be trained.
    estimator.train(input_fn=train_input_fn, max_steps=10000)

    # 5. Check that the model accuracy has increased.
    results = estimator.evaluate(eval_input_fn, steps=eval_steps)
    min_threshold = 0.65
    assert min_threshold < results['accuracy']
