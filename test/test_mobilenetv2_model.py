import pytest
import micronet.models.mobilenetv2 as mobilenetv2
import micronet.models.xiaochus_mobilenetv2 as xiaochus_mobilenetv2
import micronet.dataset.cifar as cifar_ds
import micronet.estimator
import tensorflow as tf
import test.util


def test_num_trainable_params():
    """Tests that the model has the expected number of trainable parameters."""
    input_shape = (1024, 1024, 3)
    model = mobilenetv2.create_model(input_shape=input_shape)
    # FIXME
    #assert test.util.count_trainable_params(model) \
    #       == mobilenetv2.rough_num_trainable_params(
    #            alpha=1.0, input_shape=input_shape, classes=1000)
    # FIXME: 3,504,872 is ~50,000 - 100,000 more parameters than what the
    #        paper mentions.
    assert test.util.count_trainable_params(model) == 3504872


def keras_model_fn():
    model = mobilenetv2.create_model(input_shape=cifar_ds.DEFAULT_DATA_SHAPE,
                                     classes=cifar_ds.CLASSES)
    return model


def xiaochus_model_fn():
    model = xiaochus_mobilenetv2.MobileNetv2(input_shape=cifar_ds.DEFAULT_DATA_SHAPE,
                                             k=cifar_ds.CLASSES)
    return model


@pytest.fixture
def cifar_dataset_fn(request):
    is_cloud = request.config.getoption('--cloud', default=False)

    def dataset_fn():
        ds = cifar_ds.train_dataset(cloud_storage=is_cloud)
        return ds
    return dataset_fn


# TODO: mostly copied from test_cifar_linear_model. Could be factored a bit.
@pytest.mark.slow
def test_is_trainable(estimator_fn, cifar_dataset_fn, is_cloud):
    """Test that that training and evaluation run as expected.

    Tests that:
        1. The untrained model can be evaluated, and that there is about 1%
           accuracy.
        2. The model can be trained.
        3. The trained model has higher accuracy (~20%).
    """
    # Setup
    if is_cloud:
        batch_size = 1024 # (128 per cores, 8 cores).
        eval_count = 1024
        train_steps = int(0.5 * 1000 * 1000)
    else:
        batch_size = 64
        eval_count = 256
        train_steps = 10
    eval_steps = int(eval_count / batch_size)
    assert eval_steps * batch_size == eval_count
    estimator = estimator_fn(keras_model_fn, batch_size, batch_size)

    # TODO: Move to cifar.dataset
    def input_fn(params):
        # Only the TPUEstimator needs to pass batch_size to the input_fn.
        if 'batch_size' in params:
            assert params['batch_size'] == batch_size
        del params
        ds = cifar_dataset_fn()
        map_fn = cifar_ds.preprocess_fn(augment=False,
                                        crop_to=cifar_ds.MAX_IMAGE_SIZE)
        # I'm not exactly sure what is best here for performance.
        # TODO: consider using map_and_batch.
        # When to repeat?
        ds = ds.repeat()
        # When to cache?
        ds = ds.cache()
        # Why this initial prefetch?
        ds = ds.prefetch(batch_size)
        # Replacing map and batch with the map_and_batch.
        # ds = ds.batch(params['batch_size'], drop_remainder=True)
        vcpu_count = 16
        ds = ds.apply(tf.contrib.data.map_and_batch(
            map_func=map_fn, batch_size=batch_size,
            drop_remainder=True, num_parallel_calls=vcpu_count))
        # Cache here too?
        ds = ds.cache()
        ds = ds.prefetch(micronet.estimator.ITERATIONS_PER_LOOP)
        return ds


    # 1, 2, 3
    expected_accuracy = 0.5
    test.util.check_train_and_eval(
        estimator, input_fn, input_fn, train_steps, eval_steps, num_classes=100,
        expected_post_train_accuracy=expected_accuracy)
