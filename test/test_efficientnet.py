import pytest
import micronet.dataset.imagenet as imagenet_ds
import test.util
import efficientnet.efficientnet_builder as efficientnet_builder
import tensorflow as tf


def keras_model_fn(input_tensor, is_training):
    def normalize_features(features, mean_rgb, stddev_rgb):
        """Normalize the image given the means and stddevs."""
        # TODO: support GPU by using shape=[3, 1, 1]
        features -= tf.constant(mean_rgb, shape=[1, 1, 3],
                                dtype=features.dtype)
        features /= tf.constant(stddev_rgb, shape=[1, 1, 3],
                                dtype=features.dtype)
        return features
    normalized_features = normalize_features(input_tensor,
                                             efficientnet_builder.MEAN_RGB,
                                             efficientnet_builder.STDDEV_RGB)
    logits, _ = efficientnet_builder.build_model(
        normalized_features,
        model_name='efficientnet-b0',
        training=is_training)
    return logits
    #logits = test.util.example_keras_fn(1000)(input_tensor, True)
    #return logits


@pytest.mark.slow
def test_training(estimator_fn, machine_settings):
    # Setup
    image_size = 224
    batch_size = 128
    train_image_count = 2**21 # (~2 million)
    train_step_count = train_image_count // batch_size
    assert train_step_count * batch_size == train_image_count
    eval_image_count = 2**14 # (~16 thousand)
    eval_step_count = eval_image_count // batch_size
    assert eval_step_count * batch_size == eval_image_count
    train_input_fn = imagenet_ds.create_train_input(
        image_size, num_parallel_calls=machine_settings.num_vcpu,
        for_tpu=machine_settings.is_cloud).input_fn
    eval_input_fn = imagenet_ds.create_eval_input(
        image_size, num_parallel_calls=machine_settings.num_vcpu,
        for_tpu=machine_settings.is_cloud).input_fn
    est = estimator_fn(keras_model_fn=keras_model_fn,
                       train_batch_size=batch_size,
                       eval_batch_size=batch_size)

    # Test
    test.util.check_train_and_eval(est, train_input_fn=train_input_fn,
                                   eval_input_fn=eval_input_fn,
                                   train_steps=train_step_count,
                                   eval_steps=eval_step_count,
                                   num_classes=imagenet_ds.NUM_CLASSES,
                                   expected_post_train_accuracy=0.3)
