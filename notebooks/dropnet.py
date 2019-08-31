import efficientnet.efficientnet_builder as enet_builder
import efficientnet.efficientnet_model as enet_model
import tensorflow as tf
import micronet.estimator
import micronet.gcloud as gcloud
import micronet.models
import micronet.dataset.imagenet as imagenet_ds
import os
import argparse
import numpy as np

EFFICIENTNET_CKPT_DIR = 'gs://micronet_bucket1/models/efficientnet-b0/'
tf.logging.set_verbosity(tf.logging.INFO)
use_tpu = True


def model_fn(features, labels, mode, params):
    assert mode == tf.estimator.ModeKeys.EVAL
    batch_size = params['batch_size']
    image_inputs = features
    image_inputs = micronet.models.normalize_image(image_inputs,
                                                   enet_builder.MEAN_RGB,
                                                   enet_builder.STDDEV_RGB)
    logits, endpoints = enet_builder.build_model(
        image_inputs, model_name='efficientnet-b0', training=True)
    softmax_logits = tf.nn.softmax(logits, name='orig_softmax')
    # Why is loss a requirement when just evaluating?
    num_classes = logits.get_shape()[1]
    one_hot_labels = tf.one_hot(labels, num_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=softmax_logits,
        onehot_labels=one_hot_labels,
        # What does this do?
        label_smoothing=0.1)
    estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL, loss=cross_entropy,
        eval_metrics=(metric_fn, [labels, logits]))
    if not use_tpu:
        estimator_spec = estimator_spec.as_estimator_spec()
    return estimator_spec


def metric_fn(labels, logits):
    orig_guess = tf.argmax(logits, axis=1)
    accuracy = tf.metrics.accuracy(labels, orig_guess)
    metrics = {'accuracy': accuracy}
    return metrics


def main():
    # Test-experiment identifier
    # Hard-coding the id makes it is easy to match commits to experiment notes.
    test_major = 2
    test_minor = 1
    test_patch = 5

    # Options
    parser = argparse.ArgumentParser(
        description='Run experiment {major}.{minor}.{patch}'
            .format(major=test_major, minor=test_minor, patch=test_patch)
    )
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Delete and replace any existing logs for this '
                             'experiment.')
    parser.add_argument('-c', '--continue-training', action='store_true',
                        help='Continue from a previous checkpoint if logs are'
                             ' already present.'
                             'experiment.')
    parser.add_argument('-s', '--allow-skip', action='store_true',
                        help='Allow skipping experiment patch versions.')
    parser.add_argument('-t', '--tpu', type=str, required=False,
                        help='Use a specific TPU.')
    args = parser.parse_args()
    overwrite = args.overwrite
    continue_training = args.continue_training
    if overwrite:
        dir_exists_behaviour = gcloud.DirExistsBehaviour.OVERWRITE
    elif continue_training:
        dir_exists_behaviour = gcloud.DirExistsBehaviour.CONTINUE
    else:
        dir_exists_behaviour = gcloud.DirExistsBehaviour.FAIL
    if overwrite and continue_training:
        raise Exception('Either --overwrite or --continue-training may be '
                        'provided, but not both')
    allow_skip_patch = args.allow_skip
    target_tpu = args.tpu

    # Training options
    image_size = 224
    eval_batch_size = 128 * 8
    num_eval_images = 10 * 64 * 2**10

    # Warm start
    warm_start_settings = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=EFFICIENTNET_CKPT_DIR,
        vars_to_warm_start='efficientnet-b0')

    # Input functions
    eval_input_fn = imagenet_ds.create_train_input(
        image_size=image_size,
        num_parallel_calls=os.cpu_count(),
        for_tpu=True, autoaugment=False).input_fn

    # Eval only
    def eval_only(estimator):
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn,
            steps= num_eval_images // eval_batch_size)
        tf.logging.info('Eval results: %s', eval_results)

    # Estimator
    gcloud_settings = gcloud.load_settings()
    model_dir = gcloud.experiment_dir(
        gcloud_settings, test_major, test_minor, test_patch,
        dir_exists_behaviour=dir_exists_behaviour,
        allow_skip_minor=allow_skip_patch)
    if use_tpu:
        def tpu_est():
            """Create a TPU. This function is used twice below."""
            est = micronet.estimator.create_tpu_estimator(
                gcloud_settings=gcloud_settings,
                model_dir=model_dir,
                model_fn=model_fn,
                # train_batch_size=None,
                eval_batch_size=eval_batch_size,
                warm_start_settings=warm_start_settings)
            return est
        if target_tpu:
            gcloud_settings.tpu_name = target_tpu
            eval_only(tpu_est())
        else:
            with gcloud.start_tpu(gcloud_settings.project_name,
                                  gcloud_settings.tpu_zone) as tpu_name:
                # Override the TPU setting. The abstractions are not great here.
                gcloud_settings.tpu_name = tpu_name
                eval_only(tpu_est())
    else:
        # CPU
        est = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            params={'batch_size':eval_batch_size},
            warm_start_from=warm_start_settings)
        eval_only(est)


if __name__ == '__main__':
    main()
