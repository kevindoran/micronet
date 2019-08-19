import efficientnet.efficientnet_builder as enet_builder
import efficientnet.efficientnet_model as enet_model
import tensorflow as tf
import micronet.estimator
import micronet.gcloud as gcloud
import micronet.models
import micronet.dataset.imagenet as imagenet_ds
import os
import argparse

TEST_CLASS = 145 # King penguin.
# EFFICIENTNET_CKPT_DIR = './resources/efficientnet-b0'
EFFICIENTNET_CKPT_DIR = 'gs://micronet_bucket1/models/efficientnet-b0/'
tf.logging.set_verbosity(tf.logging.INFO)


def custom_model(features, is_training):
    image_inputs = features
    image_inputs = micronet.models.normalize_image(image_inputs,
                                                   enet_builder.MEAN_RGB,
                                                   enet_builder.STDDEV_RGB)
    features, endpoints = enet_builder.build_model_base(
        image_inputs, model_name='efficientnet-b0', training=is_training)
    # Setting training=False will modify the network and remove some unneeded
    # computation. This is not sufficient to make all variables non-trainable.
    # To be selective with which variables should be the focus of training,
    # use a var_list of an optimizer. See below.
    with tf.variable_scope('test_branch'):
        sc = tf.layers.Conv2D(
            filters=1280,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=enet_model.conv_kernel_initializer,
            padding='same',
            use_bias=False)(endpoints['reduction_3'])
        # Note: do we need to configure training/not-training for bn?
        sc = enet_model.batchnorm(
            axis=-1,
            momentum=0.99,
            epsilon=1e-3)(sc)
        early_features = tf.keras.layers.GlobalAveragePooling2D()(sc)
        binary_logit = tf.layers.Dense(
            units=1,
            # If using a XX_with_logits, then the input to
            # the loss is expected to be un-normalized.
            activation=None,
            # activation='sigmoid',
            name='early_logits')(early_features)
    return binary_logit


def custom_loss_op(logits, labels, num_classes, weight_decay):
    logits_as_scalar = tf.reshape(logits, [-1, ])
    y = tf.cast(tf.math.equal(TEST_CLASS, labels), tf.float32)
    prediction = tf.nn.sigmoid(logits_as_scalar)
    weights = 0.99 * tf.ones(shape=y.shape, dtype=tf.float32) + 100.0 * y
    # weights = tf.ones(shape=y.shape, dtype=tf.float32)
    weights = tf.stop_gradient(weights)
    cross_entropy = tf.losses.log_loss(labels=y, predictions=prediction,
                                       weights=weights)
    to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    'test_branch')
    weight_sum = tf.add_n([tf.nn.l2_loss(v) for v in to_train
                           if 'batch_normalization' not in v.name])
    loss = cross_entropy + weight_decay * weight_sum
    return loss


def custom_train_op(loss, processor_type, batch_size, examples_per_decay,
                    decay_rate):
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate,
                                       # 0.1 as recommened by tensorflow docs.
                                       epsilon=0.1)
    if processor_type == micronet.estimator.ProcessorType.TPU:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    # This alteration was taken from resnet_main.py from tensorflow_tpu.
    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation. It's also present in EfficientNet.
    global_step = tf.train.get_global_step()
    test_branch_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         'test_branch')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step,
                                      var_list=test_branch_vars)
    return train_op, learning_rate


def custom_metric_fn(labels, logits):
    is_king_penguin = tf.math.equal(145, labels)
    cutoff = 0.50 # How to choose this?
    logits_as_scalar = tf.reshape(logits, [-1, ])
    prediction = tf.nn.sigmoid(logits_as_scalar)
    is_guessed = tf.math.greater(prediction, cutoff)
    accuracy = tf.metrics.accuracy(is_king_penguin, is_guessed)
    false_positives = tf.metrics.false_positives(is_king_penguin, is_guessed)
    true_positives = tf.metrics.true_positives(is_king_penguin, is_guessed)
    false_negatives = tf.metrics.false_negatives(is_king_penguin, is_guessed)
    true_negatives = tf.metrics.true_negatives(is_king_penguin, is_guessed)
    precision = tf.metrics.precision(is_king_penguin, is_guessed)
    return {'top_1_accuracy': accuracy,
            'false_positives': false_positives,
            'true_positives': true_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision}


def main():
    # Test-experiment identifier
    # Hard-coding the id makes it is easy to match commits to experiment notes.
    test_no = 4
    experiment_no = 3

    # Options
    parser = argparse.ArgumentParser(
        description='Run experiment {exp_no} for shortcutnet (test {test_no})'
            .format(exp_no=experiment_no, test_no=test_no)
    )
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Delete and replace any existing logs for this '
                             'experiment.')
    parser.add_argument('-c', '--continue-training', action='store_true',
                        help='Continue from a previous checkpoint if logs are'
                             ' already present.'
                             'experiment.')
    parser.add_argument('-s', '--allow-skip', action='store_true',
                        help='Allow skipping experiment versions.')
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
    allow_skip_minor = args.allow_skip
    target_tpu = args.tpu

    # Training options
    image_size = 224
    images_per_epoch = 1.2 * 1000 * 1000 # is this correct?
    train_images = images_per_epoch * 50
    train_batch_size = 128 * 8 # 16 runs out of mem, 8 doesn't.
    eval_batch_size = train_batch_size
    train_steps = train_images // train_batch_size
    num_eval_images = 64 * 2**10
    # steps_between_eval as 100 is nice when paying close attention to a
    # specific run. Choose a higher value for greater training performance.
    steps_between_eval = 200

    # Warm start settings
    warm_start = True
    warm_start_from_efficient_net = True
    if warm_start:
        # Warm starting from efficientnet should only be needed the first run.
        # It's not clear how warm start interacts with the default behaviour of
        # an estimator loading from it's own log dir when it's not empty.
        if warm_start_from_efficient_net:
            warm_start_settings = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=EFFICIENTNET_CKPT_DIR,
                vars_to_warm_start='efficientnet-b0')
        else:
            warm_start_settings = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from = 'gs://micronet_bucket1/models/experiments/2/1',
                vars_to_warm_start=['.*'])
    else:
        warm_start_settings = None

    # Input functions
    train_input_fn = imagenet_ds.create_train_input(
        image_size=image_size,
        num_parallel_calls=os.cpu_count()*2,
        for_tpu=True, autoaugment=False).input_fn
    eval_input_fn = imagenet_ds.create_train_input(
        image_size=image_size,
        num_parallel_calls=os.cpu_count()*2,
        for_tpu=True, autoaugment=False).input_fn

    # Train
    def train(estimator):
        micronet.estimator.train_and_eval(
            estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            train_steps=train_steps,
            steps_per_epoch=images_per_epoch//train_batch_size,
            num_eval_images=num_eval_images,
            steps_between_eval=steps_between_eval,
            eval_batch_size=eval_batch_size)

    # Eval only
    def eval_only(estimator):
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn,
            steps= num_eval_images // eval_batch_size)
        tf.logging.info('Eval results: %s', eval_results)

    # Estimator
    use_tpu = True
    gcloud_settings = gcloud.load_settings()
    model_dir = gcloud.experiment_dir(gcloud_settings, test_no, experiment_no,
                                      dir_exists_behaviour=dir_exists_behaviour,
                                      allow_skip_minor=allow_skip_minor)
    hparams = micronet.estimator.HParams(
        examples_per_epoch=images_per_epoch,
        examples_per_decay=100000)
    model_fn = micronet.estimator.create_model_fn(
        custom_model,
        processor_type=micronet.estimator.ProcessorType.TPU,
        loss_op_fn=custom_loss_op, train_op_fn=custom_train_op, hparams=hparams,
        metric_fn=custom_metric_fn)
    if use_tpu:
        def tpu_est():
            """Create a TPU. This function is used twice below."""
            est = micronet.estimator.create_tpu_estimator(
                gcloud_settings=gcloud_settings,
                model_dir=model_dir,
                model_fn=model_fn,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                warm_start_settings=warm_start_settings)
            return est

        if target_tpu:
            gcloud_settings.tpu_name = target_tpu
            train(tpu_est())
        else:
            with gcloud.start_tpu(gcloud_settings.project_name,
                                  gcloud_settings.tpu_zone) as tpu_name:
                # Override the TPU setting. The abstractions are not great here.
                gcloud_settings.tpu_name = tpu_name
                train(tpu_est())
    else:
        # CPU
        est = micronet.estimator.create_cpu_estimator(
            model_dir, model_fn, params={'batch_size': 64})
        train(est)


if __name__ == '__main__':
    main()
