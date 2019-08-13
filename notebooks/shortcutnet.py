import efficientnet.efficientnet_builder as net_builder
import tensorflow as tf
import micronet.estimator
import micronet.gcloud as gcloud
import micronet.models
import micronet.dataset.imagenet as imagenet_ds
import os

TEST_CLASS = 145 # King penguin.
# EFFICIENTNET_CKPT_DIR = './resources/efficientnet-b0'
EFFICIENTNET_CKPT_DIR = 'gs://micronet_bucket1/models/efficientnet-b0/'
tf.logging.set_verbosity(tf.logging.INFO)


def custom_model(features, is_training):
    image_inputs = features
    image_inputs = micronet.models.normalize_image(image_inputs,
                                                   net_builder.MEAN_RGB,
                                                   net_builder.STDDEV_RGB)
    features, endpoints = net_builder.build_model_base(
        image_inputs, model_name='efficientnet-b0', training=False)
    # FIXME: this is not zero, but 208! I manually added "trainable=False" to
    # many layers (Cov2D and batch norm) in the model build step. A better
    # option may be to specify via a whitelist the variables that I want
    # optimized.
    #tv = tf.trainable_variables(); import pdb;pdb.set_trace()
    num_tv = 208
    assert num_tv == len(tf.trainable_variables())
    # The 6th layer outputs the second resolution reduction.
    # See the Efficientnet readme for use of the model builder API. See the
    # paper for the layer breakdown. https://arxiv.org/pdf/1905.11946.pdf
    layer_6_outputs = endpoints['reduction_3']
    # 6th layer output is 40 channels at 56x56 resolution.
    layer_6_output_res = 28
    layer_6_output_channels = 40
    early_features = tf.keras.layers.GlobalAveragePooling2D()(layer_6_outputs)
    binary_logit = tf.layers.Dense(units=1,
                                   # If using a XX_with_logits, then the input to
                                   # the loss is expected to be un-normalized.
                                   activation=None,
                                   # activation='sigmoid',
                                   name='early_logits')(early_features)
    # TODO remove debugging.
    # tv = tf.trainable_variables()
    # import pdb;pdb.set_trace()
    assert 2 + num_tv == len(tf.trainable_variables()), '1 bias and 1 dense trainable.'
    # Calculate the number of features out at layer 6.
    # assert 0 == len(tf.trainable_variables())
    return binary_logit


def custom_loss_op(logits, labels, num_classes, weight_decay):
    #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
    #    labels=is_penguin,
    #    logits=tf.squeeze(logits),
    #    # A value of pos_weight < 1 decreases the false positive count.
    #    # What? Doesn't support float types?
    #     pos_weight=0.5)
    # reduce_mean or reduce_sum, which to use?
    # Possibly leading to negative loss:
    #cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #    labels=tf.cast(is_penguin, tf.float32),
    #    logits=logits_as_scalar))
    logits_as_scalar = tf.reshape(logits, [-1, ])
    y = tf.cast(tf.math.equal(TEST_CLASS, labels), tf.float32)
    # loss = max(x, 0) - x * y + log(1 + exp(-abs(x))
    prediction = tf.nn.sigmoid(logits_as_scalar)
    weights = tf.ones(shape=y.shape, dtype=tf.float32) + 1000 * y
    cross_entropy = tf.losses.log_loss(labels=y, predictions=prediction,
                                       weights=weights)

    early_logit = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    'early_logits')
    weight_sum = tf.add_n([tf.nn.l2_loss(v) for v in early_logit])
    # Interesting note: The following line was the source of a bug which was
    # time consuming to track down. The '+' was accidentally a '*'. The result
    # being estimation accuracy reduced to random at 10% for cifar10.
    # TODO: testing without weight decay.
    loss = cross_entropy + weight_decay * weight_sum
    return loss


def custom_metric_fn(labels, logits):
    is_king_penguin = tf.math.equal(145, labels)
    cutoff = 0.9 # How to choose this?
    logits_as_scalar = tf.reshape(logits, [-1, ])
    prediction = tf.nn.sigmoid(logits_as_scalar)
    is_guessed = tf.math.greater(prediction, cutoff)
    accuracy = tf.metrics.accuracy(is_king_penguin, is_guessed)
    false_positives = tf.metrics.false_positives(is_king_penguin, prediction)
    true_positives = tf.metrics.true_positives(is_king_penguin, prediction)
    false_negatives = tf.metrics.false_negatives(is_king_penguin, prediction)
    true_negatives = tf.metrics.true_negatives(is_king_penguin, prediction)
    precision = tf.metrics.precision(is_king_penguin, prediction)
    return {'top_1_accuracy': accuracy,
            'false_positives': false_positives,
            'true_positives': true_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision}


def main():
    gcloud_settings = gcloud.load_settings()
    model_dir = 'gs://micronet_bucket1/models/shortcutnet_tpu_3'
    restore_from= 'gs://micronet_bucket1/models/shortcutnet1_tpu/'
    image_size = 224
    images_per_epoch = 1.2 * 1000 * 1000 # is this correct?
    train_images = images_per_epoch * 10
    train_batch_size = 128 * (2**3)
    eval_batch_size = 128
    train_steps = train_images // train_batch_size
    num_eval_images = 100 * 2**10
    steps_between_eval = 80
    hparams = micronet.estimator.HParams(examples_per_decay=100000)
    model_fn = micronet.estimator.create_model_fn(
        custom_model, processor_type=micronet.estimator.ProcessorType.TPU,
        metric_fn=custom_metric_fn, loss_fn=custom_loss_op, hparams=hparams)
    warm_start_settings = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=EFFICIENTNET_CKPT_DIR,
        vars_to_warm_start='efficientnet-b0')
    # Warm starting from efficientnet-b0 should only be needed the first run.
    # warm_start_settings = restore_from
    warm_start_settings = None
    est = micronet.estimator.create_tpu_estimator(
              gcloud_settings=gcloud_settings,
              model_dir=model_dir,
              model_fn=model_fn,
              train_batch_size=train_batch_size,
              eval_batch_size=eval_batch_size,
              warm_start_settings=warm_start_settings)
    cpu_model_fn = micronet.estimator.create_model_fn(
        custom_model, processor_type=micronet.estimator.ProcessorType.CPU,
        metric_fn=custom_metric_fn, loss_fn=custom_loss_op)
    cpu_est = micronet.estimator.create_cpu_estimator(model_dir, cpu_model_fn,
                                                      params={'batch_size': 64})
    train_input_fn = imagenet_ds.create_train_input(
        image_size=image_size,
        num_parallel_calls=os.cpu_count(),
        for_tpu=True, autoaugment=False).input_fn
    eval_input_fn = imagenet_ds.create_train_input(
        image_size=image_size,
        num_parallel_calls=os.cpu_count(),
        for_tpu=True, autoaugment=False).input_fn

    micronet.estimator.train_and_eval(
        est,
        #cpu_est,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_steps,
        steps_per_epoch=images_per_epoch//train_batch_size,
        num_eval_images=num_eval_images,
        steps_between_eval=steps_between_eval,
        eval_batch_size=eval_batch_size)


if __name__ == '__main__':
    main()