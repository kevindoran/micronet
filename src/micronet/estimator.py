import tensorflow as tf
from enum import Enum
import functools
import micronet

# Number of iterations (batches) to run on the TPU workers before returning
# control to the master (not sure if the terminology is correct here).
# What is a good number? What does it depend on?
# For the test_estimator.py tests using the test model, training was almost
# twice as fast using iterations_between_model_update set to 100 as opposed to
# set at 16. Just a data point. Still not sure how the number should be chosen.
# From: https://cloud.google.com/tpu/docs/troubleshooting
#    "iterations_per_loop can be set to a very large value, with the only
#    downside being that logging messages and checkpointing can only occur at
#    the end of a loop."
ITERATIONS_PER_LOOP = 100
checkpoints_max = 0

ProcessorType = Enum('ProcessorType', 'CPU, GPU, TPU')
learning_rate_base = 0.045


def get_cluster_resolver(gcloud_settings):
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        # In the future, the tpu parameter might support lists.
        tpu=gcloud_settings.tpu_name,
        zone=gcloud_settings.tpu_zone,
        project=gcloud_settings.project_name)
    return tpu_cluster_resolver


def create_tpu_estimator(gcloud_settings, model_dir, model_fn, train_batch_size,
                         eval_batch_size):
    tpu_cluster_resolver = get_cluster_resolver(gcloud_settings)
    if train_batch_size % 128:
        raise Warning('Train batch size should be divisible by 128 as the XLA '
                      'compiler will likely pad the batch size to 128.')
    if eval_batch_size % 128:
        raise Warning('If evaluating on a TPU, eval batch size should be '
                      'divisible by 128 as the XLA compiler will likely pad '
                      'the batch size to 128.')

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        # Directory to save model parameters, graph etc. Also used as a source
        # directory when loading checkpoints.
        model_dir=model_dir,
        keep_checkpoint_max=checkpoints_max,
        session_config=tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            # The number of train steps running in TPU system before returning
            # to CPU host for each `Session.run`. This means that global step is
            # increased `iterations_per_loop` times in one `Session.run`. It is
            # recommended to be set as number of global steps between each
            # checkpoint.
            iterations_per_loop=ITERATIONS_PER_LOOP,
            # Deprecated: num_shards,
            # num_cores_per_replica:  Useful? Used for model parallelism.
            # per_host_input_for_training: No idea what this is.
            # initial_infeed_sleep_secs: delay for infeed thread. Useful to
            #     avoid issues if the model requires a long compilation time.
            # input_partition_dims: A nested list to describe the partition
            #       dims for all the tensors from input_fn(). The structure of
            #       input_partition_dims must match the structure of `features`
            #       and `labels` from input_fn(). The total number of partitions
            #       must match `num_cores_per_replica`. For example, if
            #       input_fn() returns two tensors: images with shape [N, H, W,
            #       C] and labels [N].  input_partition_dims = [[1, 2, 2, 1],
            #       None] will split the images to 4 pieces and feed into 4 TPU
            #       cores. labels tensor are directly broadcasted to all the TPU
            #       cores since the partition dims is `None`. Current
            #       limitations: This feature is only supported with the
            #       PER_HOST_V2 input mode.
            # eval_training_input_configuration: If `SLICED`, `input_fn` is only
            #       invoked once on host 0 and the tensors are broadcasted to
            #       all other replicas. Unlike
            #       per_host_input_for_training=BROADCAST, each replica will
            #       only get a slice of the data instead of a whole copy. If
            #       `PER_HOST_V1`, the behaviour is determined by
            #       per_host_input_for_training.
        )
    )


    estimator = tf.contrib.tpu.TPUEstimator(
        # A function that  returns an EstimatorSpec or TPUEstimatorSpec.
        model_fn=model_fn,
        use_tpu=True,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        # model_dir=None, # Inherited from runConfig.
        # predict_batch_size=FLAGS.batch_size, possibly needed. Put if I use
        # params={'key':'value'}. Optional. Don't need it yet.
        # Okay, I get wornings without params present.
        params={},
        # prediction on the CPU, then we don't need TPU prediction.
        # export_to_cpu=True, I might need this option for prediction.
        # batch_axis: not sure how to use this.
    )
    return estimator


def create_cpu_estimator(model_dir, model_fn):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir)
    return estimator


def metric_fn(labels, logits):
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}


# TODO: is 'op' correct here?
def create_loss_op(logits, labels):
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    return loss


def create_train_op(loss, processor_type):
    # TODO: decide what learning rate to use.
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        tf.train.get_global_step(),
        decay_steps=100000,
        decay_rate=0.98)
    # FIXME 9: RMSPropOptimizer doesn't seem to be working.
    # MobileNetv2 paper uses RMSPropOptimizer with decay and momentum as 0.9.
    # RMSProp doesn't seem to be working for me on CPU or TPU.
    #optimizer = tf.train.RMSPropOptimizer(learning_rate_base, decay=0.90, momentum=0.9)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer()
    if processor_type == ProcessorType.TPU:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    # TODO: is this the correct value for the step argument?
    train_op = optimizer.minimize(loss, tf.train.get_global_step())
    return train_op


def create_model_fn(keras_model_fn, processor_type):
    """Bind the processor type parameter and return the resulting function.

    This way of creating the model_fn means we don't need to use to pass
    parameters through the estimator and take them via the params parameter.
    That mechanism seems flaky and seems to have poor encapsulation.
    """
    fn = functools.partial(_model_fn, keras_model_fn, processor_type)
    return fn


def create_model_fn_experimental(keras_model_fn, processor_type):
    """Bind the processor type parameter and return the resulting function.

    This way of creating the model_fn means we don't need to use to pass
    parameters through the estimator and take them via the params parameter.
    That mechanism seems flaky and seems to have poor encapsulation.
    """
    # TODO: fix this up to add settings dependency.
    def tpu_model_fn(*args, **kwargs):
        settings = micronet.gcloud.load_settings()
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            # In the future, the tpu parameter might support lists.
            tpu=settings.tpu_name,
            zone=settings.tpu_zone,
            project=settings.project_name)
        keras_model = keras_model_fn(*args, **kwargs)
        tpu_model = tf.contrib.tpu.keras_to_tpu_model(
            keras_model, strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tpu_cluster_resolver))
        return tpu_model_fn
    #if processor_type == ProcessorType.CPU:
    #    fn = functools.partial(model_fn, keras_model_fn, processor_type)
    #elif processor_type == ProcessorType.TPU:
    #    fn = functools.partial(model_fn, tpu_model_fn, processor_type)
    #else:
    #    raise Exception("Unexpected processor type: {}".format(processor_type))
    fn = functools.partial(_model_fn, tpu_model_fn, processor_type)
    return fn


# Interestingly, it looks like the params argument is optional, as long as it
# is also not passed to the estimator. So removing from here, as parameter.
# Original signature:
#     def model_fn(processor_type, features, labels, mode, params):
# Okay, all of a sudden, I started getting an error complaining that the params
# parameter is needed:
#       if 'params' not in model_fn_args:
#         raise ValueError('model_fn ({}) does not include params argument, '
#                          'required by TPUEstimator to pass batch size as '
# >                        'params[\'batch_size\']'.format(self._model_fn))
# ValueError: model_fn (functools.partial(<function model_fn at 0x7f39aa67f950>, <function create_model at 0x7f39aa5f4bf8>, <ProcessorType.TPU: 3>)) does not include params argument, required by TPUEstimator to pass batch size as params['batch_size']
#def model_fn(keras_model_fn, processor_type, features, labels, mode):
def _model_fn(keras_model_fn, processor_type, features, labels, mode, params):
    del params
    image = features
    # Labels should be scalar values (not one-hot encoded).
    tf.ensure_shape(labels, shape=(None,))
    logit_outputs = keras_model_fn()(image, training=mode==tf.estimator.ModeKeys.TRAIN)
    loss_op = create_loss_op(logit_outputs, labels)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO: is it okay to have the create model within an if? Does it
        #       prevent some sort of model reuse that would otherwise happen?
        # Not sure if allowed in if statement on TPU
        #logit_outputs = keras_model_fn()(image, training=True)
        #loss_op = create_loss_op(logit_outputs, labels)
        train_op = create_train_op(loss_op, processor_type)
        # FIXME X: how to return either TPU or non TPU estimator spec?
        estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss_op,
                                                         train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        # TODO: What does the training option do?
        #logit_outputs = keras_model_fn()(image, training=False)
        #loss_op = create_loss_op(logit_outputs, labels)
        # Does the eval_metrics need to be (metric_fn, [labels, outputs])?
        # FIXME X: how to return either TPU or non TPU estimator spec?
        # estimator = tf.estimator.EstimatorSpec(mode=mode, loss=loss_op,
        #                                       eval_metric_ops=(metric_fn,))
        # From the TPUEstimatorSpec source:
        #     For evaluation, `eval_metrics `is a tuple of `metric_fn` and
        #     `tensors`, where `metric_fn` runs on CPU to generate metrics and
        #     `tensors` represents the `Tensor`s transferred from TPU system to
        #     CPU host and passed to `metric_fn`.
        estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=loss_op,
            eval_metrics=(metric_fn, [labels, logit_outputs]))
    elif mode == tf.estimator.ModeKeys.PREDICT:
        raise Exception('Unsupported.')
    else:
        raise Exception('unexpected mode: {}'.format(mode))
    if processor_type == ProcessorType.CPU:
        estimator_spec = estimator_spec.as_estimator_spec()
    return estimator_spec
