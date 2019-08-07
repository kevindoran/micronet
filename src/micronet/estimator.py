import tensorflow as tf
from enum import Enum
import functools
import time
import efficientnet
import inspect

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
SAVE_CHECKPOINTS_STEPS = ITERATIONS_PER_LOOP
checkpoints_max = 0

ProcessorType = Enum('ProcessorType', 'CPU GPU TPU')

# More parameters to be included.
const_learning_rate = 0.045
learning_rate_base = 0.5

DEFAULT_SKIP_HOST = False
# FIXME 28. Choose appropriate values.
DEFAULT_DECAY_RATE = 0.9
DEFAULT_WEIGHT_DECAY = 1e-5

TOP_1_ACCURACY_KEY = 'top_1_accuracy'
TOP_5_ACCURACY_KEY = 'top_5_accuracy'

# Currently not used, as Python 3.5 doesn't have support for defaults parameter.
# HParams = namedtuple('HParams', [
#                                  # Required
#                                  # TODO: processor_type should be deducible and
#                                  # not part of this list.
#                                  'processor_type',
#                                  'num_classes',
#                                  # Optional
#                                  'examples_per_decay',
#                                  'weight_decay',
#                                  'decay_rate',
#                                  'skip_host_call'],
#                              defaults=[ # Defaults are filled in reverse order.
#                                  DEFAULT_SKIP_HOST,
#                                  DEFAULT_DECAY_RATE,
#                                  DEFAULT_WEIGHT_DECAY,
#                                  None, # None will be interpreted as 'examples_per_epoch'.
#                                  None, # num_classes
#                                  None # processor_type
#                              ])


class HParams:
    def __init__(self,
                 # Does the epoch size really count as hparam? If not, maybe
                 # rename class and bring the processor type back in.
                 # Alternatively, move examples_per_epoch out to model_fn input.
                 examples_per_epoch=None,
                 examples_per_decay=None,
                 weight_decay=DEFAULT_WEIGHT_DECAY,
                 decay_rate=DEFAULT_DECAY_RATE,
                 skip_host_call=DEFAULT_SKIP_HOST):
        self.examples_per_epoch = examples_per_epoch
        self.examples_per_decay = examples_per_decay
        # Default to epoch size.
        if not self.examples_per_decay:
            self.examples_per_decay = examples_per_epoch
        self.weight_decay = weight_decay
        self.decay_rate = decay_rate
        self.skip_host_call = skip_host_call


class ModelFnFactory:
    """Creates a model_fn and initialized a EstimatorFactory.

    ModelFnFactory and EstimatorFactory simply wrap the individual functions
    present in this module in such a way that (I think) the usage is more
    obvious that using the individual functions.

    The construction of an estimator and the creation of a model_fn rely on
    many of the same parameters, such as model_dr and iterations_per_loop. This
    ModelFactory helps reduce the work required to pass these values around
    and to reduce the chance that these settings are not kept consistent when
    creating the model_fn and estimator.

    In addition, the ModelFactory & EstimatorFactory make it clear what
    hyper-parameters are supported by the model_fn and estimator. It also binds
    these values to the model_fn (i.e. wraps the model_fn) instead of sending
    them via the params dict. The params dict somewhat obscures the use of
    model_fn parameters, and this thus avoided where possible. 'batch_size' is
    one parameter that must be received through the params dict, as it varies
    depending on whether training, evaluation or prediction is being carried
    out.

    An Estimator's input_fn() is also passed a params dict, so it too might be
    a candidate for being included in this factory style setup. However, from
    the few input functions that I have seen, they don't seem to require many
    parameters that are also required of the model_fn or Estimator constructor.
    Thus, it seems best to create it separately.
    """

    def __init__(self, hparams: HParams):
        self.hparams = hparams
        self._model_fn = None

    def create_fn(self, keras_model_fn):
        model_fn = create_model_fn(keras_model_fn, self.hparams)
        est_factory = EstimatorFactory(model_fn, self.hparams)
        return model_fn, est_factory


class EstimatorFactory:
    """Creates an estimator."""
    def __init__(self, model_fn, params: HParams):
        self._model_fn = model_fn
        self._params = params

    def create_tpu_estimator(self, gcloud_settings, model_dir, train_batch_size,
                             eval_batch_size, iterations_per_loop):
        est = create_tpu_estimator(
            gcloud_settings, model_dir, self._model_fn,
            train_batch_size, eval_batch_size, iterations_per_loop)
        return est

    def create_cpu_estimator(self, model_dir):
        est = create_cpu_estimator(model_dir, self._model_fn)
        return est


def get_cluster_resolver(gcloud_settings):
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        # In the future, the tpu parameter might support lists.
        tpu=gcloud_settings.tpu_name,
        zone=gcloud_settings.tpu_zone,
        project=gcloud_settings.project_name)
    return tpu_cluster_resolver


# EfficientNet sets 'steps_per_epoch' in the estimator's params. This is an
# alternative to sending it to the model creation method. Worth considering
# which is better.
def create_tpu_estimator(gcloud_settings, model_dir, model_fn, train_batch_size,
                         eval_batch_size,
                         iterations_per_loop=ITERATIONS_PER_LOOP):
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
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        # What does this do? Disabling at it's not in efficientnet/main.py,
        # which is my current reference.
        # session_config=tf.ConfigProto(allow_soft_placement=True,
        #                              log_device_placement=True),
        # efficientnet/main.py also has this next setting, which I don't
        # understand. Disabling until I know why it is used.
        #session_config=tf.ConfigProto(
        #    graph_options=tf.GraphOptions(
        #        rewrite_options=rewriter_config_pb2.RewriterConfig(
        #            disable_meta_optimizer=True))),
        # Another, undocumented, setting used by efficient net.
        # log_step_count_steps=FLAGS.log_step_count_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            # The number of train steps running in TPU system before returning
            # to CPU host for each `Session.run`. This means that global step is
            # increased `iterations_per_loop` times in one `Session.run`. It is
            # recommended to be set as number of global steps between each
            # checkpoint.
            iterations_per_loop=iterations_per_loop,
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
            # This next option is used by efficientnet. Again, needs more
            # research to know if it should be used.
            # per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
            #     .PER_HOST_V2))  # pylint: disable=line-too-long
        )
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        # A function that  returns an EstimatorSpec or TPUEstimatorSpec.
        model_fn=model_fn,
        use_tpu=True,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        # model_dir=None, # Obtained from runConfig, so not needed here.
        # predict_batch_size=FLAGS.batch_size, possibly needed. Put if I use
        # params={'key':'value'}. Optional.
        params={},
        # Export a graph supporting PREDICT to be run on a TPU.
        export_to_tpu=True
        # batch_axis: not sure how to use this.
    )
    return estimator


# TODO: this should use the same initialize options as the tpu
# (except for use_tpu, I thinks that's all).
def create_cpu_estimator(model_dir, model_fn):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir)
    return estimator


def create_model_fn(keras_model_fn, processor_type, hparams=None):
    """Bind the parameters of _model_fn that are not part of the Estimator's
    model_fn signature.

    This way of creating the model_fn means we don't need to use to pass
    parameters through the estimator and take them via the params parameter.
    That mechanism seems flaky and seems to have poor encapsulation.

    Args:
        keras_model_fn: a function returning a Keras Model. The signature can
            one of the following two sigatures:
                <function_name>(): keras.models.Model
                <function_name>(input_tensor, is_training): keras.layers.Layer

         It seems that placeholder nodes aren't completely incompatible with TPUs,
         but it's not clear where they can be placed:
         https://git.codingcafe.org/Mirrors/tensorflow/tensorflow/commit/620c8383123519fcf4d987efb9776d861901ccfa

         An alternative would be to pass the input layer as a parameter into the
         keras_model_fn, however, this makes it harder to test model functions
         as they are no-longer standalone.
    """
    if not hparams:
        hparams = HParams()
    fn = functools.partial(_model_fn,
                           keras_model_fn=keras_model_fn,
                           processor_type=processor_type,
                           hparams=hparams)
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
# Update:
# So, sadly, we need to use the params['batch_size'] variable. I dislike how
# the parameter is passed within a dictionary; it would be nice if it was a
# normal named parameter. Anyway, the reason I need to use it is that the
# I wish to use the batch_size, but this changes depending on whether the
# estimator is evaluating or training, so it cannot be hard-coded in the model.
def _model_fn(features, labels, mode, params, config,
              # The following must be bound/wrapped in order for the function to
              # act as a model_fn usable by an estimator:
              keras_model_fn,
              processor_type,
              hparams):
    if processor_type == ProcessorType.TPU:
        # Only a TPUEstimator populates the 'batch_size' parameter.
        batch_size = params['batch_size']
        # Only a TPUEstimator has a tpu_config.
        iterations_per_loop = config.tpu_config.iterations_per_loop
    else:
        # Optionally, we could get the batch_size from the first dimension of
        # the features, but I'm not sure if batch_size is always present.
        batch_size = features.get_shape()[0]
        # batch_size = None
        # When not running on a TPU, this variable is not used. It is only
        # to be used in the TPU's host call.
        iterations_per_loop = None
    model_dir = config.model_dir
    # Note: EfficientNet does some transposing here. I wonder why it is done
    # in the model as opposed to the input function?
    image = features
    # Labels should be scalar values (not one-hot encoded).
    tf.ensure_shape(labels, shape=(None,))
    # Note: EfficientNet sets this globally. Is this the same thing as the
    # traiding-mode option below?
    # This is essential, if using a keras-derived model.
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    tf.keras.backend.set_learning_phase(is_training)
    # EfficientNet has an option (defaulting to false) to use bfloat32 for
    # training. Typically bfloat32 is used for eval, after model is fully
    # trained, so I'm not quite sure when this option would be used.
    # if params['use_bfloat16']:
    #     with tf.contrib.tpu.bfloat16_scope():
    #        logits = tf.cast(build_model(), tf.float32)
    # Experiment: support two keras_model_fn signatures.
    keras_model_fn_sig = inspect.signature(keras_model_fn)
    if len(keras_model_fn_sig.parameters) == 0:
        logit_outputs = keras_model_fn()(image, training=is_training)
    elif len(keras_model_fn_sig.parameters) == 2:
        logit_outputs = keras_model_fn(image, is_training)
    else:
        raise Exception('Unsupported keras_model_fn() signature encountered.')
    assert features.get_shape()[0] == batch_size
    if processor_type == ProcessorType.TPU:
        # We know the batch size and can make this assertion:
        assert logit_outputs.get_shape()[0] == batch_size
    num_classes = logit_outputs.get_shape()[1]

    loss_op = create_loss_op(logit_outputs, labels, num_classes,
                             hparams.weight_decay)
    host_call = None
    # Design choice. Create a TPUEstimator in if-else or outside the if-else?
    # mobilenet takes the first approach, and EfficientNet takes the second.
    # I guess it changes the graph. I wonder if there are performance
    # implications.
    if mode == tf.estimator.ModeKeys.TRAIN:
        estimator_spec = _train(
            loss_op,
            batch_size,
            model_dir,
            iterations_per_loop,
            processor_type,
            hparams)
    elif mode == tf.estimator.ModeKeys.EVAL:
        estimator_spec = _eval(labels, logit_outputs, loss_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        estimator_spec = _predict(logit_outputs)
    else:
        raise Exception('unexpected mode: {}'.format(mode))
    # I'm not 100% sure if this is needed. If not, then what is the point of the
    # as_estimator_spec() function?
    if processor_type != ProcessorType.TPU:
        estimator_spec = estimator_spec.as_estimator_spec()
    return estimator_spec


# metric_fn copied from efficientnet/main.py then edited.
def metric_fn(labels, logits):
    """Evaluation metric function. Evaluates accuracy.

    This function is executed on the CPU and should not directly reference
    any Tensors in the rest of the `model_fn`. To pass Tensors from the model
    to the `metric_fn`, provide as part of the `eval_metrics`. See
    https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
    for more information.

    Arguments should match the list of `Tensor` objects passed as the second
    element in the tuple passed to `eval_metrics`.

    Args:
      labels: `Tensor` with shape `[batch]`.
      logits: `Tensor` with shape `[batch, num_classes]`.

    Returns:
      A dict of the metrics to return from evaluation.
    """
    predictions = tf.argmax(logits, axis=1)
    top_1_accuracy = tf.metrics.accuracy(labels, predictions)
    in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, k=5), tf.float32)
    top_5_accuracy = tf.metrics.mean(in_top_5)
    return {TOP_1_ACCURACY_KEY: top_1_accuracy,
            TOP_5_ACCURACY_KEY: top_5_accuracy}


def _predict(logit_outputs):
    """Creates the graph portion for the predict mode."""
    predictions = {
        'classes': tf.argmax(logit_outputs, axis=1),
        'probabilities': tf.nn.softmax(logit_outputs, name='softmax_tensor')
    }
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })
    return estimator_spec


def _eval(labels, logit_outputs, loss_op):
    # From the TPUEstimatorSpec source:
    #     For evaluation, `eval_metrics `is a tuple of `metric_fn` and
    #     `tensors`, where `metric_fn` runs on CPU to generate metrics and
    #     `tensors` represents the `Tensor`s transferred from TPU system to
    #     CPU host and passed to `metric_fn`.
    estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL, loss=loss_op,
        eval_metrics=(metric_fn, [labels, logit_outputs]))
    return estimator_spec


def _train(loss_op,
           batch_size,
           model_dir,
           iterations_per_loop,
           processor_type,
           hparams):
    global_step = tf.train.get_global_step()
    # TODO: need to take in examples_per_epoch!
    if hparams.examples_per_epoch:
        current_epoch = (tf.cast(global_step, tf.float32) /
                         hparams.examples_per_epoch)
    else:
        # If no examples_per_epoch setting, act as if there is a single epoch.
        current_epoch = 1
    train_op, learning_rate = create_train_op(
        loss_op, processor_type, batch_size, hparams.examples_per_decay,
        hparams.decay_rate)

    # Taken from resnet_main.py in tensorflow_tpu repo.
    host_call = None
    # Add the host call if using TPUs and the option is enabled. CPU/GPU runs
    # don't need to use a host call.
    if processor_type == ProcessorType.TPU and not hparams.skip_host_call:
         host_call = _host_call_and_args(
             global_step, learning_rate, current_epoch, iterations_per_loop,
             model_dir)
    estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
        tf.estimator.ModeKeys.TRAIN, loss=loss_op, train_op=train_op,
        host_call=host_call)
    return estimator_spec


def create_train_op(loss,
                    processor_type,
                    batch_size,
                    examples_per_decay,
                    decay_rate):
    """Create and return the train operation. Called from model_fn.

    How often the learning rate decays is determined as:

        steps_per_decay = examples_per_decay // batch_size

    Regarding the args, we could take in an options tuple here, but there aren't
    that many parameters, and it's nice to be explicit about the dependencies.

    Returns: TODO
    """
    # TODO: decide what learning rate to use.
    global_step = tf.train.get_global_step()
    if examples_per_decay:
        steps_per_decay = examples_per_decay // batch_size
        learning_rate = tf.train.exponential_decay(
            learning_rate_base,
            tf.cast(global_step, tf.float32), # Don't think the cast is needed.
            decay_steps=steps_per_decay,
            decay_rate=decay_rate)
    else:
        learning_rate = const_learning_rate
    # TPUEstimator doesn't support summaries!
    # tf.summary.scalar('learning rate', learning_rate)
    # FIXME 9: RMSPropOptimizer doesn't seem to be working.
    # MobileNetv2 paper uses RMSPropOptimizer with decay and momentum as 0.9.
    # RMSProp doesn't seem to be working for me on CPU or TPU.
    #optimizer = tf.train.RMSPropOptimizer(learning_rate_base, decay=0.90, momentum=0.9)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # FIXME: Learning rate not working
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer()
    if processor_type == ProcessorType.TPU:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    # TODO: is this the correct value for the step argument?

    # This alteration was taken from resnet_main.py from tensorflow_tpu.
    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation. It's also present in EfficientNet.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
    return train_op, learning_rate


def _host_call_and_args(global_step, learning_rate, current_epoch,
                        iterations_per_loop, model_dir):
    def host_call_fn(global_step, learning_rate, current_epoch):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          global_step: `Tensor with shape `[batch]` for the global_step
          learning_rate: `Tensor` with shape `[batch]` for the learning_rate.
          current_epoch: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        global_step = global_step[0]
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with tf.contrib.summary.create_file_writer(
                model_dir,
                max_queue=iterations_per_loop).as_default():
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(
                    'learning_rate', learning_rate[0], step=global_step)
                tf.contrib.summary.scalar(
                    'current_epoch', current_epoch[0], step=global_step)
                return tf.contrib.summary.all_summary_ops()

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    global_step_t = tf.reshape(global_step, [1])
    learning_rate_t = tf.reshape(learning_rate, [1])
    current_epoch_t = tf.reshape(current_epoch, [1])

    host_call_and_args = (host_call_fn,
                          [global_step_t, learning_rate_t, current_epoch_t])
    return host_call_and_args


def create_loss_op(logits, labels, num_classes, weight_decay):
    one_hot_labels = tf.one_hot(labels, num_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels,
        # What does this do?
        label_smoothing=0.1)
    weight_sum = tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name])
    # Interesting note: The following line was the source of a bug which was
    # time consuming to track down. The '+' was accidentally a '*'. The result
    # being estimation accuracy reduced to random at 10% for cifar10.
    # TODO: testing without weight decay.
    loss = cross_entropy + weight_decay * weight_sum
    return loss


# TODO: mention source of code below (efficientnet/main.py:main()).
# TODO: EfficientNet has a if-case handing for just an 'eval' mode which goes
# through all existing checkpoints and runs an evaluation. This may be useful
# at some later date.

# This method isn't needed. estimator.train() is sufficient.
# def train(estimator, model_fn, train_input_fn, train_steps):
#    # TODO: EfficientNet uses async checkpointing; I don't know what is
#    #  does. It might be useful. Worth investigating.
#    estimator.train(input_fn=train_input_fn, max_steps=train_steps)


# TODO, it would be nice to wrop a lot of this context into a class, as much is
# known about the train_and_eval context when building the estimator or
# model function above.
def train_and_eval(estimator, train_input_fn, eval_input_fn, train_steps,
                   steps_per_epoch, num_eval_images, steps_between_eval,
                   eval_batch_size):
    # TPUEstimator has a public property, model_dir. Let's use that instead of
    # it needing to be passed in.
    model_dir = estimator.model_dir
    current_step = estimator._load_global_step_from_checkpoint_dir(model_dir)
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', train_steps, train_steps / steps_per_epoch, current_step)

    start_timestamp = time.time()  # This time will include compilation time

    # Manually break the training periodically. I think one purpose of this is
    # to allign the checkpointing and evaluation so that the evaluations map to
    # a specific checkpoint. However, shouldn't TPUEstimator do that
    # automatically with train_and_evaluate? Some other opinions about the
    # matter:
    # https://stackoverflow.com/questions/50596369/train-and-evaluate-batch-size-with-tpu-on-gcmle/50629758
    while current_step < train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + steps_between_eval, train_steps)
        estimator.train(input_fn=train_input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info(
            'Finished training up to step %d. Elapsed seconds %d.',
            next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate.')
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=num_eval_images // eval_batch_size)
        tf.logging.info('Eval results at step %d: %s',
                        next_checkpoint, eval_results)
        ckpt = tf.train.latest_checkpoint(model_dir)
        efficientnet.utils.archive_ckpt(eval_results,
                           eval_results[TOP_1_ACCURACY_KEY], ckpt)

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
        train_steps, elapsed_time)
    # TODO: haven't addressed exporting yet.
    # if FLAGS.export_dir:
    #     export(est, FLAGS.export_dir, input_image_size)