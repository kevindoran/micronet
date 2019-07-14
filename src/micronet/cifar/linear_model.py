import tensorflow as tf
import micronet.cifar.dataset as cifar_ds
import micronet.estimator
import functools

learning_rate_base = 0.05


def metric_fn(labels, logits):
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}


def create_model():
    model = tf.keras.Sequential()
    # Insure input dimensions are correct.
    # Idea from: https://pgaleone.eu/tensorflow/2018/07/28/understanding-tensorflow-tensors-shape-static-dynamic/
    #input = tf.placeholder(cifar_ds.DTYPE,
    #    shape=(None, cifar_ds.IMAGE_SIZE, cifar_ds.IMAGE_SIZE, 3))
    # The linear model doesn't use any layers that are dependent on shape,
    # so we can ignore any reshaping and just flatten the data.

    # FIXME: why does this break?
    #model.add(tf.keras.layers.InputLayer(input_tensor=input))


    # This might work too, instead of the two steps above:
    #model.add(tf.keras.layers.Reshape(
    #    target_shape=(cifar_ds.IMAGE_SIZE * cifar_ds.IMAGE_SIZE * 3,),
    #    input_shape=(cifar_ds.IMAGE_SIZE, cifar_ds.IMAGE_SIZE, 3)))

    model.add(tf.keras.layers.Flatten(
        input_shape=(cifar_ds.IMAGE_SIZE, cifar_ds.IMAGE_SIZE, 3)
    ))
    model.add(tf.keras.layers.Dense(100,
                                    activation='softmax'))
    model.summary()
    return model


# TODO: is 'op' correct here?
def create_loss_op(logits, labels):
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
    return loss


def create_train_op(loss, processor_type):
    # TODO: decide what learning rate to use.
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        tf.train.get_global_step(),
        decay_steps=100000,
        decay_rate=0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    if processor_type == micronet.estimator.ProcessorType.TPU:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    # TODO: is this the correct value for the step argument?
    train_op = optimizer.minimize(loss, tf.train.get_global_step())
    return train_op


def create_model_fn(processor_type):
    """Bind the processor type parameter and return the resulting function.

    This way of creating the model_fn means we don't need to use to pass
    parameters through the estimator and take them via the params parameter.
    That mechanism seems flaky and seems to have poor encapsulation.
    """
    fn = functools.partial(model_fn, processor_type)
    return fn


# Interestingly, it looks like the params argument is optional, as long as it
# is also not passed to the estimator. So removing from here, as parameter.
# Original signature:
#     def model_fn(processor_type, features, labels, mode, params):
def model_fn(processor_type, features, labels, mode):
    image = features
    tf.ensure_shape(labels, shape=(None,))
    # Labels should be scalar values (not one-hot encoded).
    if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO: is it okay to have the create model within an if? Does it
        #       prevent some sort of model reuse that would otherwise happen?
        logit_outputs = create_model()(image, training=True)
        loss_op = create_loss_op(logit_outputs, labels)
        train_op = create_train_op(loss_op, processor_type)
        # FIXME X: how to return either TPU or non TPU estimator spec?
        estimator = tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss_op,
                                                    train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        # TODO: What does the training option do?
        logit_outputs = create_model()(image, training=False)
        loss_op = create_loss_op(logit_outputs, labels)
        # Does the eval_metrics need to be (metric_fn, [labels, outputs])?
        # FIXME X: how to return either TPU or non TPU estimator spec?
        #estimator = tf.estimator.EstimatorSpec(mode=mode, loss=loss_op,
        #                                       eval_metric_ops=(metric_fn,))
        # From the TPUEstimatorSpec source:
        #     For evaluation, `eval_metrics `is a tuple of `metric_fn` and
        #     `tensors`, where `metric_fn` runs on CPU to generate metrics and
        #     `tensors` represents the `Tensor`s transferred from TPU system to
        #     CPU host and passed to `metric_fn`.
        estimator = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=loss_op,
            eval_metrics=(metric_fn, [labels, logit_outputs]))
    elif mode == tf.estimator.ModeKeys.PREDICT:
        raise Exception('Unsupported.')
    else:
        raise Exception('unexpected mode: {}'.format(mode))
    if processor_type == micronet.estimator.ProcessorType.CPU:
        estimator = estimator.as_estimator_spec()
    return estimator
