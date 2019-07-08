import tensorflow as tf
import pytz
import datetime
import functools
import logging

# For loading mnist data.
import micronet.mnist.dataset as mnist_dataset

# tf.keras.layers replaces tf.layers (this was mentioned in the class
# documentation for tf.layers.layer).
import tensorflow.keras
import tensorflow.keras.layers

tpu_name = 'kdoran1'
tpu_zone = 'us-central1-f'
gcloud_project_name = 'micronet-kdoran'
gs_bucket_url = 'gs://micronet_bucket1'
data_dir = gs_bucket_url + '/mnist'
# Model dir
model_root_dir = gs_bucket_url + '/models/mnist'
mnist_data_dir = gs_bucket_url + '/mnist'
now = datetime.datetime.now(pytz.timezone('Japan'))
timestamp_suffix = now.strftime('%Y%m%dT%H%M%S')
#model_dir = '{}/kdoran_{}'.format(model_root_dir, timestamp_suffix)
model_dir = '{}/kdoran_{}'.format(model_root_dir, 'model1')
print(model_dir)
# num_shards = 8. Deprecated. It was a parameter to TPUConfig().
training_steps = 5000
# Number of iterations to run on the TPU workers before returning control to the
# master (not sure if the terminology is correct here).
iterations_between_model_update = 50
# Mini-batch size for training (global, not per shard).
batch_size = 1024
learning_rate_base = 0.05

# Example.
# With training_steps = 1000, iterations_between_model_update = 50 and
# batch_size = 1024:
#     - there would be 1000 * 1024 = 1024,000 input images used for training.
#     - training would be broken into 1024 separate update steps (when the
#           model gets updated).
#     - Every 1024/8 = 128 steps, control would return to the master from the
#           TPUs. Note: what updates happen here, between the batches?


def _train_input_fn(mnist_data_dir, batch_size, params):
    """train_input_fn defines the input pipeline used for training."""
    del params
    # Separated a 1-liner into individual statements for commenting.
    # A DatasetV2
    ds = mnist_dataset.train(mnist_data_dir)
    # What exactly does the caching involve?
    ds = ds.cache()
    # Repeat the dataset `count` times. If `count` is None, repeat indefinitely.
    ds = ds.repeat()
    # Randomly shuffle the elements of the dataset. This dataset fills a buffer
    # with `buffer_size` elements, then randomly samples elements from this
    # buffer, replacing the selected elements with new elements. For perfect
    # shuffling, a buffer size greater than or equal to the full size of the
    # dataset is required.
    ds = ds.shuffle(buffer_size=50000)
    # Combine consecutive elements into batches. The tensors will now have an
    # extra outer dimension. `drop_remainder` set to True will cause elements
    # to be discarded so as to only have batches with size = `batch_size`.
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


# Tutorials/examples show the model_dir being sent to the input_fun through
# the estimator. However, this requires using the 'params' dict parameter of
# TPUEstimator's __init__ function, and then manually extracting it by name
# in the train_input_fn. That doesn't seem clean as two places of my code must
# use matching dictionary keys without there being an interface provider/user
# relationship of some sort.
# Instead, I'll just bind the model_dir parameter to the function.
train_input_fn = functools.partial(_train_input_fn, mnist_data_dir, batch_size)


#def create_model(input, data_format):
def create_model(data_format):
    model = tf.keras.Sequential()
    # Reshape input. channels_first better for GPU, channels_last for CPU.
    if data_format == 'channels_first':
        reshape_to = [1, 28, 28]
    elif data_format == 'channels_last':
        reshape_to = [28, 28, 1]
    else:
        raise Exception('Invalid data_format: {}'.format(data_format))
    model.add(tf.keras.layers.Reshape(target_shape=reshape_to,
                                      input_shape=(28 * 28,)))
    # Layers
    # 1a. Convolve 2D.
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format=data_format,
        activation='relu',
        padding='same',
        use_bias=True
    )) # (batch, 28, 28, 32)
    # 1b. Convolve 2D.
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format=data_format,
        activation='relu',
        padding='same',
        use_bias=True
    )) # (batch, 28, 28, 32)
    # 1c. Pool.
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2,2),
        padding='valid', # no padding.
    )) # (batch, 14, 14, 32)
    model.add(tf.keras.layers.Dropout(0.25))

    # 2a. Convolve 2D.
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format=data_format,
        activation='relu',
        padding='same',
        use_bias=True
    )) # (batch, 14, 14, 64)
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format=data_format,
        activation='relu',
        padding='same',
        use_bias=True
    )) # (batch, 14, 14, 64)
    # 2c. Pool.
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2,2),
        padding='valid', # no padding.
    )) # (batch, 7, 7, 64)
    model.add(tf.keras.layers.Dropout(0.25))

    # 3a. Flatten
    model.add(tf.keras.layers.Flatten())
    # 3b.
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model
    # The "layer call" action is like drawing an arrow from the inputs to the
    # layer that is called.
    #outputs = model(input)
    #return outputs


def create_model2(data_format):
  """Model to recognize digits in the MNIST dataset.

  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

  But uses the tf.keras API.

  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats

  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])

def metric_fn(labels, logits):
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}

def model_fn(features, labels, mode, params):
    del params
    image = features
    print("Mode is: {}".format(mode))
    model = create_model2(data_format='channels_first')
    # PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: what is going on in this mode.
        outputs = model(image, training=False)
        predictions = {
            'class_ids': tf.argmax(outputs, axis=1),
            'probabilities': tf.nn.softmax(outputs)
        }
        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)
    # EVAL
    elif mode == tf.estimator.ModeKeys.EVAL:
        outputs = model(image, training=False)
        # Why is loss here?
        loss = tf.losses.sparse_softmax_cross_entropy(logits=outputs, labels=labels)
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, outputs]))
    # TRAIN
    elif mode == tf.estimator.ModeKeys.TRAIN:
        outputs = model(image, training=True)
        # Loss
        # loss is a tensor. labels and logits must be compatible dimensions.
        loss = tf.losses.sparse_softmax_cross_entropy(logits=outputs, labels=labels)

        # Optimizer
        learning_rate = tf.train.exponential_decay(
            learning_rate_base,
            tf.train.get_global_step(),
            decay_steps=100000,
            decay_rate=0.96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # For use with TPUs, the optimizer must be wrapped like so:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(loss, tf.train.get_global_step())
        # There is also a KerasCrossShardOptimizer...what is that?

        # Estimator Spec
        # Estimator spec is a named tuple.
        # The Estimator class (in repo tensorflow_estimator) has code as follows:
        # while not mon_sess.should_stop():
        #     _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
        # So, the train_op and the loss are simply passed to Session.run(). This
        # makes them the nodes in the graph to be evaluated; their outputs will
        # be the elements of a returned tuple from Session.run().
        # Source: https://github.com/tensorflow/estimator/blob/17ec828045035073ce2ea9ad4961f795a3c74e05/tensorflow_estimator/python/estimator/estimator.py#L1360
        estimator_spec = tf.estimator.EstimatorSpec(mode, loss=loss,
                                                    train_op=train_op)
        return estimator_spec


def main():
    logging.getLogger('tensorflow').setLevel(logging.INFO)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        # In the future, the tpu parameter might support lists.
        tpu=tpu_name,
        zone=tpu_zone,
        project=gcloud_project_name)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        # Directory to save model parameters, graph etc. Also used as a source
        # directory when loading checkpoints.
        model_dir=model_dir,
        keep_checkpoint_max=0,
        session_config=tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            # The number of train steps running in TPU system before returning
            # to CPU host for each `Session.run`. This means that global step is
            # increased `iterations_per_loop` times in one `Session.run`. It is
            # recommended to be set as number of global steps between each
            # checkpoint.
            iterations_per_loop=iterations_between_model_update,
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
        model_fn=model_fn,  # TODO
        use_tpu=True,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        model_dir=None, # disable checkpoints.
        # predict_batch_size=FLAGS.batch_size, possibly needed. Put if I use
        # params={'key':'value'}. Optional. Don't need it yet.
        # prediction on the CPU, then we don't need TPU prediction.
        # export_to_cpu=True, I might need this option for prediction.
        # batch_axis: not sure how to use this.
    )

    # Train the model.
    estimator.train(input_fn=train_input_fn, max_steps=training_steps)



if __name__ == '__main__':
    main()
