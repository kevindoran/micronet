import efficientnet.efficientnet_builder as net_builder
import tensorflow as tf
import micronet.estimator
import micronet.gcloud as gcloud
import micronet.models
import micronet.dataset.imagenet as imagenet_ds
import os

TEST_CLASS = 145 # King penguin.

def custom_model(image_inputs, is_training):
    image_inputs = micronet.models.normalize_image(image_inputs,
                                                   net_builder.MEAN_RGB,
                                                   net_builder.STDDEV_RGB)
    features, endpoints = net_builder.build_model_base(
        image_inputs, model_name='efficientnet-b0', training=False)
    assert 0 == len(tf.trainable_variables())
    # The 6th layer outputs the second resolution reduction.
    # See the Efficientnet readme for use of the model builder API. See the
    # paper for the layer breakdown. https://arxiv.org/pdf/1905.11946.pdf
    layer_6_outputs = endpoints['reduction_2']
    binary_logit = tf.layers.Dense(units=1)(layer_6_outputs)
    # Calculate the number of features out at layer 6.
    # assert 0 == len(tf.trainable_variables())
    return binary_logit


def custom_loss_op(logits, labels, num_classes, weight_decay):
    is_match = tf.math.equal(145, labels)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=is_match,
        logits=logits)
    weight_sum = tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name])
    # Interesting note: The following line was the source of a bug which was
    # time consuming to track down. The '+' was accidentally a '*'. The result
    # being estimation accuracy reduced to random at 10% for cifar10.
    # TODO: testing without weight decay.
    loss = cross_entropy + weight_decay * weight_sum
    return loss

def main():
    gcloud_settings = gcloud.load_settings()
    model_dir = 'gs://micronet_bucket1/models/shortcutnet1'
    image_size = 224
    images_per_epoch = 1.2 * 1000 * 1000 # is this correct?
    train_images = images_per_epoch * 2
    train_batch_size = 128
    eval_batch_size = 128
    train_steps = train_images // train_batch_size
    num_eval_images = 5 * 2**10
    steps_between_eval = 10

    est = micronet.estimator.create_tpu_estimator(
              gcloud_settings=gcloud_settings,
              model_dir=model_dir,
              model_fn=custom_model,
              train_batch_size=train_batch_size,
              eval_batch_size=eval_batch_size)
    train_input_fn = imagenet_ds.create_train_input(
        image_size=image_size,
        num_parallel_calls=os.cpu_count(),
        for_tpu=True, autoaugment=False)
    eval_input_fn = imagenet_ds.create_train_input(
        image_size=image_size,
        num_parallel_calls=os.cpu_count(),
        for_tpu=True, autoaugment=False)

    micronet.estimator.train_and_eval(
        est,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_steps,
        steps_per_epoch=images_per_epoch//train_batch_size,
        num_eval_images=num_eval_images,
        steps_between_eval=steps_between_eval,
        eval_batch_size=eval_batch_size)


if __name__ == '__main__':
    main()