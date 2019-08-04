import pytest
import tensorflow as tf
import micronet.dataset.imagenet_old as imagenet_ds
import test.util


def test_dataset():
    # Setup
    # Test a few examples.
    test_count = 20

    # Test
    # Use the test dataset, as the files are smaller. (390 vs 1250 records).
    ds = imagenet_ds.test_dataset().take(test_count)
    ds = ds.map(imagenet_ds.preprocess_fn(is_training=False))
    ds_iter = ds.make_one_shot_iterator()
    next_element = ds_iter.get_next()
    with tf.Session() as sess:
        for i in range(test_count):
            (img, label) = sess.run(next_element)
            # dtype is actually a numpy float32. Not sure why. I'll keep the
            # data type as tf.float32 for now, and just reverse the equality,
            # as tf.float32 supports comparison to numpy float32.
            # assert img.dtype == imagenet_ds.IMAGE_DATATYPE
            assert imagenet_ds.IMAGE_DATATYPE == img.dtype
            # TODO: img is numpy array. Why so?
            # assert img.shape.rank == 3  # W X H X D
            assert len(img.shape) == 3  # W X H X D
            # assert img.shape.as_list() == [imagenet_ds.DEFAULT_IMAGE_SIZE,
            assert img.shape == (imagenet_ds.DEFAULT_IMAGE_SIZE,
                                  imagenet_ds.DEFAULT_IMAGE_SIZE,
                                  imagenet_ds.CHANNEL_COUNT)
            # assert tuple(img.shape.as_list()) == imagenet_ds.DEFAULT_DATA_SHAPE
            assert img.shape == imagenet_ds.DEFAULT_DATA_SHAPE
            assert label.shape == ()
            assert imagenet_ds.CLASS_INDEX_BOUNDS_INC[0] \
                   <= label \
                   <= imagenet_ds.CLASS_INDEX_BOUNDS_INC[1]


def test_crop():
    # Setup
    # Test a few examples.
    test_count = 20
    crop_size = 200
    is_training = True

    # Test
    ds = imagenet_ds.test_dataset().take(test_count)
    preprocess_fn = imagenet_ds.preprocess_fn(is_training=True,
                                              crop_to=crop_size)
    ds = ds.map(preprocess_fn)
    ds_iter = ds.make_one_shot_iterator()
    next_element = ds_iter.get_next()
    with tf.Session() as sess:
        for i in range(test_count):
            (img, label) = sess.run(next_element)
            assert img.shape == (crop_size, crop_size,
                                 imagenet_ds.CHANNEL_COUNT)


# My PC can't handle this test; it runs out of memory.
@pytest.mark.tpu_only
def test_with_estimator(estimator_fn):
    # Setup
    batch_size = 128
    steps = 20
    keras_fn = test.util.test_keras_fn(imagenet_ds.DEFAULT_DATA_SHAPE,
                                       imagenet_ds.NUM_CLASSES)
    estimator = estimator_fn(keras_model_fn=keras_fn,
                             train_batch_size=batch_size,
                             eval_batch_size=batch_size)

    def input_fn(params):
        del params
        ds = imagenet_ds.train_dataset().map(
            imagenet_ds.preprocess_fn(is_training=True))
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds

    # Test
    # Run an evaluate and train. Don't worry about the results.
    estimator.evaluate(input_fn, steps=steps)
    estimator.train(input_fn, max_steps=steps)
