import pytest
import micronet.cifar.dataset as cifar_ds
import tensorflow as tf
import numpy as np


# FIXME 2: It would be nice to be able to count the elements in a reasonably
# small dataset like cifar. The below doesn't seem to work (or takes too long).
def record_count(dataset):
    def init_fn(key):
        return 0

    def reduce_fn(value, state):
        return value + 1

    def finalize_fn(state):
        return state

    def key_fn(shape, type):
        return np.int64(0)

    reducer = tf.data.experimental.Reducer(init_fn, reduce_fn, finalize_fn)
    dataset = dataset.apply(tf.data.experimental.group_by_reducer(key_fn, reducer))
    count = tf.data.experimental.get_single_element(dataset).eval()
    return count


def _check_dataset(ds):
    ds_iter = ds.make_one_shot_iterator()
    # The records should be a (img, label) tuple.
    (img, label) = ds_iter.get_next()
    # FIXME 1: (switch to uint8)
    # assert img.dtype == tf.uint8
    assert img.dtype == tf.float32
    assert img.shape.rank == 4 # batch x W x H x D
    # Note: the batch dimension doesn't have a hard-coded size so as to allow
    # the network be flexible enough to use any batch sizes.
    assert [None, cifar_ds.IMAGE_SIZE, cifar_ds.IMAGE_SIZE, 3] \
        == img.shape.as_list()
    assert label.dtype == tf.uint8
    # FIXME 2
    # assert record_count(ds) == cifar_ds.TRAIN_COUNT


def test_train_dataset():
    batch_size = 100
    _check_dataset(cifar_ds.train_dataset(batch_size, augment=False))


def test_eval_dataset():
    batch_size = 100
    _check_dataset(cifar_ds.eval_dataset(batch_size, augment=False))


def test_test_dataset():
    batch_size = 100
    _check_dataset(cifar_ds.test_dataset(batch_size, augment=False))
