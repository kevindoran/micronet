import tensorflow as tf
import tensorflow_datasets as tfds

# TODO: for speed and to make the dataset splits deterministic, I seems like
# this will need to migrate to being a manual task of loading the data from
# a serialized protobuff.
# The Cifar100 dataset in tensorflow-datasets only has two predefined splits,
# train and test, so we must manually make a eval split.
# TODO: data augmentation
TRAIN_COUNT = 45000
EVAL_COUNT = 5000
TEST_COUNT = 10000

# Train and eval split percentages.
# Percentage of total Cifar "train" set, which is 50,000.
TRAIN_PERCENTAGE = 90
EVAL_PERCENTAGE = 10

train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:TRAIN_PERCENTAGE])
eval_split = tfds.Split.TRAIN.subsplit(tfds.percent[:EVAL_PERCENTAGE])

IMAGE_SIZE = 24


def train_dataset(batch_size, augment):
    ds = tfds.load(name='cifar100', split=train_split)
    ds = preprocess(ds, batch_size, augment)
    return ds


def eval_dataset(batch_size, augment):
    ds = tfds.load(name='cifar100', split=eval_split)
    ds = preprocess(ds, batch_size, augment)
    return ds


def test_dataset(batch_size, augment):
    ds = tfds.load(name='cifar100', split=tfds.Split.TEST)
    ds = preprocess(ds, batch_size, augment)
    return ds


# Copied from /tensorflow_models/tutorials/image/cifar10/cifar10_input.py
def preprocess(dataset, batch_size, augment):
    """Preprocess the images with optional augmentation.

    Args:
        dataset: dataset to process.
        batch_size: combine consecutive elements into batches. The resulting
            tensor will have an extra dimension with each entry being a batch.
        augment(bool): whether to augment the dataset (apply mutations).

    Returns:
        An infinite (repeating) batched dataset of tuples (img, label).
    """
    def map_fn(record):
        """
        Returns:
            (img, label) tuple.
        """
        img = record['image']
        label = record['label']
        # FIXME: 1 (switch to uint8)
        img = tf.cast(img, tf.uint8)
        label = tf.cast(label, tf.uint8)
        if augment:
            # Randomly crop a [height, width] section of the image.
            img = tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
            # Randomly flip the image horizontally.
            img = tf.image.random_flip_left_right(img)
            # Because these operations are not commutative, consider randomizing
            # the order their operation.
            # NOTE: since per_image_standardization zeros the mean and makes
            # the stddev unit, this likely has no effect see tensorflow#1458.
            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        else:  # Image processing for evaluation.
            # Crop the central [height, width] of the image.
            img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_SIZE,
                                                         IMAGE_SIZE)
        # Subtract off the mean and divide by the variance of the pixels.
        # FIXME: 1 (switch to uint8)
        img = tf.image.per_image_standardization(img)
        return img, label

    dataset = dataset.map(map_fn, num_parallel_calls=10) # num_parallel_calls=10, what does this do?
    # Dataset is small enough to be fully loaded on memory:
    dataset = dataset.prefetch(-1)
    dataset = dataset.repeat().batch(batch_size)
    return dataset
