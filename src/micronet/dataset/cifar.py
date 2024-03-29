import tensorflow as tf
import tensorflow_datasets as tfds

# TODO: for speed and to make the dataset splits deterministic, I seems like
#       this will need to migrate to being a manual task of loading the data
#       from a serialized protobuff.

# TODO: data augmentation

# FIXME 21: support both cloud and local storage.
_CLOUD_DATA_DIR = 'gs://micronet_bucket1/cifar100'

# TODO: this whole file could be cleaned up. The API could be improved too.

# Splits.
# The Cifar100 dataset in tensorflow-datasets only has two predefined splits,
# train and test, so we must manually make a eval split.
TRAIN_COUNT = 45000
EVAL_COUNT = 5000
TEST_COUNT = 10000
# Train and eval split percentages.
# Percentage of total Cifar "train" set, which is 50,000.
TRAIN_PERCENTAGE = 90
EVAL_PERCENTAGE = 10
train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:TRAIN_PERCENTAGE])
eval_split = tfds.Split.TRAIN.subsplit(tfds.percent[:EVAL_PERCENTAGE])

DEFAULT_IMAGE_SIZE = 24
MAX_IMAGE_SIZE = 32
COLOR_CHANNELS = 3 # RGB
DEFAULT_DATA_SHAPE = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, COLOR_CHANNELS)
# FIXME 1: (switch to uint8)
DTYPE = tf.float32
CLASSES = 100


def train_dataset(cloud_storage=False):
    # Note: is this if-else compatible with TPU's input pipeline requirements?
    # I saw a mention that some types of branching are not supported.
    if cloud_storage:
        ds = tfds.load(name='cifar100', split=train_split, download=False,
                       data_dir=_CLOUD_DATA_DIR)
    else:
        ds = tfds.load(name='cifar100', split=train_split)
    return ds


def eval_dataset(cloud_storage=False):
    if cloud_storage:
        ds = tfds.load(name='cifar100', split=eval_split, download=False,
                       data_dir=_CLOUD_DATA_DIR)
    else:
        ds = tfds.load(name='cifar100', split=eval_split)
    return ds


def test_dataset(cloud_storage=False):
    if cloud_storage:
        ds = tfds.load(name='cifar100', split=tfds.Split.TEST, download=False,
                       data_dir=_CLOUD_DATA_DIR)
    else:
        ds = tfds.load(name='cifar100', split=tfds.Split.TEST)
    return ds


# Copied from /tensorflow_models/tutorials/image/cifar10/cifar10_input.py
def preprocess_fn(augment, crop_to):
    """Preprocess the images with optional augmentation.

    Args:
        dataset: dataset to process.
        batch_size: combine consecutive elements into batches. The resulting
            tensor will have an extra dimension with each entry being a batch.
        augment(bool): whether to augment the dataset (apply mutations).

    Returns:
        An infinite (repeating) batched dataset of tuples (img, label).
    """
    # FIXME 29: augmentation breaks test_cifar_dataset.py, but I'm not sure why.
    augment = False
    def map_fn(record):
        """
        Returns:
            (img, label) tuple.
        """
        img = record['image']
        label = record['label']
        # FIXME 1: (switch to uint8)
        #img = tf.cast(img, tf.uint8)
        # FIXME 1: Why can't label's be unit8?
        #label = tf.cast(label, tf.uint8)
        # () is the shape for a scalar.
        tf.ensure_shape(label, shape=())
        if augment:
            # Randomly crop a [height, width] section of the image.
            img = tf.random_crop(img, [crop_to, crop_to, 3])
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
            img = tf.image.resize_image_with_crop_or_pad(img, crop_to, crop_to)
        # Subtract off the mean and divide by the variance of the pixels.
        # FIXME 1: (switch to uint8)
        # FIXME 6: insure consistent standardization.
        img = tf.image.per_image_standardization(img)
        return img, label

    return map_fn
