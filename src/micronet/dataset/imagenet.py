import tensorflow as tf

import tensorflow_models.research.slim.preprocessing.inception_preprocessing as \
    slim_preprocessing

# Data storage
# ------------
# The dataset was built using Tensorflow/model's build_imagenet_data.py:
# https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py
#
# The dataset consists of:
# Training data files, with ~1250 records each:
# data_dir/train-00000-of-01024
#      ...
# data_dir/train-01023-of-01024
#
# and validation files, with ~390 records each:
# data_dir/validation-00000-of-00128
#      ...
# data_dir/validation-00127-of-00128

_tf_record_dir = 'gs://micronet_bucket1/imageNet'
_num_orig_train_files = 1024
_examples_per_train_file = 1250
_num_orig_validation_files = 128

NUM_CLASSES = 1000
CLASS_START_INDEX = 1
CLASS_INDEX_BOUNDS_INC = (CLASS_START_INDEX, CLASS_START_INDEX + NUM_CLASSES-1)
# Data format
# -----------
# The Examples have the following fields (among others):
# image/encoded: string containing JPEG encoded image in RGB colorspace
# image/height: integer, image height in pixels
# image/width: integer, image width in pixels
# image/filename: string containing the basename of the image file
#           e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
# image/class/label: integer specifying the index in a classification layer.
#   The label ranges from [1, 1000] where 0 is not used.
# image/class/text: string specifying the human-readable version of the label
#   e.g. 'red fox, Vulpes vulpes'
# image/object/bbox/label: integer specifying the index in a classification
# layer. The label ranges from [1, 1000] where 0 is not used. Note this is
# always identical to the image label.
IMAGE_DATATYPE = tf.float32
DEFAULT_IMAGE_SIZE = 224
CHANNEL_COUNT = 3
DEFAULT_DATA_SHAPE = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, CHANNEL_COUNT)

# Train, validation and test splits.
# ----------------------------------
# In order to have a three way split (train, validation and test), the train
# data will be further split to create the validation set, while the validation
# data will be used as the test data.
#
# Aiming for 50,000 records from the training set to be repurposed as
# validation records, 50,000/1250 = 40 files will be used for validation.
# Instead of renaming the files, they mapping will be done here.

_validation_count = 50000
_num_validation_files = int(_validation_count / _examples_per_train_file)
assert _num_validation_files == 40
_num_train_files = _num_orig_train_files - _num_validation_files
_last_orig_train_file = _num_orig_train_files - 1
first_last_train_file = (0, _last_orig_train_file - _num_validation_files)
first_last_validation_file = \
    (first_last_train_file[1] + 1, _last_orig_train_file)
assert first_last_validation_file[1] - first_last_validation_file[0] \
       == _num_validation_files - 1
_train_files = \
    ['{base_url}/train-{:05d}-of-01024'.format(i, base_url=_tf_record_dir)
        for i in range(*first_last_train_file)]
_validation_files = \
    ['{base_url}/train-{:05d}-of-01024'.format(i, base_url=_tf_record_dir)
        for i in range(*first_last_validation_file)]
_test_files = \
    ['{base_url}/validation-{:05d}-of-00128'.format(i, base_url=_tf_record_dir)
        for i in range(_num_orig_validation_files)]

# Read settings
# 100 MB read buffer, as suggested for I/O bound pipelines:
# https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
_buffer_size_bytes = 100 * 1024 * 1024
_num_parallel_reads = 1#16


def set_num_parallel_reads(num_parallel_reads):
    global _num_parallel_reads
    _num_parallel_reads = num_parallel_reads


def _create_dataset(files):
    return tf.data.TFRecordDataset(files,
                                   num_parallel_reads=_num_parallel_reads,
                                   buffer_size=_buffer_size_bytes)


keys_to_features = {
    'image/encoded': tf.FixedLenFeature(
        (), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature(
        (), tf.string, default_value='jpeg'),
    'image/class/label': tf.FixedLenFeature(
        [], dtype=tf.int64, default_value=-1),
    'image/class/text': tf.FixedLenFeature(
        [], dtype=tf.string, default_value=''),
    'image/object/bbox/xmin': tf.VarLenFeature(
        dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(
        dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(
        dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(
        dtype=tf.float32),
    'image/object/class/label': tf.VarLenFeature(
        dtype=tf.int64),
}

slim = tf.contrib.slim
items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
    'object/bbox': slim.tfexample_decoder.BoundingBox(
        ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
    'object/label': slim.tfexample_decoder.Tensor(
        'image/object/class/label'),
}

decoder = slim.tfexample_decoder.TFExampleDecoder(
    keys_to_features, items_to_handlers)

# This method needs to always be called, as otherwise, only a tensor with
# encoded data is returned. A dataset.map() call could be placed in the
# XX_dataset() calls, however, we also want the flexibility for users to decide
# when to call this method.
def preprocess_fn(is_training, crop_to=DEFAULT_IMAGE_SIZE):

    def preprocess(record):
        items = decoder.list_items()
        tensors = decoder.decode(record, items)
        record = dict(zip(items, tensors))
        img = record['image']
        label = record['label'] - CLASS_START_INDEX
        distorted_img = slim_preprocessing.preprocess_image(
            img, height=crop_to, width=crop_to, is_training=is_training)
        return distorted_img, label
    return preprocess


def train_dataset():
    return _create_dataset(_train_files)


def eval_dataset():
    return _create_dataset(_validation_files)


def test_dataset():
    return _create_dataset(_test_files)

