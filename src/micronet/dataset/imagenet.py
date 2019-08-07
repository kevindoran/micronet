# FIXME 27. Use 3 splits.

import efficientnet.imagenet_input


_tf_record_dir = 'gs://micronet_bucket1/imageNet'
# DEFAULT_IMAGE_SIZE = 224 # Resolution for EfficientNet B0.
NUM_CLASSES = 1000


# Note: EfficientNet's main keeps a reference to an ImageNetInput object as
# opposed to just holding on to the ImageNetInput's input_fn(). Why is that?
# Is the construction resource consuming? I'll keep it the same way and return
# an ImageNetInput object, but it's worth revisiting.
def _create_input(image_size, is_training, num_parallel_calls, for_tpu,
                  autoaugment):
    use_tpu_transpose_trick = for_tpu
    # FIXME: get bfloat16 working.
    # use_bfloat16 = for_tpu
    use_bfloat16 = False
    autoaugment_name = 'v0' if autoaugment else None
    imagenet_input = efficientnet.imagenet_input.ImageNetInput(
        is_training=is_training,
        data_dir=_tf_record_dir,
        # What is this option for? It puts the batch dim last.
        # transpose_input=use_tpu_transpose_trick,
        transpose_input=False,
        cache=is_training,
        image_size=image_size,
        num_parallel_calls=num_parallel_calls,
        use_bfloat16=use_bfloat16,
        # Whether to use 1001 classes and ignore class index 0, or use 1000.
        include_background_label=False,
        autoaugment_name=autoaugment_name)
    return imagenet_input


def create_train_input(image_size, num_parallel_calls, for_tpu=False,
                       autoaugment=False):
    is_training = True
    return _create_input(image_size, is_training, num_parallel_calls, for_tpu,
                         autoaugment)


def create_eval_input(image_size, num_parallel_calls, for_tpu=False,
                       autoaugment=False):
    is_training = False
    return _create_input(image_size, is_training, num_parallel_calls, for_tpu,
                         autoaugment)



