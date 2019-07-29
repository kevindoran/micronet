"""
Comment from: https://www.tensorflow.org/guide/using_tpu

The simplest way to maintain a model that can be run both on CPU/GPU or on a 
Cloud TPU is to define the model's inference phase (from inputs to predictions) 
outside of the model_fn. Then maintain separate implementations of the Estimator
setup and model_fn, both wrapping this inference step. For an example of this 
pattern compare the mnist.py and mnist_tpu.py implementation in tensorflow/models.
"""

Keras Tensorflow backend:
https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py

Keras implementation of MobileNet V2 (much easier to read):
https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py

And another implementation, but I haven't checked the quality:
https://github.com/xiaochus/MobileNetV2/blob/master/mobilenet_v2.py

Good interpretation of MobileNetv2:
https://machinethink.net/blog/mobilenet-v2/
Although, he claims that compression/decompression are valid analogies, which
I'm not sure I understand. My current perspective is making the input to a layer
highly redundant so that the information is not lost once ReLU is applied. 
Preserving the information allows for smaller feature vectors (as the 
information doesn't need to be maintained scattered within a larger model.)


TPU performance
===============
https://cloud.google.com/tpu/docs/performance-guide
Batch and feature dimensions will be padded. One of these dimensions will be 
padded to 8 and the other will be padded to 128. The XLA compiler will choose.
Best to choose a batch size that is a multiple of 128. 


Troubleshooting
===============
Possible solutions to problems.

1. "Shape must have fixed size for dimension 0"
-----------------------------------------------
If you encounter this problem when running estimator.evaluate() or estimator.train(), one mistake that can cause this
issue is calling dataset.batch(batch_size) instead of calling dataset.batch(batch_size, drop_remainder=True), while
running on a TPU. Without the drop_remainder, the dataset example size cannot be guaranteed, and it becomes 'None' 
hence the error.

      dims = shape.as_list()
      if dims[self._shard_dimension] is None:
        raise ValueError("shape %s must have a fixed size for dimension %d "
                         "that is known at graph construction time." %
>                        (shape.as_list(), self._shard_dimension))
E       ValueError: shape [None, 224, 224, 3] must have a fixed size for dimension 0 that is known at graph construction time.

../venv/lib/python3.5/site-packages/tensorflow/contrib/tpu/python/tpu/tpu_sharding.py:183: ValueError


2. NaN value when run on TPU
----------------------------
Running networks on a TPU generate NaN errors more often than running them on a CPU. Sometimes, this seems to just be
a result of other errors being uncaught on the TPU. For example, an NaN error was encountered on a TPU. Running the
same program with a CPU produced a clear error stating that an example label was out of the expected [0, 1000) range. 
Fixing this issue and running on a TPU lead to no errors.  


Optimization
============
This is an area with significant potential for improvement.

Terminology
-----------
iteration: a period of training where a single batch of data has been processed.
global step: the number of batches/iterations processed so far.
epoch: a period of training where the graph has seen every sample once.
(what if the dataset doesn't repeat, as it uses endless random augmentation?)

Common approaches
-----------------
A common approach is to choose a popular optimizer, such as a SGD or Adam
optimizer. Set the learning rate to decay exponentially (decay occurring at
epoch boundaries, which I think seems like an arbitrary place to execute the 
decay). Then experiment with different initial rates and decay factors.

Better approach
---------------
A better approach would be to monitor how the model accuracy develops and to
update the learning rate accordingly. A simulated annealing approach could be 
taken where the accuracy after X number of iterations/batches determines whether
the model should continue or be reset to a previous state (with the addition of
another hyper-parameter 'energy' which is used to determine what level of 
accuracy _decrease_ still permits continuing without restoring from checkpoint).

Surely there are some online algorithms in Tensorflow already?

LRFinder
--------
For choosing the initial rate: 
https://medium.com/octavian-ai/how-to-use-the-learning-rate-finder-in-tensorflow-126210de9489

