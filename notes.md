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