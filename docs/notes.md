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
