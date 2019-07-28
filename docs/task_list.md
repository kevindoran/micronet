
Tasks
=====
1. Use unit8 instead of float32
Use uint8 datatype for cifar images and labels to reduce required memory.
The image preprocessing step normalizes the pixel values to [0, 1) and thus 
converts the data to float32. To switch to 8-bit we will lose any benefits of
data normalization.

The normalization step that will need to be removed:

    img = tf.image.per_image_standardization(img)
    
2. It would be nice to be able to count the elements of a small dataset.
For a small dataset like cifar, counting elements would be useful in tests. I 
tried using a dataset reduction, but either it's not working or it's taking too 
long.

3. Determine if 'channels_first' or 'channels_last' is better for TPUs.

4. Found out if labels should be loaded as scalars or as one-hot vectors. 

5. Finish implementing a count of trainable parameter bytes in test/common.py. 

6. Make sure that the input image normalization is identical between training
and test data; they should be normalized by the same mean and standard 
deviation.

7. Implement a method to approximate the trainable parameter counting for 
MobileNetv2.

8. Decide which layers have L2 weight regularization applied.

9. Get RMSPropOptimizer working.

10. Insure the BatchNormalization layer has the correct axis set.

11. Add residuals.

12. Add more checks to test_get_cluster_resolver.

13. Fix the test_micronetv2_model.py's assert for random performance.

14. Factor out some of the testing code that does the steps:
        evaluate, train, evaluate.
        
15. Store tfrecords in the correct dimension. Storing the features in the 
(32, 32, 3) dimension will avoid a reshape, which is potentially an expensive 
operation on TPUs. https://cloud.google.com/tpu/docs/performance-guide

16.Implement the test 'A warning is raised if batch size is not divisible by 
   128'. This is for test_estimator.py.
   
17. Add checks for global_steps/sec in test_estimator.py. This is motivated by
the create_tpu_estimator() method creating an estimator that operates slower
that the one copied from tensorflow/tpu repo example.

18. Check out why the TPU is becoming 'Unhealthy' when multiple TPU tests are
collected in a single pytest run. Could be a timing issue (maybe there is a 
need to introduce a delay between tests that use TPUs).

19. Figure out why we are getting the error: 
        'ERROR:tensorflow:Operation of type Placeholder (input_1) is not 
        supported on the TPU. Execution will fail if this op is used in the 
        graph.' 
Possibly related to: https://github.com/tensorflow/tensorflow/issues/25652

20. Add a test for the estimator_fn fixture for when it returns an Estimator
    (not a TPUEstimator).
    
21. Make cifar.dataset source the data from either Google cloud storage or the 
local drive.

22. Create propper assert near that accounts for the [0, 1] probability bound.

23. Inspect overfitting with test_cifar_dataset.py's test_with_estimator test.

24. Switch to AdafactorOptimizer as noted in tpu troubleshooting docs.
    https://cloud.google.com/tpu/docs/troubleshooting
    
25. Enable live logging in pytest so as to investigate performance such as
issues.
