
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
