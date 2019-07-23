import tensorflow as tf
import micronet.cifar.dataset as cifar_ds


NUM_TRAINABLE_PARAM = 100 * (cifar_ds.DEFAULT_IMAGE_SIZE * cifar_ds.DEFAULT_IMAGE_SIZE * 3 + 1)


def create_model():
    model = tf.keras.Sequential()
    # Insure input dimensions are correct.
    # Idea from: https://pgaleone.eu/tensorflow/2018/07/28/understanding-tensorflow-tensors-shape-static-dynamic/
    #input = tf.placeholder(cifar_ds.DTYPE,
    #    shape=(None, cifar_ds.IMAGE_SIZE, cifar_ds.IMAGE_SIZE, 3))
    # The linear model doesn't use any layers that are dependent on shape,
    # so we can ignore any reshaping and just flatten the data.

    # FIXME: why does this break?
    # model.add(tf.keras.layers.InputLayer(input_tensor=input))

    # This might work too, instead of the two steps above:
    # model.add(tf.keras.layers.Reshape(
    #    target_shape=(cifar_ds.IMAGE_SIZE * cifar_ds.IMAGE_SIZE * 3,),
    #    input_shape=(cifar_ds.IMAGE_SIZE, cifar_ds.IMAGE_SIZE, 3)))

    model.add(tf.keras.layers.Flatten(
        input_shape=(cifar_ds.DEFAULT_IMAGE_SIZE, cifar_ds.DEFAULT_IMAGE_SIZE, 3)
    ))
    model.add(tf.keras.layers.Dense(100,
                                    activation='softmax'))
    model.summary()
    return model


