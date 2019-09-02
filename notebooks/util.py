import tensorflow as tf
import efficientnet.efficientnet_builder as efnet_builder
import efficientnet.efficientnet_model as efnet_model

def create_efficientnetb0():
    image_inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    block_args, global_params = efnet_builder.get_model_params('efficientnet-b0', None)
    #with tf.variable_scope('efficientnet-b0'):
    #    model = efnet_model.Model(block_args, global_params)
    features, endpoints = efnet_builder.build_model_base(
        image_inputs, model_name='efficientnet-b0', training=True)
    return features #model #, features


def num_multiadds_in_fc_layer(inputs, outputs):
    return inputs * outputs


def num_multi_adds_in_conv_layer(out_height, out_width, out_channels, 
                                 filter_size=(3,3)):
    filter_factor = 1
    for d in filter_size:
        filter_factor *= d
    ans = out_height * out_width * out_channels * filter_factor
    return ans

def num_multi_adds_in_ds_conv_layer(out_height, out_width, out_channels,
                                   filter_width, filter_height, input_channels):
    out_size = out_width * out_height
    conv_cost = input_channels * filter_width * filter_height * out_size
    depth_cost = out_size * input_channels * out_size
    ans = conv_cost + depth_cost
    return ans