import numpy as np
from collections import OrderedDict
import tensorflow as tf


def unet(feature_in, labels, n_blocks=5, kernel_size=3, feature_num=64, n_classes=2, pool_size=2):
    down_conv = OrderedDict()
    b_constant = 0.1
    w_stddev = np.sqrt(2 / (kernel_size ** n_blocks * feature_num))

    for i_layer in range(0, n_blocks):
        if i_layer == 0:
            channel_out = feature_num
        else:
            channel_out = 2 ** i_layer * feature_num

        feature_out1 = tf.nn.relu(normalization(
            feature_in=tf.layers.conv2d(inputs=feature_in, filters=channel_out, kernel_size=kernel_size, padding='SAME',
                                        kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                        bias_initializer=tf.initializers.constant(b_constant)),
            name='Downconv_%d_Conv_1' % i_layer))

        down_conv[i_layer] = tf.nn.relu(normalization(
            feature_in=tf.layers.conv2d(inputs=feature_out1, filters=channel_out, kernel_size=kernel_size,
                                        padding='SAME',
                                        kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                        bias_initializer=tf.initializers.constant(b_constant)),
            name='Downconv_%d_Conv_2' % i_layer))
        feature_in = tf.layers.max_pooling2d(inputs=down_conv[i_layer], pool_size=pool_size, strides=pool_size)

    for i_layer in range(n_blocks, -1, -1):

        channel_out = 2 ** i_layer * feature_num

        feature_out1 = tf.nn.relu(normalization(
            feature_in=tf.layers.conv2d(inputs=feature_in, filters=channel_out, kernel_size=kernel_size, padding='SAME',
                                        kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                        bias_initializer=tf.initializers.constant(b_constant)),
            name='Upconv_%d_Conv_1' % i_layer))

        feature_out2 = tf.nn.relu(normalization(
            feature_in=tf.layers.conv2d(inputs=feature_out1, filters=channel_out, kernel_size=kernel_size,
                                        padding='SAME',
                                        kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                        bias_initializer=tf.initializers.constant(b_constant)),
            name='Upconv_%d_Conv_2' % i_layer))

        if i_layer != 0:

            feature_out3 = tf.nn.relu(
                normalization(feature_in=tf.layers.conv2d_transpose(inputs=feature_out2, filters=channel_out // 2,
                                                                    kernel_size=pool_size, strides=pool_size,
                                                                    padding='VALID',
                                                                    kernel_initializer=tf.initializers.truncated_normal(
                                                                        stddev=w_stddev),
                                                                    bias_initializer=tf.initializers.constant(
                                                                        b_constant)),
                              name='Deconv_%d' % i_layer))
            feature_in = tf.concat([down_conv[i_layer - 1], feature_out3], 3)
        else:
            logits = tf.layers.conv2d(inputs=feature_out2, filters=n_classes, kernel_size=1, padding='SAME',
                                      use_bias=False,
                                      kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev))

    prediction = conclusion(logits=logits, labels=labels, n_classes=n_classes)

    return prediction, logits


def normalization(feature_in, name, epsilon=1e-12):
    shape = feature_in.shape
    n_channels = shape[-1]
    mean, var = tf.nn.moments(feature_in, [0, 1, 2], keep_dims=True)
    with tf.variable_scope(name):
        gamma = tf.get_variable(name='Gamma', shape=n_channels, dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable(name='Beta', shape=n_channels, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))

        return tf.nn.batch_normalization(x=feature_in, mean=mean, variance=var, offset=beta, scale=gamma,
                                         variance_epsilon=epsilon)


def layer_normalization(feature_in, name, epsilon=1e-12):
    shape = feature_in.shape
    n_channels = shape[-1]
    Mean, Var = tf.nn.moments(feature_in, [0, 1, 2, 3, 4], keep_dims=True)
    with tf.variable_scope(name):
        Gamma = tf.get_variable(name='Gamma', shape=n_channels, dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0))
        Beta = tf.get_variable(name='Beta', shape=n_channels, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=feature_in, mean=Mean, variance=Var, offset=Beta, scale=Gamma,
                                     variance_epsilon=epsilon)


def group_normalization(feature_in, name, epsilon=1e-12):
    Group = 4
    shape_in = feature_in.shape
    n_channels = shape_in[-1]

    feature_in_reshape = tf.reshape(feature_in, shape=[shape_in[0], shape_in[1], shape_in[2], shape_in[3], Group,
                                                       shape_in[4] // Group])
    Mean_reshape, Var_reshape = tf.nn.moments(feature_in_reshape, [0, 1, 2, 3, 5], keep_dims=True)
    feature_in = tf.reshape((feature_in_reshape - Mean_reshape) / (Var_reshape + epsilon), shape=shape_in)

    Mean = tf.constant(value=0.0, dtype=tf.float32, shape=shape_in)
    Var = tf.constant(value=1.0, dtype=tf.float32, shape=shape_in)

    with tf.variable_scope(name):
        Gamma = tf.get_variable(name='Gamma', shape=n_channels, dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0))
        Beta = tf.get_variable(name='Beta', shape=n_channels, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=feature_in, mean=Mean, variance=Var, offset=Beta, scale=Gamma,
                                     variance_epsilon=0)


def conclusion(logits, labels, n_classes):
    foreground = tf.cast(
        tf.greater_equal(tf.slice(tf.nn.softmax(logits=logits), begin=[0, 0, 0, 1], size=[-1, -1, -1, n_classes - 1]),
                         y=0.5), dtype=tf.int32)
    background = tf.subtract(x=1, y=foreground)
    classes = tf.concat(values=[background, foreground], axis=3)

    foreground_gt = tf.cast(x=tf.slice(input_=labels, begin=[0, 0, 0, 1], size=[-1, -1, -1, 1]), dtype=tf.int32)
    intersection = tf.bitwise.bitwise_and(foreground, foreground_gt)
    union = tf.bitwise.bitwise_or(foreground, foreground_gt)

    iou_1 = tf.constant(1, dtype=tf.float32)
    iou_2 = tf.cast(tf.reduce_sum(intersection), dtype=tf.float32) / tf.cast(tf.reduce_sum(union), dtype=tf.float32)
    iou = tf.cond(tf.equal(tf.reduce_sum(union), 0), lambda: iou_1, lambda: iou_2)

    prediction = {"probabilities": tf.nn.softmax(logits=logits, name="probabilities"),
                  "classes": classes,
                  'IoU': iou,
                  'Or': tf.reduce_sum(union),
                  'And': tf.reduce_sum(intersection)}

    return prediction
