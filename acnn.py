import numpy as np
import tensorflow as tf


def acnn(feature_in, labels, n_blocks=128, kernel_size=3, feature_num=16, n_classes=2):
    stddev = np.sqrt(2 / (kernel_size ** 5 * feature_num))
    for layer in range(0, n_blocks):

        if layer == 0:
            features = tf.layers.conv2d(inputs=feature_in, filters=feature_num, kernel_size=kernel_size,
                                        activation=None, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        bias_initializer=tf.constant_initializer(value=0.1),
                                        name='Conv%d_1' % layer)
            features_add = tf.nn.relu(normalization(features, 'Conv%d_1_normed' % layer))
        else:
            features = tf.layers.conv2d(inputs=features, filters=feature_num, kernel_size=kernel_size,
                                        activation=None, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        bias_initializer=tf.constant_initializer(value=0.1),
                                        name='Conv%d_1' % layer)
            features_add = tf.nn.relu(normalization(features, 'Conv%d_1_normed' % layer) + features_add)

        features = tf.layers.conv2d(inputs=features_add, filters=feature_num, kernel_size=kernel_size,
                                    activation=None, padding='same', dilation_rate=3,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    bias_initializer=tf.constant_initializer(value=0.1),
                                    name='Conv%d_2' % layer)
        features = tf.nn.relu(normalization(features, 'Conv%d_2_normed' % layer))

    logits = tf.layers.conv2d(inputs=features, filters=n_classes, kernel_size=1, activation=None, padding='same',
                              use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))

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
