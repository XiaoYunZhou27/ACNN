import numpy as np
import tensorflow as tf


def deeplab_v3plus(feature_in, labels, backbone='resnet101', feature_num=64, n_classes=2):
    # Feature extraction
    if backbone == 'resnet101':
        feature_out = resnet_backbone(feature_in, feature_num)

    # Atrous spatial pyramid pooling
    encoder_out = aspp(feature_out['output_stride16'], 4 * feature_num)

    # Decoder
    logits = deeplab_decoder(feature_out['output_stride4'], encoder_out, 4 * feature_num, n_classes)

    # Prediction result
    prediction = conclusion(logits=logits, labels=labels, n_classes=n_classes)

    return prediction, logits


def resnet_backbone(feature_in, feature_num):
    with tf.variable_scope('resnet101'):
        # output_stride = 2
        conv1 = conv_layer(feature_in, feature_num, 'RESNET_conv1', kernel_size=7, strides=2)

        # output_stride = 4
        conv2 = pooling_layer(conv1, 'RESNET_pool1')
        for i in range(3):
            if i == 0:
                conv2 = bottleneck(conv2, feature_num, 'RESNET_conv2_%i' % (i + 1), shortcut='projection')
            else:
                conv2 = bottleneck(conv2, feature_num, 'RESNET_conv2_%i' % (i + 1))

        # output_stride = 8
        for i in range(4):
            if i == 0:
                conv3 = bottleneck(conv2, 2 * feature_num, 'RESNET_conv3_%i' % (i + 1), shortcut='projection',
                                   downsample=True)
            else:
                conv3 = bottleneck(conv3, 2 * feature_num, 'RESNET_conv3_%i' % (i + 1))

        # output_stride = 16
        for i in range(23):
            if i == 0:
                conv4 = bottleneck(conv3, 4 * feature_num, 'RESNET_conv4_%i' % (i + 1), shortcut='projection',
                                   downsample=True)
            else:
                conv4 = bottleneck(conv4, 4 * feature_num, 'RESNET_conv4_%i' % (i + 1))

        return {'output_stride2': conv1, 'output_stride4': conv2, 'output_stride8': conv3, 'output_stride16': conv4}


# Atrous Spatial Pyramid Pooling
def aspp(feature_in, feature_num):
    with tf.variable_scope('aspp'):
        # Global average pooling
        feature_size = tf.shape(feature_in)[1:3]
        ip = tf.reduce_mean(feature_in, [1, 2], name='ASPP_image_pooling', keepdims=True)
        ip = tf.image.resize_bilinear(ip, feature_size)

        # 1x1 conv
        aconv1 = atrous_conv_layer(feature_in, feature_num, 1, 'ASPP_1x1_conv', kernel_size=1)

        # 3x3 conv rate 6
        aconv2 = atrous_conv_layer(feature_in, feature_num, 6, 'ASPP_3x3_conv_rate_6')

        # 3x3 conv rate 12
        aconv3 = atrous_conv_layer(feature_in, feature_num, 12, 'ASPP_3x3_conv_rate_12')

        # 3x3 conv rate 18
        aconv4 = atrous_conv_layer(feature_in, feature_num, 18, 'ASPP_3x3_conv_rate_18')

        # Concatenate
        cat = tf.concat([ip, aconv1, aconv2, aconv3, aconv4], axis=3, name='ASPP_concat')

        return conv_layer(cat, feature_num, 'ASPP_1x1_conv_output', kernel_size=1)


# Decoder
def deeplab_decoder(feature_in, encoder_out, feature_num, n_classes, reduced_feature_num=48):
    with tf.variable_scope('decoder'):
        low_level_features = conv_layer(feature_in, reduced_feature_num, 'DECODER_1x1_conv', kernel_size=1)

        feature_size = tf.shape(low_level_features)[1:3]
        feature_upsampled = tf.image.resize_bilinear(encoder_out, feature_size)

        cat = tf.concat([low_level_features, feature_upsampled], axis=3, name='DECODER_concat')

        feature_out = conv_layer(cat, feature_num, 'DECODER_3x3_conv1')
        feature_out = conv_layer(feature_out, feature_num, 'DECODER_3x3_conv2')
        feature_out = conv_layer(feature_out, n_classes, 'DECODER_out', kernel_size=1, norm=None, activation=None,
                                 use_bias=False)

        return tf.image.resize_bilinear(feature_out, 4 * feature_size)


# Deeper ResNet uses 'bottleneck' building blocks
def bottleneck(feature_in, feature_num, name, shortcut='identity', downsample=False):
    with tf.variable_scope(name):
        if downsample:
            conv_a = conv_layer(feature_in, feature_num, 'branch2a', kernel_size=1, strides=2)
        else:
            conv_a = conv_layer(feature_in, feature_num, 'branch2a', kernel_size=1)
        conv_b = conv_layer(conv_a, feature_num, 'branch2b')
        conv_c = conv_layer(conv_b, 4 * feature_num, 'branch2c', kernel_size=1, activation=None)

        if shortcut == 'identity':
            # Identity mapping
            if downsample:
                # TODO: Pad with extra zero entries
                pass
            else:
                feature_out = tf.add_n([conv_c, feature_in], name='branch1')

        elif shortcut == 'projection':
            # Projection mapping
            if downsample:
                projection = conv_layer(feature_in, 4 * feature_num, 'projection', kernel_size=1, strides=2,
                                        activation=None)
            else:
                projection = conv_layer(feature_in, 4 * feature_num, 'projection', kernel_size=1, activation=None)
            feature_out = tf.add_n([conv_c, projection], name='branch1')

        return tf.nn.relu(feature_out)


# Convolution with normalization and activation
def conv_layer(feature_in, channel_out, name, kernel_size=3, strides=1, norm='norm', activation='relu', use_bias=True):
    w_stddev = np.sqrt(2 / (kernel_size ** 2 * channel_out))
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=feature_in,
                                filters=channel_out,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='SAME',
                                use_bias=use_bias,
                                kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                bias_initializer=tf.initializers.constant(0.1),
                                trainable=True)
        if norm == 'norm':
            conv = normalization(conv, '%s_norm' % name)

        if activation == 'relu':
            return tf.nn.relu(conv)
        return conv


# Dilated convolution with normalization and activation
def atrous_conv_layer(feature_in, channel_out, dilation_rate, name, kernel_size=3, strides=1):
    w_stddev = np.sqrt(2 / (kernel_size ** 2 * channel_out))
    with tf.variable_scope(name):
        return tf.layers.conv2d(inputs=feature_in,
                                filters=channel_out,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='SAME',
                                dilation_rate=dilation_rate,
                                kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                bias_initializer=tf.initializers.constant(0.1),
                                trainable=True)


# Pooling
def pooling_layer(feature_in, name, pool_size=3, strides=2, padding='same'):
    with tf.variable_scope(name):
        return tf.layers.max_pooling2d(inputs=feature_in,
                                       pool_size=pool_size,
                                       strides=strides,
                                       padding=padding)


# Normalization, default instance normalization
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
