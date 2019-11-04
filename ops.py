import tensorflow as tf
import cv2
import numpy as np

def get_w(shape, name='w'):
    w = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    return w


def get_b(shape, name='b'):
    b = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    return b


def get_mask_w(shape):
    w = tf.ones(shape=shape, dtype=tf.float32)
    return w

def pconv2d(input, ksize, filters, strides, padding='SAME', activate='relu', bn=True, name='pconv', is_train=True):
    input_dim = input[0].get_shape().as_list()
    w_shape = [ksize, ksize, input_dim, filters]
    w = get_w(w_shape, name=name+'_w')
    b_shape = [filters]
    b = get_b(b_shape, name=name+'_b')
    mask_sum = tf.reduce_sum(input[1], [1, 2], keepdims=True)
    mask_sum = tf.tile(mask_sum, [1, input[1].get_shape().as_list()[1], input[1].get_shape().as_list()[2], 1])
    img_output = tf.nn.conv2d((input[0]*input[1]) / mask_sum, w, [1, strides, strides, 1], padding=padding, name=name)
    img_output = tf.nn.bias_add(img_output, b, name=name+'_bias_add')
    if bn:
        img_output = tf.layers.batch_normalization(img_output, training=is_train, name=name+'_bn')
    if activate == 'relu':
        img_output = tf.nn.relu(img_output)
    elif activate == 'leaky_relu':
        img_output = tf.nn.leaky_relu(img_output)
    elif activate == 'sigmoid':
        img_output = tf.nn.sigmoid(img_output)
    mask_w = get_mask_w(w_shape)
    mask_output = tf.nn.conv2d(input[1], mask_w, strides=strides, padding=padding)
    mask_output = tf.cast(tf.greater(mask_output, 0), tf.float32)

    return img_output, mask_output


def decode_layer(img_in, mask_in, e_conv, e_mask, ksize, filters, stride=1, bn=True, is_train=True, name='decode'):
    height_in, width_in = img_in.get_shape().as_list()[1:3]
    height_new, width_new = height_in*2, width_in*2
    img_up = tf.image.resize_nearest_neighbor(img_in, (height_new, width_new))
    mask_up = tf.image.resize_nearest_neighbor(mask_in, (height_new, width_new))
    concat_img = tf.concat([img_up, e_conv], axis=-1)
    concat_mask = tf.concat([mask_up, e_mask], axis=-1)
    conv, mask = pconv2d([concat_img, concat_mask], ksize=ksize, filters=filters, strides=stride, activate='leaky_relu', bn=bn,
                         is_train=is_train, name=name)

    return conv, mask
