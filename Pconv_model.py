from ops import pconv2d, decode_layer
import numpy as np
import os
import tensorflow as tf
import cv2
from VGG import VGG

class PConvUnet(object):

    def __init__(self):
        pass


    def pconv_net (self, img, mask, is_train=True):
        input_img = img
        input_mask = mask

        with tf.variable_scope("PconvNet"):
            # encode stage
            e_conv1, e_mask1 = pconv2d([input_img, input_mask], ksize=7, filters=64, strides=2, bn=False, name='e_conv1', is_train=is_train)
            e_conv2, e_mask2 = pconv2d([e_conv1, e_mask1], ksize=5, filters=128, strides=2, name='e_conv2', is_train=is_train)
            e_conv3, e_mask3 = pconv2d([e_conv2, e_mask2], ksize=5, filters=256, strides=2, name='e_conv3', is_train=is_train)
            e_conv4, e_mask4 = pconv2d([e_conv3, e_mask3], ksize=3, filters=512, strides=2, name='e_conv4', is_train=is_train)
            e_conv5, e_mask5 = pconv2d([e_conv4, e_mask4], ksize=3, filters=512, strides=2, name='e_conv5', is_train=is_train)
            e_conv6, e_mask6 = pconv2d([e_conv5, e_mask5], ksize=3, filters=512, strides=2, name='e_conv6', is_train=is_train)
            e_conv7, e_mask7 = pconv2d([e_conv6, e_conv6], ksize=3, filters=512, strides=2, name='e_conv7', is_train=is_train)
            e_conv8, e_mask8 = pconv2d([e_conv7, e_mask7], ksize=3, filters=512, strides=2, name='e_conv8', is_train=is_train)

            # decode stage
            d_conv9, d_mask9 = decode_layer(e_conv8, e_mask8, e_conv7, e_mask7, ksize=3, filters=512, name='d_conv9', is_train=is_train)
            d_conv10, d_mask10 = decode_layer(d_conv9, d_mask9, e_conv6, e_mask6, ksize=3, filters=512, name='d_conv10', is_train=is_train)
            d_conv11, d_mask11 = decode_layer(d_conv10, d_mask10, e_conv5, e_mask5, ksize=3, filters=512, name='d_conv11', is_train=is_train)
            d_conv12, d_mask12 = decode_layer(d_conv11, d_mask11, e_conv4, e_mask4, ksize=3, filters=512, name='d_conv12', is_train=is_train)
            d_conv13, d_mask13 = decode_layer(d_conv12, d_mask12, e_conv3, e_mask3, ksize=3, filters=256, name='d_conv13', is_train=is_train)
            d_conv14, d_mask14 = decode_layer(d_conv13, d_mask13, e_conv2, e_mask2, ksize=3, filters=128, name='d_conv14', is_train=is_train)
            d_conv15, d_mask15 = decode_layer(d_conv14, d_mask14, e_conv1, e_mask1, ksize=3, filters=64, name='d_conv15', is_train=is_train)
            d_conv16, d_mask16 = decode_layer(d_conv15, d_mask15, input_img, input_mask, ksize=3, filters=3, bn=False, name='d_conv16')

            output = pconv2d(d_conv16, ksize=1, filters=3, strides=1, padding='SAME', activate='sigmoid', bn=False, name='conv17')

        return output


    def loss_total(self, mask, y_true, y_pred):
        y_comp = mask * y_true + (1 - mask) * y_pred

        vgg_out = self.vgg(y_pred)
        vgg_gt = self.vgg(y_true)
        vgg_comp = self.vgg(y_comp)

        l_valid = self.loss_valid(mask, y_true, y_pred)
        l_hole = self.loss_hole(mask, y_true, y_pred)
        l_perceptual = self.loss_preceptual(vgg_out, vgg_gt, vgg_comp)
        l_style = self.loss_style(vgg_out, vgg_gt) + self.loss_style(vgg_comp, vgg_gt)
        l_tv = self.loss_tv(mask, y_comp)

        return l_valid + 6*l_hole + 0.05*l_perceptual + 120*l_style + 0.1*l_tv

    def loss_valid(self, mask, y_true, y_pred):
        return tf.losses.absolute_difference(mask*y_true, mask*y_pred)

    def loss_hole(self, mask, y_true, y_pred):
        return tf.losses.absolute_difference((1-mask)*y_true, (1-mask)*y_pred)

    def loss_preceptual(self, vgg_out, vgg_gt, vgg_comp):
        loss = 0
        for o, g, c in zip(vgg_out, vgg_gt, vgg_comp):
            loss += tf.losses.absolute_difference(o, g) + tf.losses.absolute_difference(o, c)

        return loss

    def loss_style(self, vgg_out, vgg_gt):
        loss = 0
        for o, g in zip(vgg_out, vgg_gt):
            loss += tf.losses.absolute_difference(self.gram_matrix(o), self.gram_matrix(g))

        return loss

    def loss_tv(self, mask, y_comp):
        loss = 0
        kernel = tf.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = tf.nn.conv2d(1-mask, kernel, strides=[1, 1, 1, 1], padding='SAME')
        dilated_mask = tf.cast(tf.greater(dilated_mask, 0), tf.float32)
        P = dilated_mask * y_comp
        loss += tf.losses.absolute_difference(P[:, 1:, :, :], P[:, :-1, :, :])
        loss += tf.losses.absolute_difference(P[:, :, 1:, :], P[:, :, :-1, :])

        return loss

    def gram_matrix(self, x):
        x = tf.transpose(x, [0, 3, 1, 2])
        shape = x.get_shape().as_list()
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]

        feature = tf.reshape(x, [B, C, H*W])
        feature_t = tf.transpose(feature, [0, 2, 1])
        gram = tf.matmul(feature, feature_t)

    def vgg(self, input):
        self.vgg_parameter = []

        # # zero-mean input
        # with tf.name_scope('preprocess') as scope:
        #     mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #     input = input - mean

        # conv1_1
        with tf.variable_scope('conv1_1'):
            kernel = tf.get_variable(shape=[3, 3, 3, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2'):
            kernel = tf.get_variable(shape=[3, 3, 64, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1'):
            kernel = tf.get_variable(shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2'):
            kernel = tf.get_variable(shape=[3, 3, 128, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1'):
            kernel = tf.get_variable(shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2'):
            kernel = tf.get_variable(shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3'):
            kernel = tf.get_variable(shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # pool3
        pool3 = tf.nn.max_pool(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # conv4_1
        with tf.variable_scope('conv4_1'):
            kernel = tf.get_variable(shape=[3, 3, 256, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding=['SAME'])
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # pool4
        pool4 = tf.nn.max_pool(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # conv5_1
        with tf.variable_scope('conv5_1'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # conv5_3
        with tf.get_variable('conv5_3'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                     trainable=False, name='weights')
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAEM')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                     trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out)
            self.vgg_parameter += [kernel, biases]

        # pool5
        pool5 = tf.nn.max_pool(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # fc1
        with tf.variable_scope('fc1'):
            dim = np.prod(pool5.get_shape().as_list()[1:])
            fc1_w = tf.get_variable(shape=[dim, 4096], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                    trainable=False, name='weights')
            fc1_b = tf.get_variable(shape=[4096], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                    trainable=False, name='biases')
            pool5_flat = tf.reshape(pool5, [-1, dim])
            fc1 = tf.nn.bias_add(tf.matmul(pool5_flat, fc1_w), fc1_b)
            fc1 = tf.nn.relu(fc1)
            self.vgg_parameter += [fc1_w, fc1_b]

        # fc2
        with tf.variable_scope('fc2'):
            fc2_w = tf.get_variable(shape=[4096, 4096], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                    trainable=False, name='weights')
            fc2_b = tf.get_variable(shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                    trainable=False, name='biases')
            fc2 = tf.nn.bias_add(tf.matmul(fc1, fc2_w), fc2_b)
            fc2 = tf.nn.relu(fc2)
            self.vgg_parameter += [fc2_w, fc2_b]

        # fc3
        with tf.variable_scope('fc3'):
            fc3_w = tf.get_variable(shape=[4096, 1000], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03),
                                    trainable=False, name='weights')
            fc3_b = tf.get_variable(shape=[1000], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0),
                                    trainable=False, name='biases')
            fc3 = tf.nn.bias_add(tf.matmul(fc2, fc3_w), fc3_b)
            fc3 = tf.nn.relu(fc3)
            self.vgg_parameter += [fc3_w, fc3_b]

        return pool1, pool2, pool3

    def load_weight(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            sess.run(self.vgg_parameter[i].assign(weights[k]))