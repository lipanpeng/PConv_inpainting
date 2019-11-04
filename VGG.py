import tensorflow as tf
import numpy as np

class VGG(object):

    def __init__(self, img):
        # self.img = img
        self.parameter = []

    def build_vgg(self, img):

        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        self.img = img - mean

        # conv1_1
        with tf.variable_scope('conv1_1'):
            kernel = tf.get_variable(shape=[3, 3, 3, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.img, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2'):
            kernel = tf.get_variable(shape=[3, 3, 64, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1'):
            kernel = tf.get_variable(shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2'):
            kernel = tf.get_variable(shape=[3, 3, 128, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1'):
            kernel = tf.get_variable(shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2'):
            kernel = tf.get_variable(shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3'):
            kernel = tf.get_variable(shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # conv4_1
        with tf.variable_scope('conv4_1'):
            kernel = tf.get_variable(shape=[3, 3, 256, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding=['SAME'])
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # conv5_1
        with tf.variable_scope('conv5_1'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # conv5_3
        with tf.get_variable('conv5_3'):
            kernel = tf.get_variable(shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAEM')
            biases = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out)
            self.parameter += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # fc1
        with tf.variable_scope('fc1'):
            dim = np.prod(self.pool5.get_shape().as_list()[1:])
            fc1_w = tf.get_variable(shape=[dim, 4096], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            fc1_b = tf.get_variable(shape=[4096], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, dim])
            fc1 = tf.nn.bias_add(tf.matmul(pool5_flat, fc1_w), fc1_b)
            self.fc1 = tf.nn.relu(fc1)
            self.parameter += [fc1_w, fc1_b]

        # fc2
        with tf.variable_scope('fc2'):
            fc2_w = tf.get_variable(shape=[4096, 4096], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            fc2_b = tf.get_variable(shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            fc2 = tf.nn.bias_add(tf.matmul(self.fc1, fc2_w), fc2_b)
            self.fc2 = tf.nn.relu(fc2)
            self.parameter += [fc2_w, fc2_b]

        # fc3
        with tf.variable_scope('fc3'):
            fc3_w = tf.get_variable(shape=[4096, 1000], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
            fc3_b = tf.get_variable(shape=[1000], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0), name='biases')
            fc3 = tf.nn.bias_add(tf.matmul(fc2, fc3_w), fc3_b)
            self.fc3 = tf.nn.relu(fc3)
            self.parameter += [fc3_w, fc3_b]

        return self.pool1, self.pool2, self.pool3


    def load_weight(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameter[i].assign(weights[k]))


    def get_feature(self, img, weight_file, sess):
        input = tf.placeholder(dtype=tf.float32, shape=(1, 224, 224, 3))
        feature1, feature2, feature3 = self.build_net(input)
        self.load_weight(weight_file, sess)
        pool1, pool2, pool3 = sess.run([feature1, feature2, feature3], feed_dict={input: img})

        return pool1, pool2, pool3


















