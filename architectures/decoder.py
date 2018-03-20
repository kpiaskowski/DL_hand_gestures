import tensorflow as tf


def decoder_model(latent_vector):
    with tf.variable_scope('decoder'):
        net = tf.layers.dense(latent_vector, units=768*4,
                              activation=tf.nn.elu)
        net = tf.reshape(net, [-1, 6, 8, 16*4])

        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(3, 3),
                               padding='same', activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3),
                               padding='same', activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        net = tf.image.resize_images(net, size=(12, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3),
                               padding='same', activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3),
                               padding='same', activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        net = tf.image.resize_images(net, size=(24, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3),
                               padding='same', activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3),
                               padding='same', activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        net = tf.image.resize_images(net, size=(48, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3),
                               padding='same', activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3),
                               padding='same', activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))

        net = tf.layers.conv2d(inputs=net, filters=1, kernel_size=(3, 3),
                               padding='same', activation=tf.nn.sigmoid)
        predictions = tf.squeeze(net, -1)
        return predictions
