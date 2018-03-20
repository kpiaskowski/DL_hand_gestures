import tensorflow as tf

def latent_space(encoder_out, latent_units):
    with tf.variable_scope('latent_space'):
        reduction_conv = tf.layers.conv2d(encoder_out, 256, 1, activation=tf.nn.elu)
        flat = tf.layers.flatten(reduction_conv)
        mean = tf.layers.dense(flat, units=latent_units)
        stddev = 1e-6 + tf.layers.dense(flat, units=latent_units, activation=tf.nn.softplus)
        shape = tf.shape(stddev)
        sample = tf.random_normal([shape[0], shape[1]], mean=0.0, stddev=1.0)
        latent_vector = mean + sample * stddev
        return latent_vector, mean, stddev
