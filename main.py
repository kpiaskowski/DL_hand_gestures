import os

import tensorflow as tf

from architectures.decoder import decoder_model
from architectures.latent_space import latent_space
from architectures.pretrained_encoder import encoder_model
from dataprovider import DataProvider

epochs = 50
batch_size = 20
latent_units = 200
l_rate = 0.00001
label_w = 64
label_h = 48
model_name = 'model'

# data
data_provider = DataProvider(batch_size, root_folder='../data', label_w=label_w, label_h=label_h)
train_num_batches, val_num_batches = data_provider.get_num_batches()

training_dataset_init, val_dataset_init, images, labels = data_provider.get_data()

# model
encoder = encoder_model(images)
latent_vector, mean, stddev = latent_space(encoder, latent_units)
predictions = decoder_model(latent_vector)

# losses
generative_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(predictions, labels), axis=1), axis=1)
latent_loss = -0.5 * tf.reduce_mean(1.0 + 2.0 * stddev - tf.square(mean) - tf.exp(2.0 * stddev), 1)
loss = tf.reduce_mean(generative_loss + latent_loss)

# summaries
preview_images = tf.reshape(images, [-1, 240, 320, 3])
preview_predictions = tf.expand_dims(predictions, -1)
preview_predictions = tf.image.grayscale_to_rgb(preview_predictions)
preview_predictions = tf.image.resize_images(preview_predictions, [240, 320])
preview_labels = tf.reshape(labels, [-1, label_h, label_w])
preview_labels = tf.expand_dims(preview_labels, -1)
preview_labels = tf.image.grayscale_to_rgb(preview_labels)
preview_labels = tf.image.resize_images(preview_labels, [240, 320])
concat_preview = tf.concat([preview_images, preview_labels, preview_predictions], 2)
tf.summary.image('predictions', concat_preview, max_outputs=5)
tf.summary.scalar('loss', loss)

train_op = tf.train.AdamOptimizer(l_rate).minimize(loss)

merged = tf.summary.merge_all()
pretrained_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))
saver = tf.train.Saver()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summaries/train/' + model_name, flush_secs=60)
    val_writer = tf.summary.FileWriter('summaries/val/' + model_name, flush_secs=60)
    sess.run(tf.global_variables_initializer())

    if len(os.listdir('saved_model')) > 1:
        saver.restore(sess, 'saved_model/' + model_name + '.ckpt')
        print('Continuing training')
    else:
        pretrained_loader.restore(sess, '../pretrained_imagenet/pretrained_imagenet.ckpt')  # load encoder pretrained on imagenet
        print('Training from scratch')

    for epoch in range(epochs):
        # training
        sess.run(training_dataset_init)
        i = 0
        while True:
            try:
                _, cost, summary = sess.run([train_op, loss, merged])
                train_writer.add_summary(summary, epoch * train_num_batches + i)
                train_writer.flush()
                print('Training epoch: {} of {}, batch: {} of {}, cost: {}'.format(epoch, epochs, i, train_num_batches, cost))
                i += 1
            except tf.errors.OutOfRangeError:
                break

        # validation
        sess.run(val_dataset_init)
        i = 0
        while True:
            try:
                cost, summary = sess.run([loss, merged])
                val_writer.add_summary(summary, epoch * val_num_batches + i)
                val_writer.flush()
                print('Validation epoch: {} of {}, batch: {} of {}, cost: {}'.format(epoch, epochs, i, val_num_batches, cost))
                i += 1
            except tf.errors.OutOfRangeError:
                break

        saver.save(sess, 'saved_model/' + model_name + '.ckpt')
