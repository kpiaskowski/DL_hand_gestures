import tensorflow as tf
from architectures.pretrained_encoder import encoder_model
from architectures.latent_space import latent_space
from architectures.decoder import decoder_model
from dataprovider import DataProvider
import os
epochs = 50
batch_size = 20
latent_units = 200
l_rate = 0.0001

# data
data_provider = DataProvider(batch_size, root_folder='../data')
train_num_batches, val_num_batches = data_provider.get_num_batches()

training_dataset_init, val_dataset_init, images, labels = data_provider.get_data()

# model
encoder = encoder_model(images)
latent_vector, mean, stddev = latent_space(encoder, latent_units)
predictions = decoder_model(latent_vector)

# losses
generative_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(predictions, labels), axis=1), axis=1)
latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mean) + tf.square(stddev) - tf.log(1e-8 + tf.square(stddev)) - 1, 1))
loss = tf.reduce_mean(generative_loss + latent_loss)

# summaries
preview_images = tf.reshape(images, [-1, 240, 320, 3])
preview_images = tf.image.resize_images(preview_images, [data_provider.label_h, data_provider.label_w])
preview_images = tf.squeeze(tf.image.rgb_to_grayscale(preview_images), -1)
tf.summary.image('predictions', tf.expand_dims(tf.concat([preview_images, predictions, labels], axis=-1), -1), max_outputs=3)
tf.summary.scalar('loss', loss)

train_op = tf.train.AdamOptimizer(l_rate).minimize(loss)

merged = tf.summary.merge_all()
pretrained_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))
saver = tf.train.Saver()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summaries/train', flush_secs=60)
    val_writer = tf.summary.FileWriter('summaries/val', flush_secs=60)
    sess.run(tf.global_variables_initializer())

    if len(os.listdir('saved_model')) > 1 :
        saver.restore(sess, 'saved_model/model.ckpt')
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
                print('Validation epoch: {} of {}, batch: {} of {}, cost: {}'.format(epoch, epochs, i, val_num_batches, cost))
                i += 1
            except tf.errors.OutOfRangeError:
                break

        saver.save(sess, 'saved_model/model.ckpt')