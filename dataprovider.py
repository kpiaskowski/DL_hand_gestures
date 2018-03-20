import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


class DataProvider():
    def __init__(self, batch_size, root_folder='data', label_w=64, label_h=48):
        self.label_w, self.label_h = label_w, label_h
        self.batch_size = batch_size

        self.img_names, self.label_names = self.get_filenames(root_folder)
        self.img_names, self.label_names = shuffle([self.img_names, self.label_names])


    def get_num_batches(self):
        return int(len(self.img_names) * 0.95) // self.batch_size, int(len(self.img_names) * 0.05) // self.batch_size

    def get_filenames(self, root_folder):
        """
        Searches through folders and extract filenames
        """
        images = []
        labels = []
        for dirName, subdirList, fileList in os.walk(root_folder):
            files = [fname for fname in fileList]
            if len(files) > 0:
                if 'Depth' in dirName:
                    labels.extend([os.path.join(dirName, fname) for fname in fileList])
                elif 'Color' in dirName:
                    images.extend([os.path.join(dirName, fname) for fname in fileList])
        return sorted(images), sorted(labels)

    def read_images(self, img_name, label_name):
        """
        Used within TF DatasetAPI, loads image and label
        """
        image_string = tf.read_file(img_name)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        label_string = tf.read_file(label_name)
        label_decoded = tf.image.decode_jpeg(label_string, channels=1)
        return image_decoded, label_decoded

    def create_label(self, loaded_img, loaded_label):
        """
        Creates label by hand binary image extraction. Part of TF Dataset pipeline.
        Requires uint8 img and label, returns img and label in uin8 format
        """
        _, label = cv2.threshold(loaded_label, 127, 255, cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        label = cv2.dilate(label, kernel, iterations=1)
        _, contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            x, y, w, h = cv2.boundingRect(contours[np.argmax(areas)])
            label = label[y:y + h, x:x + w]
            return loaded_img.astype(np.float32) / 255, cv2.resize(label, (self.label_w, self.label_h)).astype(np.float32) / 255
        else:
            return loaded_img.astype(np.float32) / 255, np.zeros([self.label_h, self.label_w], dtype=np.float32)

    def get_data(self):
        # training dataset
        train_image_names = tf.constant(self.img_names[:int(len(self.img_names) * 0.95)])
        train_label_names = tf.constant(self.label_names[:int(len(self.label_names) * 0.95)])

        training_dataset = tf.data.Dataset.from_tensor_slices((train_image_names, train_label_names))
        training_dataset = training_dataset.shuffle(buffer_size=500000)
        training_dataset = training_dataset.map(self.read_images, num_parallel_calls=4)
        training_dataset = training_dataset.map(
            lambda img, label: tuple(tf.py_func(self.create_label, [img, label], [tf.float32, tf.float32], stateful=False)),
            num_parallel_calls=4)
        training_dataset = training_dataset.prefetch(self.batch_size * 10)
        training_dataset = training_dataset.batch(self.batch_size)

        # val dataset
        val_image_names = tf.constant(self.img_names[int(len(self.img_names) * 0.95):])
        val_label_names = tf.constant(self.label_names[int(len(self.label_names) * 0.95):])

        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_names, val_label_names))
        val_dataset = val_dataset.shuffle(buffer_size=500000)
        val_dataset = val_dataset.map(self.read_images, num_parallel_calls=4)
        val_dataset = val_dataset.map(
            lambda img, label: tuple(tf.py_func(self.create_label, [img, label], [tf.float32, tf.float32], stateful=False)),
            num_parallel_calls=4)
        val_dataset = val_dataset.prefetch(self.batch_size * 3)
        val_dataset = val_dataset.batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
        images, labels = iterator.get_next()

        training_dataset_init = iterator.make_initializer(training_dataset)
        val_dataset_init = iterator.make_initializer(val_dataset)

        return training_dataset_init, val_dataset_init, images, labels
