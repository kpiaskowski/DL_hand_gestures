import csv
import itertools
import os
import pickle
import tensorflow as tf

class DataProvider:
    def __init__(self, labels_root='../labels', data_root='../data'):
        self.labels_root = labels_root
        self.data_root = data_root
        self.val_subjects = ['Subject38', 'Subject39']

        self.used_subjects, self.subject_mapping = self.get_used_subjects(data_root)

        t_images, _, v_images, _ = pickle.load(open('filenames.p', 'rb'))
        self.train_images, self.train_labels, self.val_images, self.val_labels = self.annotate_images(t_images, v_images)

    def get_used_subjects(self, data_root):
        used_subjects = [os.listdir(os.path.join(data_root, parent_dir)) for parent_dir in os.listdir(data_root)]
        used_subjects = list(itertools.chain(*used_subjects))

        subject_mapping = {}
        for parent_dir in os.listdir(data_root):
            for subject in os.listdir(os.path.join(data_root, parent_dir)):
                subject_mapping[subject] = parent_dir
        return used_subjects, subject_mapping

    def annotate_images(self, t_images, v_images):
        """
        Creates annotation for each image used for vae training
        """
        train_data = dict(zip(t_images, [0] * len(t_images)))
        val_data = dict(zip(v_images, [0] * len(v_images)))

        for dirName, subdirList, fileList in os.walk(self.labels_root):
            if any(subj.lower() in dirName for subj in self.used_subjects) and 'Scene' in dirName:
                files = [os.path.join(dirName, fname) for fname in fileList]
                for file in files:
                    if not '.csv' in file:
                        continue
                    matching_imgname = file.replace('../labels/s', 'S').replace('Group', 'Color/rgb').replace('.csv', '/')
                    prefix = self.data_root + '/' + self.subject_mapping[matching_imgname.split('/')[0]] + '/'
                    matching_imgname = prefix + matching_imgname
                    with open(file, newline='') as csvfile:
                        spamreader = csv.reader(csvfile)
                        for row in spamreader:
                            class_id = int(row[0])
                            for i in range(int(row[1]), int(row[2]) + 1):
                                name = matching_imgname + str(i).zfill(6) + '.jpg'
                                if name in train_data.keys():
                                    train_data[name] = class_id
                                elif name in val_data.keys():
                                    val_data[name] = class_id

        train_images, train_labels = zip(*sorted(train_data.items()))
        val_images, val_labels = zip(*sorted(train_data.items()))
        return train_images, train_labels, val_images, val_labels

    # def get_data(self):
    #     # training dataset
    #     train_image_names = tf.constant(self.train_images)
    #     train_label_names = tf.constant(self.train_labels)
    #
    #     training_dataset = tf.data.Dataset.from_tensor_slices((train_image_names, train_label_names))
    #     training_dataset = training_dataset.shuffle(buffer_size=500000)
    #     training_dataset = training_dataset.map(self.read_images, num_parallel_calls=4)
    #     training_dataset = training_dataset.map(
    #         lambda img, label: tuple(tf.py_func(self.create_label, [img, label], [tf.float32, tf.float32], stateful=False)),
    #         num_parallel_calls=4)
    #     training_dataset = training_dataset.prefetch(self.batch_size)
    #     training_dataset = training_dataset.batch(self.batch_size)
    #
    #     # # val dataset
    #     # val_image_names = tf.constant(self.v_img_names)
    #     # val_label_names = tf.constant(self.v_label_names)
    #     #
    #     # val_dataset = tf.data.Dataset.from_tensor_slices((val_image_names, val_label_names))
    #     # val_dataset = val_dataset.shuffle(buffer_size=500000)
    #     # val_dataset = val_dataset.map(self.read_images, num_parallel_calls=4)
    #     # val_dataset = val_dataset.map(
    #     #     lambda img, label: tuple(tf.py_func(self.create_label, [img, label], [tf.float32, tf.float32], stateful=False)),
    #     #     num_parallel_calls=4)
    #     # val_dataset = val_dataset.prefetch(self.batch_size)
    #     # val_dataset = val_dataset.batch(self.batch_size)
    #
    #     iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
    #     images, labels = iterator.get_next()
    #
    #     training_dataset_init = iterator.make_initializer(training_dataset)
    #     # val_dataset_init = iterator.make_initializer(val_dataset)
    #
    #     return training_dataset_init,
    #     # return training_dataset_init, val_dataset_init, images, labels

a = DataProvider()
