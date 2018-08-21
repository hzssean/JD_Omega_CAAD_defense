"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time

import numpy as np
from scipy.misc import imread

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2

from watchdog.observers import Observer
from watchdog.events import *
from category.category import CategoryHelper

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images created.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def load_one_image(src_file, batch_shape):
    """Read one png image from input directory in batches.

    Args:
      src_file: new image file
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    with tf.gfile.Open(src_file) as f:
        image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(src_file))
    idx += 1
    if idx == batch_size:
        yield filenames, images
        filenames = []
        images = np.zeros(batch_shape)
        idx = 0
    if idx > 0:
        yield filenames, images


class FileEventHandler(FileSystemEventHandler):
    def __init__(self, batch_shape, sess, x_input, predicted_labels, gradients, output_dir):
        FileSystemEventHandler.__init__(self)
        self._batch_shape = batch_shape
        self._sess = sess
        self._x_input = x_input
        self._predicted_labels = predicted_labels
        self._gradients = gradients
        self._output_dir = output_dir
        self._category_helper = CategoryHelper("category/categories.csv")

    def on_moved(self, event):
        if event.is_directory:
            print("directory moved from {0} to {1}".format(event.src_path, event.dest_path))
        else:
            print("file moved from {0} to {1}".format(event.src_path, event.dest_path))

    def _defense_for_img_created(self, img_file):
        """ defense one image: xxx.png,
            write res to xxx.txt with two line(lable human_string),
            copy the src image file to output dir then delete it
        :param img_file:
        :return None:
        """
        if img_file.endswith('.png'):
            output_file_name = ""
            for filenames, images in load_one_image(img_file, self._batch_shape):
                #labels = self._sess.run(self._predicted_labels, feed_dict={self._x_input: images})
                a = self._sess.run(self._gradients, feed_dict={self._x_input: images})
                labels = self._sess.run(self._predicted_labels, {self._x_input: images - a[0]})

                for filename, label in zip(filenames, labels):
                    res_file_name = os.path.basename(filename)[:-4] + '.txt'
                    output_file_name = os.path.join(self._output_dir, filename)
                    print("res_file_name: " + res_file_name)
                    with open(os.path.join(self._output_dir, res_file_name), 'w+') as res_file:
                        res_file.write('{0}\n{1}\n'.format(label,
                                                           self._category_helper.get_category_name(label)))
                        res_file.flush()
            if os.path.exists(output_file_name):
                os.remove(output_file_name)
            shutil.copy(img_file, output_file_name)
            os.remove(img_file)

    def on_created(self, event):
        if event.is_directory:
            print("directory created:{0}".format(event.src_path))
        else:
            print("file created:{0}".format(event.src_path))
            self._defense_for_img_created(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            print("directory deleted:{0}".format(event.src_path))
        else:
            print("file deleted:{0}".format(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            print("directory modified:{0}".format(event.src_path))
        else:
            print("file modified:{0}".format(event.src_path))


def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

        y_logits = tf.reduce_max(end_points['Predictions'])
        predicted_labels = tf.argmax(end_points['Predictions'], 1)
        gradients = tf.gradients(y_logits, x_input)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2')) # only train InceptionResnetV2
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            # with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
            # for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            #     labels = sess.run(predicted_labels, feed_dict={x_input: images})
            #     for filename, label in zip(filenames, labels):
            #         out_file.write('{0},{1}\n'.format(filename, label))

            ''' watch the input dir for defense '''
            observer = Observer()
            event_handler = FileEventHandler(batch_shape=batch_shape,
                                             sess=sess,
                                             x_input=x_input,
                                             predicted_labels=predicted_labels,
                                             gradients = gradients,
                                             output_dir=FLAGS.output_dir)

            observer.schedule(event_handler, FLAGS.input_dir, recursive=True)
            observer.start()

            print("watchdog start...")

            try:
                while True:
                    time.sleep(0.000001)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

            print("\nwatchdog stoped!")


if __name__ == '__main__':
    tf.app.run()
