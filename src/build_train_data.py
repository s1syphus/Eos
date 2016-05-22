"""
    This needs to be written at some point


    Mostly sourced from here:

    https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py
"""

import numpy as np
import tensorflow as tf
import os
import sys
import random

from datetime import datetime

import threading

# Flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_directory', '../data/Flowers',
                           'Training data directory')

tf.app.flags.DEFINE_string('output_directory', '../data/sharded_data/Flowers/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 24,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads available.')


# These help transform features to tensorflow features
def _int64_feature(value):
    """
        Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """
        Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """
        Wrapper for inserting bytes features into Example proto.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# This reads in the raw data with no changes
def _convert_to_example(filename, image_buffer, label, height, width):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _bytes_feature(label),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)
    }))
    return example


class ImageCoder(object):
    """
        Helper class that provides TensorFlow image coding utilities
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb, feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """
        Process a single image file.

        Args:
            filename: string, path to an image file e.g., '/path/to/example.JPG'.
            coder: instance of ImageCoder to provide TensorFlow image coding utils.
        Returns:
            image_buffer: string, JPEG encoding of RGB image.
            height: integer, image height in pixels.
            width: integer, image width in pixels.
    """
    # Read the image file.
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, labels, num_shards):
    """
        Processes and saves list of images as TFRecord in 1 thread.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            image_buffer, height, width = _process_image(filename, coder)
            example = _convert_to_example(filename, image_buffer, label, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
        print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, labels, num_shards):
    """
         Process and save list of images as TFRecord of Example protos.
    """
    assert len(filenames) == len(labels)
    # Break all images into batches with a [ranges[i][0], ranges[i][1]]
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []

    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []

    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir):
    """
        Builds a list of all image files in a directory
    """
    print('Determining list of input files and labels from %s.' % data_dir)

    folders = os.listdir(data_dir)
    filenames = []
    labels = []
    # Leave label index 0 empty as a background class.
    label_index = 1

    for folder in folders:
        matching_files = tf.gfile.Glob(data_dir + '/' + str(folder) + '/*')
        labels.extend([folder] * len(matching_files))
        filenames.extend(matching_files)
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(folders), data_dir))

    return filenames, labels


def _process_dataset(name, directory, num_shards):
    """
        Turns a directory into shards of TF records
    """
    filenames, labels = _find_image_files(directory)
    _process_image_files(name, filenames, labels, num_shards)


def main(unused_argv):
    _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards)


if __name__ == '__main__':
    tf.app.run()