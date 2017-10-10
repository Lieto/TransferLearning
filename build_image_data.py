from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string("train_directory", "/media/kuoppves/My Passport/tmp/data/train/", "Training data directory")
tf.app.flags.DEFINE_string("validation_directory", "/media/kuoppves/My Passport/tmp/data/val/", "Validation data directory")
tf.app.flags.DEFINE_string("output_directory", "/media/kuoppves/My Passport/tmp/data/", "Output data directory")
tf.app.flags.DEFINE_integer("train_shards", 2, "Number of shards in training TFRecord files.")
tf.app.flags.DEFINE_integer("validation_shards", 2, "Number of shards in validation TFRecord files.")
tf.app.flags.DEFINE_integer("num_threads", 2, "Number of threads to preprocess the images.")

tf.app.flags.DEFINE_string("labels_file", "./labels.txt", "Labels file")

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):

    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, label, text, height, width):

    colorspace = "RGB"
    channels = 3
    image_format = "JPEG"

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))


    return example

class ImageCoder(object):

    def __init__(self):

        self._sess = tf.Session()

        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format="rgb", quality=100)

        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):

        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):

    return filename.endswith(".png")

def _process_image(filename, coder):

    with tf.gfile.FastGFile(filename, "rb") as f:
        image_data = f.read()

    if _is_png(filename):

        print("Converting PNG to JPEG for {}".format(filename))
        image_data = coder.png_to_jpeg(image_data)

    image = coder.decode_jpeg(image_data)

    assert len(image.shape) == 3

    height = image.shape[0]
    width = image.shape[1]

    assert image.shape[2] == 3

    return image_data, height, width

def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
    num_threads = len(ranges)
    assert not num_shards % num_threads

    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)

    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0

    for s in range(num_shards_per_batch):

        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)

        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print("SKIPPED: Unexpected error while decoding {}".format(filename))
                continue

            example = _convert_to_example(filename, image_buffer, label, text, height, width)

            writer.write(example.SerializeToString())

            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print("{} [thread: {}]: Processes {} of {} images in thread batch.".format(
                    datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()

        print("{} [thread {}]: Wrote {} images to {}".format(datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0

    print("{} [thread {}]: Wrote {} images to {} shards.".format(
        datetime.now(), thread_index, counter, num_files_in_thread
    ))
    sys.stdout.flush()

def _process_image_files(name, filenames, texts, labels, num_shards):

    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []

    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    print("Launching {} threads for spacings: {}".format(FLAGS.num_threads, ranges))
    sys.stdout.flush()

    coord = tf.train.Coordinator()

    coder = ImageCoder()

    threads = []

    for thread_index in range(len(ranges)):

        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards)

        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)

    print("{}: Finished writing all {} images in data set.".format(datetime.now(), len(filenames)))

    sys.stdout.flush()




def _find_image_files(data_dir, labels_file):

    print("Determining list of input files and labels from {}".format(data_dir))
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]


    filenames = []
    labels = []
    texts = []

    label_index = 1

    for text in unique_labels:
        jpeg_file_path = "%s/%s/*" % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print("Finished finding files in {} of {} classes.".format(label_index, len(labels)))

        label_index += 1

    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print("Found {} JPEG files across {} labels inside {}".format(len(filenames), len(unique_labels), data_dir))

    return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file):

    filenames, texts, labels = _find_image_files(directory, labels_file)
    _process_image_files(name, filenames, texts, labels, num_shards)

def main(unused_argv):

    assert not FLAGS.train_shards % FLAGS.num_threads, (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert  not FLAGS.validation_shards % FLAGS.num_threads, (
        "Please make the FLAGS.num_threads commensurate with FLAGS.validation_shards"
    )

    print("Saving results to {}".format(FLAGS.output_directory))

    # Run it
    _process_dataset("validation", FLAGS.validation_directory,
                     FLAGS.validation_shards, FLAGS.labels_file)

    _process_dataset("train", FLAGS.train_directory,
                     FLAGS.train_shards, FLAGS.labels_file)

if __name__ == "__main__":

    tf.app.run()