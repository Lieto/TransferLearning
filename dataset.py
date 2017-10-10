from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_dir", "/media/kuoppves/My Passport/tmp/data", "Path to the processes data, i.e TFRecord of "
                                                                          "Example protos.")

class Dataset(object):

    __metaclass__ = ABCMeta

    def __init__(self, name, subset):

        assert subset in self.available_subsets(), self.available_subsets()
        self.name = name
        self.subset = subset

    @abstractmethod
    def num_classes(self):

        pass

    @abstractmethod
    def num_examples_per_epoch(self):

        pass

    @abstractmethod
    def download_message(self):
        pass

    def available_subsets(self):

        return ["train", "validation"]

    def data_files(self):

        tf_record_pattern = os.path.join(FLAGS.data_dir, "%s-*" % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)

        if not data_files:
            print("No files found for dataset {}/{} at {}".format(
                self.name,
                self.subset,
                FLAGS.data_dir
            ))

            self.download_message()
            exit(-1)

        return data_files

    def reader(self):

        return tf.TFRecordReader()

