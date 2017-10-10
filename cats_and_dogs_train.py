from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import inception_train
from cats_and_dogs_data import CatsAndDogsData

FLAGS =  tf.app.flags.FLAGS

def main(_):

    dataset = CatsAndDogsData(subset=FLAGS.subset)
    assert dataset.data_files()

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)

    tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_train.train(dataset)

if __name__ == "__main__":
    tf.app.run()
