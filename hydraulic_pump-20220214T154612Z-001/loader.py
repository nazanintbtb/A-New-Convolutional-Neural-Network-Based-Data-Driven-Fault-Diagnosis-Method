# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import scipy.io as sio

def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature([3], tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64)
    }
 
    example = tf.io.parse_single_example(serialized_example,feature_description)
    
    image = tf.io.parse_tensor(example['image'], out_type = float)
    image_shape = [example['height'], example['width'],example['depth']]
    image = tf.reshape(image, image_shape)
    
    image=image/255.0
    label = tf.reshape(example['label'], [3])
    label = tf.cast(label, tf.int32)
       
    return image, label



def get_train_data(batch_size):
    filenames = ['./traindata.tfrecords']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(map_func=read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset 

def get_valid_data(batch_size):
    filenames = ['./validdata.tfrecords']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=10000)
    dataset = dataset.map(map_func=read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset 

def get_test_data(batch_size):
    filenames = ['./testdata.tfrecords']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=10000)
    dataset = dataset.map(map_func=read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset 