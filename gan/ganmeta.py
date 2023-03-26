import tensorflow as tf
import sys, os, json, random
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def read_json(save_path = '../dataset/parsed/doom/'):
    file_path = save_path + 'metadata.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as jsonfile:
            map_meta = json.load(jsonfile)
        return map_meta
    else:
        print('No metadata found')
        sys.exit()

def read_record(batch_size=32, save_path='../dataset/parsed/doom/',sample_wgan=False): 
    file_path = save_path + 'data.tfrecords'
    metadata = read_json()
    if not os.path.isfile(file_path):
        print('No dataset record found')
        sys.exit()
    tfr_dataset = tf.data.TFRecordDataset(file_path)
    sample = parse_tfrecord(tfr_dataset,metadata,sample_wgan)
    train_set = tfr_dataset.map(_parse_tfr_element)
    # Returns a shuffled training set that is seperated into batches
    return train_set.shuffle(metadata['count']*100).batch(batch_size, drop_remainder=True), metadata['maps_meta'], sample


# Read TF Records and View the scaled maps
def parse_tfrecord(record, meta, sample_wgan):
    # If adding the wgan generated maps as the sample
    if sample_wgan:
        save_path = '../dataset/generated/doom/hybrid/'
        file_path = save_path + 'sample.tfrecords'
        if not os.path.isfile(file_path):
            print('No dataset record found')
            sys.exit()
        record = tf.data.TFRecordDataset(file_path)
        map_keys = ['floormap','wallmap','heightmap']
        sample_id = 0
    else:
        map_keys = list(meta['maps_meta'].keys())
        sample_id = random.randrange(meta['count'])
    parse_dic = {
        key: tf.io.FixedLenFeature([], tf.string) for key in map_keys
        }
    features = dict()
    for i,element in enumerate(record):
        if i != sample_id:
            continue
        else:
            example_message = tf.io.parse_single_example(element, parse_dic)
            for key in map_keys:
                b_feature = example_message[key] # get byte string
                feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
                features[key] = feature
            break
    return features


# Read the TF Records
def _parse_tfr_element(element):
    metadata = read_json()
    map_keys = list(metadata['maps_meta'].keys())
    parse_dic = {
        key: tf.io.FixedLenFeature([], tf.string) for key in map_keys
        }
    example_message = tf.io.parse_single_example(element, parse_dic)
    features = dict()
    for key in map_keys:
        b_feature = example_message[key] # get byte string
        feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
        features[key] = feature
    return features


def scaling_maps(x, map_meta, map_names, use_sigmoid=True):
    """
    Compute the scaling of eve ry map based on their .meta statistics (max and min)
     to bring all values inside (0,1) or (-1,1)
    :param x: the input vector. shape(batch, width, height, len(map_names))
    :param map_names: the name of the feature maps for .meta file lookup
    :param use_sigmoid: if True data will be in range 0,1, if False it will be in -1;1
    :return: a normalized x vector
    """
    a = 0 if use_sigmoid else -1
    b = 1
    max = tf.constant([map_meta[m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    min = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    min_mat = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.cast(tf.cast(x,tf.bool),tf.float32)
    return a + ((x-min_mat)*(b-a))/(max-min)


def rescale_maps(x, map_meta, map_names, use_sigmoid=True):
    a = 0 if use_sigmoid else -1
    b = 1
    min_mat = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.cast(tf.cast(x,tf.bool),tf.float32)
    min = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    max = tf.constant([map_meta[m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    return tf.math.round(((max-min) * (x - a)/(b-a)) + min_mat)


def generate_loss_graph(d_loss, g_loss, location = 'generated_maps/'):
    plt.figure()
    plt.xlabel('Number of Steps')
    plt.ylabel('Loss')
    plt.plot(d_loss)
    # plt.legend() # must be after labels
    plt.savefig(location+'disc_loss_graph')
    plt.close()

    plt.figure()
    plt.xlabel('Number of Steps')
    plt.ylabel('Loss')
    plt.plot(g_loss)
    # plt.legend() # must be after labels
    plt.savefig(location+'gen_loss_graph')
    plt.close()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Created as it is unable to do it inline 
def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def generate_sample(imgs ,keys, save_path = '../dataset/generated/doom/', file_name = 'sample.tfrecords'):
    file_path = save_path + file_name
    with tf.io.TFRecordWriter(file_path) as writer:
        gen_maps = dict()
        for i in range(len(keys)):
            gen_maps[keys[i]] = imgs[0, :, :, i]
        serialized_array = {key: serialize_array(gen_maps[key]) for key in keys}
        feature = {key: _bytes_feature(serialized_array[key]) for key in keys}
        example_message = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_message.SerializeToString())
    file = save_path + 'meta.json'
    with open(file, 'w') as jsonfile:
        json.dump({'keys':keys}, jsonfile)