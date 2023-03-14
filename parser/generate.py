from WADEditor import WADWriter
import tensorflow as tf
import numpy as np
import os, sys

def parse_tfrecord(record,map_keys):
    parse_dic = {
        key: tf.io.FixedLenFeature([], tf.string) for key in map_keys
        }
    features = dict()
    for element in record:
        example_message = tf.io.parse_example(element, parse_dic)
        for key in map_keys:
            b_feature = example_message[key] # get byte string
            feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
            features[key] = feature
    return features

def read_record(keys,save_path='../dataset/generated/doom/'): 
    file_path = save_path + 'test.tfrecords'
    if not os.path.isfile(file_path):
        print('No dataset record found')
        sys.exit()
    tfr_dataset = tf.data.TFRecordDataset(file_path)
    sample_input = parse_tfrecord(tfr_dataset,keys)
    return sample_input


key_pref = ['floormap','wallmap','heightmap','essentials']
maps = read_record(key_pref)
writer = WADWriter()
writer.add_level('MAP01')
writer.from_images(maps['floormap'],maps['heightmap'],maps['wallmap'],maps['essentials'])
wad_mine = writer.save('test.wad')

