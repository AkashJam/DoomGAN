from WADEditor import WADReader
from matplotlib import pyplot as plt
import tensorflow as tf
import os

reader = WADReader()

wad_with_features = reader.extract("../dataset/scraped/doom/crusades/CRUSADES.WAD")
wad = wad_with_features["wad"]
levels = wad_with_features["levels"]
features = levels[0]["features"]
maps = levels[0]["maps"]

keypref = ['thingsmap', 'heightmap', 'floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap', 'leftwalltexturemap']

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


# Write TFrecord file
save_path = '../dataset/parsed/doom/'
file_name = 'data.tfrecords'
file_path = save_path + file_name
if os.path.exists(save_path):
  if os.path.isfile(file_path):
    print('found location of parsed files')
else:
    os.makedirs(save_path)
serialized_array = dict()
with tf.io.TFRecordWriter(file_path) as writer:
  for key in keypref:
    serialized_array[key] = serialize_array(maps[key])
  feature = {key: _bytes_feature(serialized_array[key]) for key in keypref}
  example_message = tf.train.Example(features=tf.train.Features(feature=feature))
  writer.write(example_message.SerializeToString())

# Read TFRecord file
def _parse_tfr_element(element):
  parse_dic = {
    key: tf.io.FixedLenFeature([], tf.string) for key in keypref # Note that it is tf.string, not tf.float32
    }
  example_message = tf.io.parse_single_example(element, parse_dic)
  features = dict()
  for key in keypref:
    b_feature = example_message[key] # get byte string
    feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
    features[key] = feature
  return features


tfr_dataset = tf.data.TFRecordDataset(file_path)
dataset = tfr_dataset.map(_parse_tfr_element)
print(dataset)
for instance in dataset:
  print(instance)