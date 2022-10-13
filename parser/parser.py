from tkinter import E
from WADEditor import WADReader
from matplotlib import pyplot as plt
import tensorflow as tf
import os, json

dataset_path = '../dataset/scraped/doom/'
json_path = dataset_path + 'doom.json'
save_path = '../dataset/parsed/doom/'
file_name = 'data.tfrecords'
file_path = save_path + file_name
reader = WADReader()

scraped_ids = list()
if os.path.isfile(json_path):
  print('Collecting scraped info...')
  with open(json_path, 'r') as jsonfile:
    scraped_info = json.load(jsonfile)
    print('Loaded {} records.'.format(len(scraped_info)))
    if len(scraped_info) != 0:
      scraped_ids = [info['id'] for info in scraped_info if 'id' in info]
    else:
      print('no scraped info present')
else:
  print('No files been scraped')


feature_maps = list()
for id in scraped_ids:
  wad_path = dataset_path + id + "/"
  # Listing all files in directories that have wad_id as their name
  for file in os.listdir(wad_path):
    if file.endswith(".WAD") or file.endswith(".wad"):
      try:
        wad_with_features = reader.extract(wad_path+file)
        if wad_with_features != None:
          levels = wad_with_features["levels"]
          print('added level',id)
          # feature_map = levels[0]["maps"]
          # feature_map.update(levels[0]["features"])
          # feature_maps.append(feature_map)
          feature_maps.append(levels[0]["maps"])
          break
      except:
          print('failed to parse level',id)
          continue
  break

# print('number of levels', len(feature_maps))

keypref = ['thingsmap', 'heightmap', 'floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap', 'leftwalltexturemap', 'triggermap']

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


# Write TFrecord file
if os.path.exists(save_path):
  if os.path.isfile(file_path):
    print('found location of parsed files')
else:
    os.makedirs(save_path)
serialized_array = dict()
with tf.io.TFRecordWriter(file_path) as writer:
  for maps in feature_maps:
    for key in keypref:
      serialized_array[key] = serialize_array(maps[key])
    feature = {key: _bytes_feature(serialized_array[key]) for key in keypref}
    example_message = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example_message.SerializeToString())

