from WADEditor import WADReader
import tensorflow as tf
import os, json

def read_json(scraped_path):
  json_path = scraped_path + 'doom.json'
  scraped_ids = list()
  if os.path.isfile(json_path):
    print('Found scraped info...')
    with open(json_path, 'r') as jsonfile:
      scraped_info = json.load(jsonfile)
      print('Loaded {} records.'.format(len(scraped_info)))
      if len(scraped_info) == 0:
        print('No scraped info present')
        return None
      else:
        scraped_ids = [info['id'] for info in scraped_info if 'id' in info]
        return scraped_ids
  else:
    print('JSON not found')
    return None

def parse_wads(wad_ids, dataset_path):
  feature_maps = list()
  reader = WADReader()
  valid_dataset = list()
  for id in wad_ids:
    wad_path = dataset_path + id + "/"
    # Listing all files in directories that have wad_id as their name
    for file in os.listdir(wad_path):
      if file.endswith(".WAD") or file.endswith(".wad"):
        try:
          wad_with_features = reader.extract(wad_path+file)
          if wad_with_features is not None and not wad_with_features['wad']['exception']:
            levels = wad_with_features["levels"]
            print('added level', id, 'from', file)
            # feature_map = levels[0]["maps"]
            # feature_map.update(levels[0]["features"])
            # feature_maps.append(feature_map)
            feature_maps.append(levels[0]["maps"])
            valid_dataset.append({'id':id, 'name':file})
            break
        except:
            print('failed to parse level',id)
            continue

  return feature_maps, valid_dataset

def metadata_gen(wad_list,keys,save_path):
  
  map_meta = { "thingsmap": {"type": "uint8", "min": 0.0, "max": 119.0}, 
  "heightmap": {"type": "uint8", "min": 0.0, "max": 255.0}, 
  "floortexturemap": {"type": "uint8", "min": 0.0, "max": 182.0}, 
  "ceilingtexturemap": {"type": "uint8", "min": 0.0, "max": 182.0}, 
  "rightwalltexturemap": {"type": "uint8", "min": 0.0, "max": 506.0}, 
  "leftwalltexturemap": {"type": "uint8", "min": 0.0, "max": 506.0}}
  map_dict = dict()
  map_dict['wads'] = {wad['id']: i for i,wad in enumerate(wad_list)}
  map_dict['count'] = len(wad_list)
  key_meta = {key: map_meta[key] for key in map_meta if key in keys}
  map_dict['key_meta'] = key_meta

  if not os.path.exists(save_path):
    os.makedirs(save_path)

  file = save_path + 'metadata.json'
  with open(file, 'w') as jsonfile:
    json.dump(map_dict, jsonfile)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


# Write TFrecord file
def generate_tfrecord(maps,key_prefs,save_path):
  file_name = 'dataset.tfrecords'
  file_path = save_path + file_name
  
  serialized_array = dict()
  with tf.io.TFRecordWriter(file_path) as writer:
    for map in maps:
      for key in key_prefs:
        serialized_array[key] = serialize_array(map[key])
      feature = {key: _bytes_feature(serialized_array[key]) for key in key_prefs}
      example_message = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example_message.SerializeToString())


def doom_parser(wads_path,save_path,keys):
  wad_ids = read_json(wads_path)
  feature_maps, wads = parse_wads(wad_ids, wads_path)
  metadata_gen(wads,keys,save_path)
  generate_tfrecord(feature_maps,keys,save_path)



dataset_path = '../dataset/scraped/doom/'
save_path = '../dataset/parsed/doom/'
keys = ['thingsmap', 'heightmap', 'floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap', 'leftwalltexturemap']
doom_parser(dataset_path,save_path,keys)