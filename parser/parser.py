from WADEditor import WADReader
import tensorflow as tf
import os, json, sys, math
from matplotlib import pyplot as plt
import Dictionaries.ThingTypes as ThingTypes
import numpy as np

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
  maps_meta = dict()
  maps_meta['max_rooms'] = 0
  for id in wad_ids:
    wad_path = dataset_path + id + "/"
    # Listing all files in directories that have wad_id as their name
    for file in os.listdir(wad_path):
      if file.endswith(".WAD") or file.endswith(".wad"):
        try:
          wad_with_features = reader.extract(wad_path+file)
          if wad_with_features is not None and not wad_with_features['wad']['exception']:
            levels = wad_with_features["levels"]
            features = levels[0]["features"]
            if features["has_teleporters"] and features["number_of_monsters"]>0 and features["number_of_weapons"]>0 and features["start_location_x_px"] != -1 and features["start_location_y_px"] != -1:
              # Adding extracted features with the map
              # feature_map = levels[0]["maps"]
              # feature_map.update(levels[0]["features"])
              # feature_maps.append(feature_map)
              map = levels[0]["maps"]["roommap"]
              if map.max() > maps_meta['max_rooms']: maps_meta['max_rooms'] = map.max()
              feature_maps.append(levels[0]["maps"])
              valid_dataset.append({'id':id, 'name':file})
              print('added level', id, 'from', file)
              break
        except:
          print('failed to add level', id, 'from', file)
    # if len(valid_dataset) >= 12: break
  return feature_maps, valid_dataset, maps_meta

def metadata_gen(wad_list,map_keys,cate_pref,save_path,meta,graphics_meta):

  map_meta = dict()
  map_meta["floormap"] = {"type": "uint8", "min": 0.0, "max": 255.0}
  map_meta["wallmap"] = {"type": "uint8", "min": 0.0, "max": 255.0}
  map_meta["heightmap"] = {"type": "uint8", "min": 0.0, "max": 255.0} 
  map_meta["triggermap"] = {"type": "uint8", "min": 0.0, "max": 255.0}
  map_meta["roommap"] = {"type": "uint8", "min": 0.0, "max": int(meta['max_rooms'])}
  map_meta["floortexturemap"] = {"type": "uint8", "min": 0.0, "max": graphics_meta["flats"]}
  map_meta["ceilingtexturemap"] =  {"type": "uint8", "min": 0.0, "max": graphics_meta["flats"]}
  map_meta["rightwalltexturemap"] = {"type": "uint8", "min": 0.0, "max": graphics_meta["textures"]}
  map_meta["leftwalltexturemap"] = {"type": "uint8", "min": 0.0, "max": graphics_meta["textures"]}
  map_meta["thingsmap"] = {"type": "uint8", "min": 0.0, "max": 123.0}

  obj = dict()
  cate = ThingTypes.get_all_categories()
  for cat in cate:
    obj[cat] = ThingTypes.get_index_by_category(cat)
  map_meta["start"] = {"type": "uint8", "min": obj["start"][0], "max": obj["start"][-1]}
  map_meta["other"]=  {"type": "uint8", "min":obj["other"][0], "max": obj["other"][-1]}
  map_meta["keys"] = {"type": "uint8", "min": obj["keys"][0], "max": obj["keys"][-1]}
  map_meta["decorations"] = {"type": "uint8", "min": obj["decorations"][0], "max": obj["decorations"][-1]}
  map_meta["obstacles"] = {"type": "uint8", "min": obj["obstacles"][0], "max": obj["obstacles"][-1]}
  map_meta["monsters"] = {"type": "uint8", "min": obj["monsters"][0], "max": obj["monsters"][-1]}
  map_meta["ammunitions"] = {"type": "uint8", "min": obj["ammunitions"][0], "max": obj["ammunitions"][-1]}
  map_meta["weapons"] = {"type": "uint8", "min": obj["weapons"][0], "max": obj["weapons"][-1]}
  map_meta["powerups"] = {"type": "uint8", "min": obj["powerups"][0], "max": obj["powerups"][-1]}
  map_meta["artifacts"] = {"type": "uint8", "min": obj["artifacts"][0], "max": obj["artifacts"][-1]}

  obj['essentials'] = ThingTypes.get_index_by_category('essentials')
  map_meta["essentials"] = {"type": "uint8", "min": obj['essentials'][0], "max": obj['essentials'][-1]}


  map_dict = dict()
  map_dict['wads'] = {wad['id']: i for i,wad in enumerate(wad_list)}
  map_dict['count'] = len(wad_list)
  keys = map_keys + cate_pref
  key_meta = {key: map_meta[key] for key in keys}
  map_dict['maps_meta'] = key_meta

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

# Created as it is unable to do it inline 
def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


# pads the maps to show relatve scale, needs addition of image size recification to take into consideration
def map_padding(mat, keys, cate):
  organized_map = dict()
  for key in keys:
    if key == 'thingsmap':
      organized_map[key] = mat[key][key] # if not pad else np.pad(mat[key][key], ((x, x), (y, y)), 'constant')
      for cat in cate:
        organized_map[cat] = mat[key][cat] # if not pad else np.pad(mat[key][cat], ((x, x), (y, y)), 'constant')
    else:
      organized_map[key] = mat[key] # if not pad else np.pad(mat[key], ((x, x), (y, y)), 'constant')
  return organized_map


# Write TFrecord file
def generate_tfrecord(maps,keys_pref,cate_pref,save_path):
  file_name = 'data.tfrecords'
  file_path = save_path + file_name
  keys = keys_pref+cate_pref
  with tf.io.TFRecordWriter(file_path) as writer:
    for map in maps:
      org_maps = map_padding(map,keys_pref,cate_pref)
      serialized_array = {key: serialize_array(org_maps[key]) for key in keys}
      feature = {key: _bytes_feature(serialized_array[key]) for key in keys}
      example_message = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example_message.SerializeToString())
  print("tf record written successfully")


def doom_parser(wads_path,keys,cate,save_path):
  file_path = save_path + 'graphics.json'
  if os.path.isfile(file_path):
    with open(file_path, 'r') as jsonfile:
        graphics = json.load(jsonfile)
  else:
    print('generate graphics json first')
    sys.exit()
  wad_ids = read_json(wads_path)
  feature_maps, wad_list, meta = parse_wads(wad_ids, wads_path)
  generate_tfrecord(feature_maps,keys,cate,save_path)
  metadata_gen(wad_list,keys,cate,save_path,meta,graphics["meta"])


dataset_path = '../dataset/scraped/doom/'
save_path = '../dataset/parsed/doom/'
# All the generated maps are ['thingsmap', 'floormap', 'wallmap', 'heightmap', 'triggermap', 'roommap', 'floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap', 
# 'leftwalltexturemap'] with things map containing ['start','other', 'keys', 'decorations', 'obstacles',  'monsters', 'ammunitions', 'weapons', 'powerups', 'artifacts']
keys = ['floormap', 'wallmap', 'heightmap', 'triggermap', 'roommap', 'thingsmap', 'floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap','leftwalltexturemap']
things_cate = ['essentials','start', 'other', 'keys','obstacles',  'monsters', 'ammunitions', 'weapons', 'powerups', 'artifacts']
doom_parser(dataset_path,keys,things_cate,save_path)
