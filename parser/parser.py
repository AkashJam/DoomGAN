from WADEditor import WADReader
import tensorflow as tf
import os, json, sys, math
from matplotlib import pyplot,colors,ticker
import Dictionaries.ThingTypes as ThingTypes
import numpy as np
from skimage.morphology import label

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
  
def plot_dims(widths, heights):
  fig, ax = pyplot.subplots(tight_layout=True)
  pyplot.xlabel('Map Width in DOOM Units')
  pyplot.ylabel('Map Length in DOOM Units')
  sdh = ax.hexbin(widths, heights, gridsize=20, cmap='viridis_r', mincnt=1)
  bar = fig.colorbar(sdh)
  # bar.formatter.set_useOffset(True)
  # hist = ax.hist2d(maps_width, maps_height, bins=50, norm=colors.LogNorm())
  ax.set_xlim([0,20000])
  ax.set_ylim([0,20000])
  pyplot.show()

def parse_wads(wad_ids, dataset_path):
  maps_height = list()
  maps_width = list()
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
            sections, floors = label(levels[0]["maps"]["floormap"], connectivity=2, return_num=True)
            if floors <=2 and features["number_of_monsters"]>0 and features["number_of_weapons"]>0:
              # pyplot.imshow(map)
              # pyplot.show()
              # Adding extracted features with the map 1495 without floor check, 693 with floor = 1 , 949 with floor <=2
              # feature_map = levels[0]["maps"]
              # feature_map.update(levels[0]["features"])
              # feature_maps.append(feature_map)
              map = levels[0]["maps"]["roommap"]
              if map.max() > maps_meta['max_rooms']: maps_meta['max_rooms'] = map.max()
              feature_maps.append(levels[0]["maps"])
              valid_dataset.append({'id':id, 'name':file})
              print('added level', id, 'from', file)
              maps_height.append(features['x_max']-features['x_min'])
              maps_width.append(features['y_max']-features['y_min'])
              break
        except:
          print('failed to add level', id, 'from', file)
    # if len(valid_dataset) >= 10: break
  plot_dims(maps_width, maps_height)
  print(len(maps_height))
  return feature_maps, valid_dataset, maps_meta

def metadata_gen(wad_list,map_keys,cate_pref,save_path,meta,graphics_meta):

  map_meta = dict()
  map_meta["floormap"] = {"type": "uint8", "min": 0, "max": 255}
  map_meta["wallmap"] = {"type": "uint8", "min": 0, "max": 255}
  map_meta["heightmap"] = {"type": "uint8", "min": 0, "max": 255} 
  map_meta["triggermap"] = {"type": "uint8", "min": 0, "max": 255}
  map_meta["roommap"] = {"type": "uint8", "min": 0, "max": int(meta['max_rooms'])}
  map_meta["floortexturemap"] = {"type": "uint8", "min": 0, "max": graphics_meta["flats"]}
  map_meta["ceilingtexturemap"] =  {"type": "uint8", "min": 0, "max": graphics_meta["flats"]}
  map_meta["rightwalltexturemap"] = {"type": "uint8", "min": 0, "max": graphics_meta["textures"]}
  map_meta["leftwalltexturemap"] = {"type": "uint8", "min": 0, "max": graphics_meta["textures"]}
  map_meta["thingsmap"] = {"type": "uint8", "min": 0, "max": 123}

  obj = dict()
  cate = ThingTypes.get_all_categories()
  for cat in cate:
    obj[cat] = ThingTypes.get_index_by_category(cat)
  map_meta["start"] = {"type": "uint8", "min": obj["start"][0]-1, "max": obj["start"][-1]}
  map_meta["other"]=  {"type": "uint8", "min":obj["other"][0]-1, "max": obj["other"][-1]}
  map_meta["keys"] = {"type": "uint8", "min": obj["keys"][0]-1, "max": obj["keys"][-1]}
  map_meta["decorations"] = {"type": "uint8", "min": obj["decorations"][0]-1, "max": obj["decorations"][-1]}
  map_meta["obstacles"] = {"type": "uint8", "min": obj["obstacles"][0]-1, "max": obj["obstacles"][-1]}
  map_meta["monsters"] = {"type": "uint8", "min": obj["monsters"][0]-1, "max": obj["monsters"][-1]}
  map_meta["ammunitions"] = {"type": "uint8", "min": obj["ammunitions"][0]-1, "max": obj["ammunitions"][-1]}
  map_meta["weapons"] = {"type": "uint8", "min": obj["weapons"][0]-1, "max": obj["weapons"][-1]}
  map_meta["powerups"] = {"type": "uint8", "min": obj["powerups"][0]-1, "max": obj["powerups"][-1]}
  map_meta["artifacts"] = {"type": "uint8", "min": obj["artifacts"][0]-1, "max": obj["artifacts"][-1]}

  obj['essentials'] = ThingTypes.get_index_by_category('essentials')
  map_meta["essentials"] = {"type": "uint8", "min": 0, "max": obj['essentials'][-1]}


  map_dict = dict()
  map_dict['wads'] = {wad['id']: i for i,wad in enumerate(wad_list)}
  map_dict['count'] = len(wad_list)
  keys = map_keys + cate_pref
  key_meta = {key: map_meta[key] for key in keys}
  map_dict['maps_meta'] = key_meta

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
  if not os.path.exists(save_path):
    os.makedirs(save_path)
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
  # generate_tfrecord(feature_maps,keys,cate,save_path)
  # metadata_gen(wad_list,keys,cate,save_path,meta,graphics["meta"])


dataset_path = '../dataset/scraped/doom/'
save_path = '../dataset/parsed/doom/'
# All the generated maps are ['thingsmap', 'floormap', 'wallmap', 'heightmap', 'triggermap', 'roommap', 'floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap', 
# 'leftwalltexturemap'] with things map containing ['start','other', 'keys', 'decorations', 'obstacles',  'monsters', 'ammunitions', 'weapons', 'powerups', 'artifacts']
keys = ['floormap', 'wallmap', 'heightmap', 'triggermap', 'roommap', 'thingsmap', 'floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap','leftwalltexturemap']
things_cate = ['essentials','start', 'other', 'keys','obstacles',  'monsters', 'ammunitions', 'weapons', 'powerups', 'artifacts']
doom_parser(dataset_path,keys,things_cate,save_path)
