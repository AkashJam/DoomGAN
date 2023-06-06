from WADEditor import WADReader
import tensorflow as tf
import os, json, sys
from matplotlib import pyplot
import Dictionaries.ThingTypes as ThingTypes
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
        sys.exit()
      else:
        scraped_ids = [info['id'] for info in scraped_info if 'id' in info]
        return scraped_ids
  else:
    print('JSON not found')
    sys.exit()
  
  
def plot_dims(widths, heights):
  fig, ax = pyplot.subplots(tight_layout=True)
  pyplot.xlabel('Map Width in DOOM Units')
  pyplot.ylabel('Map Length in DOOM Units')
  sdh = ax.hexbin(widths, heights, gridsize=23, cmap='viridis_r', mincnt=1)
  bar = fig.colorbar(sdh)
  ax.set_xlim([0,30000])
  ax.set_ylim([0,30000])
  pyplot.show()


def parse_wads(wad_ids, dataset_path):
  feature_maps = list()
  reader = WADReader()
  valid_dataset = list()
  maps_meta = dict()
  maps_meta['lengths'] = list()
  maps_meta['widths'] = list()
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
            if floors > 1:
              areas = list()
              for i in range(floors):
                if i != 0:
                  area = int(tf.reduce_sum(tf.cast(sections==i,tf.int32)))
                  areas.append(area)
            if (floors ==1 or max(areas)/8>(sum(areas)-max(areas))) and features["number_of_monsters"]>0 and features["number_of_weapons"]>0:
              map = levels[0]["maps"]["roommap"]
              if map.max() > maps_meta['max_rooms']: maps_meta['max_rooms'] = map.max()
              feature_maps.append(levels[0]["maps"])
              valid_dataset.append({'id':id, 'name':file})
              print('added level', id, 'from', file)
              maps_meta['widths'].append(features['width'])
              maps_meta['lengths'].append(features['height'])
              break
        except:
          print('failed to add level', id, 'from', file)
    # if len(valid_dataset) >= 10: break
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
  obj = ThingTypes.get_index_by_category('essentials')
  map_meta["essentials"] = {"type": "uint8", "min": 0, "max": obj[-1]}

  cate = ThingTypes.get_all_categories()
  for cat in cate:
    obj = ThingTypes.get_index_by_category(cat)
    map_meta[cat] = {"type": "uint8", "min": obj[0]-1, "max": obj[-1]}

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


def organize_maps(mat, keys, cate):
  organized_map = dict()
  for key in keys:
    if key == 'thingsmap':
      organized_map[key] = mat[key][key]
      for cat in cate:
        organized_map[cat] = mat[key][cat]
    else:
      organized_map[key] = mat[key]
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
      org_maps = organize_maps(map,keys_pref,cate_pref)
      serialized_array = {key: tf.io.serialize_tensor(org_maps[key]) for key in keys}
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
  plot_dims(meta['widths'], meta['lengths'])
  print(len(meta['lengths']))
  generate_tfrecord(feature_maps,keys,cate,save_path)
  metadata_gen(wad_list,keys,cate,save_path,meta,graphics["meta"])


dataset_path = '../dataset/scraped/doom/'
save_path = '../dataset/parsed/doom/'
keys = ['floormap', 'wallmap', 'heightmap', 'triggermap', 'roommap', 'thingsmap', 'floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap','leftwalltexturemap']
things_cate = ['essentials','start', 'other', 'keys','obstacles',  'monsters', 'ammunitions', 'weapons', 'powerups', 'artifacts']
doom_parser(dataset_path,keys,things_cate,save_path)
