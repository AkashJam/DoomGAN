from WADParser.WADEditor import WADReader
import tensorflow as tf
import os, json, sys
import WADParser.Dictionaries.ThingTypes as ThingTypes
from skimage.morphology import label
from matplotlib import pyplot as plt
from WADParser.TextureDictGen import TextureDict


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))): # if value ist tensor
      value = value.numpy() # get value of tensor
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def plot_dims(widths, heights):
  fig, ax = plt.subplots(tight_layout=True)
  plt.xlabel('Map Width in DOOM Units')
  plt.ylabel('Map Length in DOOM Units')
  sdh = ax.hexbin(widths, heights, gridsize=23, cmap='viridis_r', mincnt=1)
  bar = fig.colorbar(sdh)
  ax.set_xlim([0,30000])
  ax.set_ylim([0,30000])
  plt.savefig('artifacts/eval_metrics/area_distribution')
  plt.close()


class Parser:
  def __init__(self):
    super().__init__()
    self.dataset_path = 'dataset/scraped/doom/'
    self.save_path = 'dataset/parsed/doom/'
    self.feature_keys = ['floormap', 'wallmap', 'heightmap', 'triggermap', 'roommap', 'thingsmap', 'texturemap']
    self.categories = ['essentials','start', 'other', 'keys','obstacles',  'monsters', 'ammunitions', 'weapons', 'powerups', 'artifacts']
    self.textures = ['floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap','leftwalltexturemap']

  def read_json(self):
    json_path = self.dataset_path + 'doom.json'
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


  def extract_features(self, wad_ids):
    feature_maps = list()
    reader = WADReader()
    valid_dataset = list()
    maps_meta = dict()
    maps_meta['lengths'] = list()
    maps_meta['widths'] = list()
    maps_meta['max_rooms'] = 0
    for id in wad_ids:
      wad_path = self.dataset_path + id + "/"
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

  def metadata_gen(self, wad_list, meta, graphics_meta):
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
    keys = self.feature_keys + self.categories
    key_meta = {key: map_meta[key] for key in keys}
    map_dict['maps_meta'] = key_meta

    file = self.save_path + 'metadata.json'
    with open(file, 'w') as jsonfile:
      json.dump(map_dict, jsonfile)


  def organize_maps(self, mat):
    organized_map = dict()
    for key in self.feature_keys:
      if key == 'thingsmap':
        organized_map[key] = mat[key][key]
        for cat in self.categories:
          organized_map[cat] = mat[key][cat]
      elif key == 'texturemap':
        for cat in self.textures:
          organized_map[cat] = mat[key][cat]
      else:
        organized_map[key] = mat[key]
    return organized_map


  # Write TFrecord file
  def generate_tfrecord(self, maps):
    file_name = 'data.tfrecords'
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    file_path = self.save_path + file_name
    keys = self.feature_keys + self.categories + self.textures
    with tf.io.TFRecordWriter(file_path) as writer:
      for map in maps:
        org_maps = self.organize_maps(map)
        serialized_array = {key: tf.io.serialize_tensor(org_maps[key]) for key in keys}
        feature = {key: _bytes_feature(serialized_array[key]) for key in keys}
        example_message = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_message.SerializeToString())
    print("tf record written successfully")


  def parse_wads(self):
    texture_dict = TextureDict()
    texture_dict.texturedict_gen()
    file_path = self.save_path + 'graphics.json'
    if os.path.isfile(file_path):
      with open(file_path, 'r') as jsonfile:
          graphics = json.load(jsonfile)
    else:
      print('generate graphics json first')
      sys.exit()
    wad_ids = self.read_json()
    feature_maps, wad_list, meta = self.extract_features(wad_ids)
    plot_dims(meta['widths'], meta['lengths'])
    print(len(meta['lengths']))
    self.generate_tfrecord(feature_maps)
    self.metadata_gen(wad_list, meta, graphics["meta"])