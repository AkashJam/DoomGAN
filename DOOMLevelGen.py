import tensorflow as tf
import os, sys
from gan.NetworkArchitecture import topological_maps, object_maps
from gan.cgan import cGAN
from gan.wgan import WGAN_GP
from gan.DataProcessing import generate_sample, view_maps, rescale_maps, normalize_maps
from gan.NNMeta import read_json, read_record, parse_tfrecord
from WADParser.WADEditor import WADWriter
import numpy as np
from skimage import morphology
from gan.ModelEvaluation import plot_RipleyK, plot_train_metrics, calc_proportions, calc_stats
from WADScraper.scraper import Scraper
from WADParser.parser import Parser


def train_model(params):
    wgan = WGAN_GP(params['use_hybrid'])
    wgan.train(params['epochs'])
    if params['use_hybrid']:
        cgan = cGAN(params['mod_cgan'])
        cgan.train(params['epochs'])


def wgan_fmaps(model, seed, meta):
    maps = model(seed, training=False)
    keys = topological_maps + ['essentials']
    # batch_size = seed.shape[0]
    # floor_id = topological_maps.index('floormap')
    # process_maps = maps.numpy()
    # for i in range(batch_size):
    #         map_size = np.sum((process_maps[i,:,:,floor_id]>0.1).astype(int))
    #         sm_hole_size = int(process_maps.shape[1]/24)
    #         map = morphology.remove_small_holes(process_maps[i,:,:,floor_id]>0.1,sm_hole_size)
    #         map = morphology.remove_small_objects(map,map_size/3).astype(int)
    #         for j in range(len(topological_maps)):
    #             process_maps[i,:,:,j] = process_maps[i,:,:,j]*map
    # maps = tf.convert_to_tensor(process_maps, dtype=tf.float32)
    scaled_maps = rescale_maps(maps,meta,keys)
    gen_maps = tf.cast(scaled_maps,tf.uint8)
    return gen_maps, keys


def hybrid_fmaps(wgan_model, can_model, seed, noisy_img, meta, use_sample=False, level_layout=None):
    if not use_sample:
        batch_size = seed.shape[0]
        level_layout = wgan_model(seed, training=False)
        floor_id = topological_maps.index('floormap')
        gen_layout = level_layout.numpy()
        for i in range(batch_size):
            map_size = np.sum((gen_layout[i,:,:,floor_id]>0.1).astype(int))
            sm_hole_size = int(gen_layout.shape[1]/24)
            map = morphology.remove_small_holes(gen_layout[i,:,:,floor_id]>0.1,sm_hole_size)
            map = morphology.remove_small_objects(map,map_size/3).astype(int)
            for j in range(len(topological_maps)):
                gen_layout[i,:,:,j] = gen_layout[i,:,:,j]*map
        level_layout = tf.convert_to_tensor(gen_layout, dtype=tf.float32)
        if tf.reduce_sum(level_layout[0,:,:,floor_id])==0:
            print('use another seed')
            sys.exit()
    level_objs = can_model([level_layout,noisy_img], training=True)
    samples = tf.concat([level_layout,level_objs],axis=-1)
    keys = topological_maps + object_maps
    scaled_maps = rescale_maps(samples,meta,keys)
    actual_maps = tf.cast(scaled_maps,tf.uint8)
    for i,key in enumerate(keys):
        if key in object_maps:
            id = keys.index(key)
            min = meta[key]['min']
            max = meta[key]['max']
            corrected_mask = tf.cast(tf.logical_and(tf.logical_and(actual_maps[:,:,:,id]>min, actual_maps[:,:,:,id]<=max),
                                                    actual_maps[:,:,:,keys.index('floormap')]>0),tf.uint8)
            corrected_map = actual_maps[:,:,:,id] * corrected_mask
            essentials = corrected_map if id==len(topological_maps) else tf.maximum(essentials,corrected_map)
    gen_maps = tf.stack([actual_maps[:,:,:,0], actual_maps[:,:,:,1], actual_maps[:,:,:,2], essentials],axis=-1)
    title = topological_maps + ['essentials']
    return gen_maps, title


def sample_generation(params):
    z = tf.random.normal([params['batch_size'], params['z_dim']],seed=params['seed'])
    noisy_img = tf.random.normal([params['batch_size'], 256, 256, 1],seed=params['seed'])
    meta = read_json()
    Tgen = WGAN_GP(params['use_hybrid'])
    Tgen.load_checkpoint()
    if params['use_hybrid']:
        Fgen = cGAN(params['mod_cgan'])
        Fgen.load_checkpoint()
        if params['use_sample']:
            loc = 'dataset/generated/doom/wgan/test.tfrecords'
            if not os.path.isfile(loc):
                print('Generate and save trad wgan sample')
                sys.exit()
            tfr_dataset = tf.data.TFRecordDataset(loc)
            sample_input = parse_tfrecord(tfr_dataset,topological_maps+['essentials'],sample_wgan=True,file_path='dataset/generated/doom/wgan/test.tfrecords')
            level_maps = tf.stack([sample_input[m] for m in topological_maps], axis=-1).reshape(1, 256, 256, len(topological_maps))
            level_maps = tf.cast(level_maps,tf.float32)
            level_maps = normalize_maps(level_maps,meta['maps_meta'],topological_maps)
            feature_maps, feature_keys = hybrid_fmaps(Tgen.generator, Fgen.generator, z, noisy_img, meta['maps_meta'], use_sample=True, level_layout=level_maps) 
        else: 
            feature_maps, feature_keys = hybrid_fmaps(Tgen.generator, Fgen.generator, z, noisy_img, meta['maps_meta'])
        loc = 'dataset/generated/doom/hybrid/'
        path = loc + 'mod_cgan/' if params['mod_cgan'] else loc + 'trad_cgan/'
    else:
        feature_maps,feature_keys = wgan_fmaps(Tgen.generator, z, meta['maps_meta'])
        path='dataset/generated/doom/wgan/'
    view_maps(feature_maps, feature_keys, meta['maps_meta'], split_objs=True, only_objs=True)
    if not os.path.exists(path):
        os.makedirs(path)
    if params['save_sample']: 
        generate_sample(feature_maps, feature_keys, save_path=path, file_name = 'test.tfrecords')
    if params['gen_level']:
        maps = dict()
        for i in range(len(feature_keys)):
            maps[feature_keys[i]] = feature_maps[0, :, :, i]
        writer = WADWriter()
        writer.add_level('MAP01')
        writer.from_images(maps['floormap'],maps['heightmap'],maps['wallmap'],maps['essentials'])
        writer.save(path+'test.wad')
        print('created sample level record')


def model_eval(params):
    model = WGAN_GP(use_hybrid=False)
    model.load_checkpoint()
    TWgen = model.generator
    model = WGAN_GP(use_hybrid=True)
    model.load_checkpoint()
    HWgen = model.generator
    model = cGAN(use_mod=False)
    model.load_checkpoint()
    TFgen = model.generator
    model = cGAN(use_mod=True)
    model.load_checkpoint()
    MFgen = model.generator
    del model
    training_set, validation_set, map_meta, sample= read_record(batch_size=params['test_size'])
    for i, data in training_set.enumerate().as_numpy_iterator():
        z = tf.random.normal([params['test_size'], params['z_dim']])
        noise = tf.random.normal([params['test_size'], 256, 256, 1])
        wgan_maps, keys = wgan_fmaps(TWgen, z, map_meta)
        hybrid_trad_maps, keys = hybrid_fmaps(HWgen, TFgen, z, noise, map_meta)
        hybrid_mod_maps, keys = hybrid_fmaps(HWgen, MFgen, z, noise, map_meta)
        real_maps = np.stack([data[m] for m in keys], axis=-1)
        r_maps = np.concatenate((r_maps,real_maps), axis=0) if i != 0 else real_maps
        w_maps = np.concatenate((w_maps,wgan_maps.numpy()), axis=0) if i != 0 else wgan_maps.numpy()
        ht_maps = np.concatenate((ht_maps,hybrid_trad_maps.numpy()), axis=0) if i != 0 else hybrid_trad_maps.numpy()
        hm_maps = np.concatenate((hm_maps,hybrid_mod_maps.numpy()), axis=0) if i != 0 else hybrid_mod_maps.numpy()
        if i==1: break
    calc_proportions(r_maps,w_maps,ht_maps,hm_maps,keys,map_meta,params['test_size'])
    calc_stats(r_maps,w_maps,ht_maps,hm_maps,keys,map_meta)
    plot_RipleyK(r_maps,w_maps,ht_maps,hm_maps,keys,params['test_size'])
    plot_train_metrics()


def define_flags():
    flags = dict()
    flags['batch_size'] = 1
    flags['seed'] = 19
    flags['z_dim'] = 100
    flags['n_tmaps'] = len(topological_maps)
    flags['n_omaps'] = len(object_maps)
    flags['use_hybrid'] = True
    flags['mod_cgan'] = True
    flags['epochs'] = 100
    flags['use_sample'] = False
    flags['gen_level'] = True
    flags['save_sample'] = False  
    flags['test_size'] = 100
    return flags


if __name__ == "__main__":
    params = define_flags()
    if len(sys.argv)==2:
        if sys.argv[1] =='-scrap':
            wad_scraper = Scraper()
            wad_scraper.scrap_levels()
        elif sys.argv[1]=='-parse':
            wad_parser = Parser()
            wad_parser.parse_wads()
        elif sys.argv[1] =='-train':
            train_model(params)
        elif sys.argv[1]=='-evaluate':
            model_eval(params)
        elif sys.argv[1]=='-generate':
            sample_generation(params)
        else:
            print('Please provide one of the these flags -scrap -parse, -train, -evaluate, -generate')
    else:
        print('Please provide one of the these flags -scrap -parse, -train, -evaluate, -generate')
    