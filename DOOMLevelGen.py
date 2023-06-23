import tensorflow as tf
import os, sys
from gan.NetworkArchitecture import topological_maps, object_maps
from gan.cgan import Generator as cganGen
from gan.wgan import Generator as wganGen
from gan.DataProcessing import generate_sample, read_json, rescale_maps, parse_tfrecord, normalize_maps
from gan.NNMeta import view_maps
from WADParser.WADEditor import WADWriter
import numpy as np
from skimage import morphology


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
    meta = read_json('dataset/parsed/doom/')
    wgen = wganGen(params['n_tmaps'], params['z_dim']) if params['use_hybrid'] else wganGen(params['n_tmaps']+1, params['z_dim'])
    checkpoint_dir = 'gan/training_checkpoints/hybrid/wgan' if params['use_hybrid'] else 'gan/training_checkpoints/wgan'
    checkpoint = tf.train.Checkpoint(generator=wgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    if params['use_hybrid']:
        cgen = cganGen(params['n_tmaps'],params['n_omaps'])
        checkpoint_dir = 'gan/training_checkpoints/hybrid/mod_cgan' if params['mod_cgan'] else 'gan/training_checkpoints/hybrid/trad_cgan'
        checkpoint = tf.train.Checkpoint(generator=cgen)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
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
            feature_maps, feature_keys = hybrid_fmaps(wgen, cgen, z, noisy_img, meta['maps_meta'], use_sample=True, level_layout=level_maps) 
        else: 
            feature_maps, feature_keys = hybrid_fmaps(wgen, cgen, z, noisy_img, meta['maps_meta'])
        loc = 'dataset/generated/doom/hybrid/'
        path = loc + 'mod_cgan/' if params['mod_cgan'] else loc + 'trad_cgan/'
    else:
        feature_maps,feature_keys = wgan_fmaps(wgen, z, meta['maps_meta'])
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


def define_flags():
    flags = dict()
    flags['batch_size'] = 1
    flags['seed'] = 7
    flags['z_dim'] = 100
    flags['n_tmaps'] = len(topological_maps)
    flags['n_omaps'] = len(object_maps)
    flags['use_hybrid'] = True
    flags['mod_cgan'] = True
    flags['use_sample'] = False
    flags['gen_level'] = False
    flags['save_sample'] = False
    return flags


if __name__ == "__main__":
    params = define_flags()
    sample_generation(params)
    