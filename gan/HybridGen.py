import tensorflow as tf
from matplotlib import pyplot as plt
import os
from can import Generator as canGen
from wgan import Generator as wganGen
from NetworkArchitecture import topological_maps, object_maps
from DataProcessing import generate_sample, read_json, rescale_maps, parse_tfrecord, normalize_maps
from NNMeta import view_maps
import numpy as np
from skimage import morphology
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def hybrid_fmaps(wgan_model, can_model, seed, noisy_img, meta, use_sample=False, level_layout=None):
    if not use_sample:
        batch_size = seed.shape[0]
        level_layout = wgan_model(seed, training=False)
        floor_id = topological_maps.index('floormap')
        gen_layout = level_layout.numpy()

        for i in range(batch_size):
            map_size = np.sum((gen_layout[i,:,:,floor_id]>0.1).astype(int))
            sm_hole_size = int(gen_layout.shape[1]/10)
            map = morphology.remove_small_holes(gen_layout[i,:,:,floor_id]>0.1,sm_hole_size)
            map = morphology.remove_small_objects(map,map_size/3).astype(int)

            for j in range(len(topological_maps)):
                gen_layout[i,:,:,j] = gen_layout[i,:,:,j]*map

        level_layout = tf.convert_to_tensor(gen_layout, dtype=tf.float32)

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
            # print(key,tf.reduce_sum((actual_maps[:,:,:,id]>0).astype(tf.int32)))
            corrected_map = actual_maps[:,:,:,id] * tf.cast(tf.logical_and(actual_maps[:,:,:,id]>min,actual_maps[:,:,:,id]<=max),tf.uint8)
            essentials = corrected_map if id==len(topological_maps) else tf.maximum(essentials,corrected_map)

    gen_maps = tf.stack([actual_maps[:,:,:,0], actual_maps[:,:,:,1], actual_maps[:,:,:,2], essentials],axis=-1)
    title = topological_maps + ['essentials']
    return gen_maps, title



if __name__ == "__main__":
    batch_size = 1
    z_dim = 100
    trad = False
    use_sample = False
    z = tf.random.normal([batch_size, z_dim])
    noisy_img = tf.random.normal([batch_size, 256, 256, 1])
    n_tmaps = len(topological_maps)
    n_omaps = len(object_maps)
    meta = read_json()


    wgen = wganGen(n_tmaps, z_dim)
    checkpoint_dir = './training_checkpoints/hybrid/wgan'
    checkpoint = tf.train.Checkpoint(generator=wgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    

    if trad:
        cgen = canGen(n_tmaps,n_omaps)
        checkpoint_dir = './training_checkpoints/hybrid/trad_can'
        checkpoint = tf.train.Checkpoint(generator=cgen)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    else:
        cgen = canGen(n_tmaps,n_omaps)
        checkpoint_dir = './training_checkpoints/hybrid/mod_can'
        checkpoint = tf.train.Checkpoint(generator=cgen)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    if use_sample:
        tfr_dataset = tf.data.TFRecordDataset('../dataset/generated/doom/wgan/test.tfrecords')
        sample_input = parse_tfrecord(tfr_dataset,topological_maps+['essentials'],sample_wgan=True,file_path='../dataset/generated/doom/wgan/test.tfrecords')
        level_maps = tf.stack([sample_input[m] for m in topological_maps], axis=-1).reshape(1, 256, 256, len(topological_maps))
        level_maps = tf.cast(level_maps,tf.float32)
        level_maps = normalize_maps(level_maps,meta['maps_meta'],topological_maps)
        feature_maps, feature_keys = hybrid_fmaps(wgen, cgen, z, noisy_img, meta['maps_meta'], use_sample=True, level_layout=level_maps) 
    else: 
        feature_maps, feature_keys = hybrid_fmaps(wgen, cgen, z, noisy_img, meta['maps_meta'])

    

    view_maps(feature_maps, feature_keys, meta['maps_meta'], split_objs=False)

    loc = '../dataset/generated/doom/hybrid/'
    path = loc + 'trad_can/' if trad else loc + 'mod_can/'
    if not os.path.exists(path):
        os.makedirs(path)
    # generate_sample(feature_maps, feature_keys, save_path=loc, file_name = 'test.tfrecords')
    print('created sample level record')