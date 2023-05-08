import tensorflow as tf
from matplotlib import pyplot as plt
from pix2pix import Generator as pix2pixGen
from wgan import Generator as wganGen
from NetworkArchitecture import topological_maps, object_maps
from GanMeta import generate_sample, read_json, rescale_maps
import numpy as np
from skimage import morphology

def hybrid_fmaps(wgan_model, p2p_model, seed, noisy_img):
    level_layout = wgan_model(seed, training=False)
    floor_id = topological_maps.index('floormap')
    gen_layout = level_layout

    for i in range(batch_size):
        map_size = np.sum((topological_maps[i,:,:,floor_id]>0).astype(int))
        sm_hole_size = int(level_layout.shape[1]/10)
        map = morphology.remove_small_holes(level_layout[i,:,:,floor_id]>0,sm_hole_size)
        map = morphology.remove_small_objects(map,map_size/4).astype(int)

        for j in range(len(topological_maps)):
            level_layout[i,:,:,j] = level_layout[:,:,:,j]*map
    
    plt.subplot(2,3,1)
    plt.imshow(level_layout[0,:,:,0])
    plt.subplot(2,3,2)
    plt.imshow(level_layout[0,:,:,1])
    plt.subplot(2,3,3)
    plt.imshow(level_layout[0,:,:,2])
    plt.subplot(2,3,4)
    plt.imshow(gen_layout[0,:,:,0])
    plt.subplot(2,3,5)
    plt.imshow(gen_layout[0,:,:,1])
    plt.subplot(2,3,6)
    plt.imshow(gen_layout[0,:,:,2])
    plt.show()

    # level_objs = p2p_model([level_layout,noisy_img], training=True)
    # samples = tf.concat([topological_maps,level_objs],axis=-1)
    # meta = read_json()
    # n_items = meta['maps_meta']['essentials']['max']
    # keys = topological_maps + object_maps
    # scaled_maps = rescale_maps(samples,meta['maps_meta'],keys)
    # actual_maps = tf.cast(scaled_maps,tf.uint8)
    # for i,key in enumerate(object_maps):
    #     id = keys.index(key)
    #     essentials = actual_maps[:,:,:,id] if i == 0 else tf.maximum(essentials,actual_maps[:,:,:,id])

    # gen_maps = tf.stack([actual_maps[:,:,:,0], actual_maps[:,:,:,1], actual_maps[:,:,:,2],essentials],axis=-1)
    # title = topological_maps + ['essentials']
    # return gen_maps, title, n_items



if __name__ == "__main__":
    batch_size = 1
    z_dim = 100
    trad = False
    z = tf.random.normal([batch_size, z_dim])
    noisy_img = tf.random.normal([batch_size, 256, 256, 1])
    n_tmaps = len(topological_maps)
    n_omaps = len(object_maps)

    wgen = wganGen(n_tmaps, z_dim)
    checkpoint_dir = './training_checkpoints/hybrid/wgan'
    checkpoint = tf.train.Checkpoint(generator=wgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    if trad:
        tpgen = pix2pixGen(n_tmaps,n_omaps)
        checkpoint_dir = './training_checkpoints/hybrid/trad_pix2pix'
        checkpoint = tf.train.Checkpoint(generator=tpgen)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    else:
        mpgen = pix2pixGen(n_tmaps,n_omaps)
        checkpoint_dir = './training_checkpoints/hybrid/pix2pix'
        checkpoint = tf.train.Checkpoint(generator=mpgen)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    feature_maps, feature_keys, max_items = hybrid_fmaps(wgen, tpgen, z, noisy_img) if trad else hybrid_fmaps(wgen, mpgen, z, noisy_img)
    
    for i in range(10):
        plt.figure(figsize=(8, 4))
        for j in range(len(feature_keys)):
            plt.subplot(1, 4, j+1)
            # plt.title(feature_keys[j] if feature_keys[j] != 'essentials' else 'thingsmap')
            if feature_keys[j] in ['essentials','thingsmap']:
                plt.imshow((feature_maps[i,:,:,j]*55/max_items)+tf.cast(feature_maps[i,:,:,j]>0,tf.float32)*200, cmap='gray')
            else:
                plt.imshow(feature_maps[i,:,:,j], cmap='gray')
            plt.axis('off')
        plt.show()

    # generate_sample(feature_maps, feature_keys, save_path='../dataset/generated/doom/hybrid/', file_name = 'test.tfrecords')
    print('created sample level record')