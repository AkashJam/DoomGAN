import tensorflow as tf
from matplotlib import pyplot as plt
from wgan import Generator as wganGen
from DataProcessing import read_json, rescale_maps, generate_sample
from NNMeta import view_maps
from NetworkArchitecture import topological_maps


def wgan_fmaps(model, seed, meta):
    maps = model(seed, training=False)
    keys = topological_maps + ['essentials']
    scaled_maps = rescale_maps(maps,meta,keys)
    gen_maps = tf.cast(scaled_maps,tf.uint8)
    return gen_maps, keys

    
if __name__ == "__main__":
    z_dim = 100
    batch_size = 1
    z = tf.random.normal([batch_size, z_dim])
    map_keys = topological_maps + ['essentials']
    n_maps = len(map_keys)
    meta = read_json()
    
    generator = wganGen(n_maps, z_dim)
    checkpoint_dir = './training_checkpoints/wgan'
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    feature_maps,feature_keys = wgan_fmaps(generator, z, meta['maps_meta'])
    view_maps(feature_maps, feature_keys, meta['maps_meta'], split_objs=False)

    generate_sample(feature_maps,feature_keys,save_path='../dataset/generated/doom/wgan/', file_name = 'test.tfrecords')
    print('created sample level record')