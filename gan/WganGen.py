import tensorflow as tf
from matplotlib import pyplot as plt
from wgan import Generator as wganGen
from GanMeta import generate_sample, read_json, rescale_maps
from NetworkArchitecture import topological_maps


def wgan_fmaps(model, seed):
    maps = model(seed, training=False)
    meta = read_json()
    n_items = meta['maps_meta']['essentials']['max']
    keys = topological_maps + ['essentials']
    scaled_maps = rescale_maps(maps,meta['maps_meta'],keys)
    gen_maps = tf.cast(scaled_maps,tf.uint8)
    return gen_maps, keys, n_items

    
if __name__ == "__main__":
    z_dim = 100
    batch_size = 1
    z = tf.random.normal([batch_size, z_dim])
    map_keys = topological_maps + ['essentials']
    n_maps = len(map_keys)
    
    generator = wganGen(n_maps, z_dim)
    checkpoint_dir = './training_checkpoints/wgan'
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    feature_maps,feature_keys, max_items = wgan_fmaps(generator, z)

    plt.figure(figsize=(8, 4))
    for j in range(len(feature_keys)):
        plt.subplot(1, 4, j+1)
        # plt.title(feature_keys[j] if feature_keys[j] != 'essentials' else 'thingsmap')
        if feature_keys[j] in ['essentials','thingsmap']:
            plt.imshow((feature_maps[0,:,:,j]*55/max_items)+tf.cast(feature_maps[0,:,:,j]>0,tf.float32)*200, cmap='gray')
        else:
            plt.imshow(feature_maps[0,:,:,j], cmap='gray')
        plt.axis('off')
    plt.show()

    generate_sample(feature_maps,feature_keys,save_path='../dataset/generated/doom/wgan/', file_name = 'test.tfrecords')
    print('created sample level record')