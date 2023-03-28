import tensorflow as tf
from matplotlib import pyplot as plt
from pix2pix import Generator as pix2pixGen
from wgan import Generator as wganGen
from GanMeta import generate_sample, read_json, rescale_maps


def generate_wgan_maps(seed):
    latent_dim = seed.shape[1]
    generator = wganGen(3, latent_dim)
    checkpoint_dir = './training_checkpoints/hybrid/wgan'
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    prediction = generator(seed, training=False)
    return prediction

def generate_p2p_maps(input_maps,noise):
    generator = pix2pixGen()
    checkpoint_dir = './training_checkpoints/hybrid/pix2pix'
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    prediction = generator([input_maps,noise], training=True)
    return prediction

def hybrid_fmaps(seed,noisy_img):
    topological_maps = generate_wgan_maps(seed)
    functional_maps = generate_p2p_maps(topological_maps,noisy_img)
    samples = tf.concat([topological_maps,functional_maps],axis=-1)
    meta = read_json()
    n_items = meta['maps_meta']['essentials']['max']
    topological_keys = ['floormap', 'wallmap', 'heightmap']
    functional_keys = ['monsters','ammunitions','powerups','artifacts','weapons']
    keys = topological_keys+functional_keys
    # obj_count = dict()
    # for key in functional_keys:
    #     id = keys.index(key)
    #     obj_count[key] = tf.reduce_sum(tf.cast(samples[0,:,:,id].astype(tf.bool),tf.uint8))
    # print({key: obj_count[key] for key in functional_keys})
    scaled_maps = rescale_maps(samples,meta['maps_meta'],keys)
    actual_maps = tf.cast(scaled_maps,tf.uint8)
    for i,key in enumerate(functional_keys):
        id = keys.index(key)
        essentials = actual_maps[:,:,:,id] if i == 0 else tf.maximum(essentials,actual_maps[:,:,:,id])

    gen_maps = tf.stack([actual_maps[:,:,:,0], actual_maps[:,:,:,1], actual_maps[:,:,:,2],essentials],axis=-1)
    title = ['floormap', 'wallmap', 'heightmap', 'essentials']
    return gen_maps, title, n_items



if __name__ == "__main__":
    batch_size = 1
    z_dim = 100
    z = tf.random.normal([batch_size, z_dim])
    noisy_img = tf.random.normal([batch_size, 256, 256, 1])
    feature_maps, feature_keys, max_items = hybrid_fmaps(z, noisy_img)
    
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

    generate_sample(feature_maps, feature_keys, save_path='../dataset/generated/doom/hybrid/', file_name = 'test.tfrecords')
    print('created sample level record')