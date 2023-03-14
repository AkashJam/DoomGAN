import tensorflow as tf
import os, sys
from matplotlib import pyplot as plt
from pix2pix import Generator as pix2pixGen
from wgan import Generator as wganGen
import numpy as np
from ganmeta import generate_sample, read_json


def unscale_maps(x,map_meta,map_names):
    min = tf.constant([map_meta['maps_meta'][m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    max = tf.constant([map_meta['maps_meta'][m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    return tf.math.round(((max-min) * x) + min)

def generate_maps(seed):
    generator = wganGen()
    checkpoint_dir = './training_checkpoints/wgan'
    checkpoint = tf.train.Checkpoint(generator=generator)
    if os.path.exists(checkpoint_dir):                            
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    prediction = generator(seed, training=False)
    return prediction

def generate_images(test_input,noise_img):
    generator = pix2pixGen()
    checkpoint_dir = './training_checkpoints/pix2pix'
    checkpoint = tf.train.Checkpoint(generator=generator)
    if os.path.exists(checkpoint_dir):                            
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    prediction = generator([test_input,noise_img], training=False)
    samples = tf.concat([test_input,prediction],axis=-1)
    meta = read_json()
    keys = ['floormap', 'wallmap', 'heightmap','essentials']
    gen_maps = unscale_maps(samples,meta,keys)
    actual_maps = tf.cast(gen_maps,tf.uint8)

    monsters = tf.reduce_sum(tf.logical_and(actual_maps[0,:,:,3]<18, actual_maps[0,:,:,3]>0).astype(tf.uint8))
    ammunition = tf.reduce_sum(tf.logical_and(actual_maps[0,:,:,3]<26, actual_maps[0,:,:,3]>17).astype(tf.uint8))
    weapons = tf.reduce_sum(tf.logical_and(actual_maps[0,:,:,3]<34, actual_maps[0,:,:,3]>25).astype(tf.uint8))
    print(monsters,ammunition,weapons)

    plt.figure(figsize=(8, 8))

    display_list = [actual_maps[0,:,:,0], actual_maps[0,:,:,1], actual_maps[0,:,:,2], actual_maps[0,:,:,3]]
    title = ['Floor Map', 'Wall Map', 'Height Map', 'Things Map']

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()
    generate_sample(actual_maps,keys,test=True)
    print('created sample level record')

seed = tf.random.normal([1, 100])
test_sample = generate_maps(seed)
noisy_img = tf.random.normal([1, 256, 256, 1])
generate_images(test_sample,noisy_img)