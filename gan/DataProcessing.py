import os, json, random
import tensorflow as tf
from matplotlib import pyplot as plt
from gan.NetworkArchitecture import topological_maps, object_maps
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def generate_images(model, seed, epoch, keys, is_cgan = False, is_mod=False, test_input = None, meta = None):
    prediction = model([test_input, seed], training=True) if is_cgan else model(seed, training=False)
    if is_cgan:
        scaled_pred = rescale_maps(prediction,meta,keys)
        for i in range(len(keys)):
            essentials = scaled_pred[0,:,:,i] if i == 0 else tf.maximum(essentials, scaled_pred[0,:,:,i])
    display_list = ([test_input[0,:,:,i] for i in range(len(topological_maps))] + [essentials] if is_cgan 
                    else [prediction[0,:,:,i] for i in range(len(keys))])

    plt.figure(figsize=(8, 8))
    for i in range(len(display_list)):
        plt.subplot(2, 2, i+1)
        if keys[i] == 'essentials':
            plt.imshow((display_list[i]*2)+(display_list[i]>0).astype(tf.float32)*155, cmap='gray') 
        else:
            plt.imshow(display_list[i], cmap='gray') 
        plt.axis('off')
    loc = ('artifacts/generated_maps/hybrid/mod_cgan/' if is_mod else 'artifacts/generated_maps/hybrid/trad_cgan/' if is_cgan 
           else 'artifacts/generated_maps/wgan/' if 'essentials' in keys else 'artifacts/generated_maps/hybrid/wgan/')
    if not os.path.exists(loc):
        os.makedirs(loc)
    plt.savefig(loc + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def view_maps(maps, keys, meta, split_objs=True, only_objs=True):
    n_maps = maps.shape[0]
    if split_objs:
        if not only_objs:
            for i in range(maps.shape[3]):
                thingsid = keys.index('essentials')
                if i!=thingsid:
                    for j in range(n_maps):
                        plt.subplot(2*n_maps, 4, i+j*8+1)
                        plt.imshow(maps[j,:,:,i], cmap='gray')
                        plt.axis('off')
                else:
                    for j,cate in enumerate(object_maps):
                        min = meta[cate]['min']
                        max = meta[cate]['max']
                        for k in range(n_maps):
                            cate_mask = tf.cast(tf.logical_and(maps[k,:,:,thingsid]>min,maps[k,:,:,thingsid]<=max),tf.uint8)
                            cate_objs = maps[k,:,:,thingsid]*cate_mask
                            plt.subplot(2*n_maps, 4, i+k*8+j+1)
                            plt.imshow((cate_objs*55/max)+cate_mask*200, cmap='gray')
                            plt.axis('off')
            plt.tight_layout(pad=0.2)
            plt.show()
        else:
            for i in range(n_maps):
                for j in range(maps.shape[3]):
                    floorid = keys.index('floormap')
                    thingsid = keys.index('essentials')
                    if j== floorid:
                            plt.subplot(2*n_maps, 3, i*6+1)
                            plt.imshow(maps[i,:,:,j]>0, cmap='gray')
                            plt.axis('off')
                    elif j== thingsid:
                        for k,cate in enumerate(object_maps):
                            min = meta[cate]['min']
                            max = meta[cate]['max']
                            cate_mask = tf.cast(tf.logical_and(maps[i,:,:,j]>min,maps[i,:,:,j]<=max),tf.uint8)
                            cate_objs = maps[i,:,:,j]*cate_mask
                            plt.subplot(2*n_maps, 3, i*6+k+2)
                            plt.imshow((cate_objs*55/max)+cate_mask*200, cmap='gray')
                            plt.axis('off')
            plt.tight_layout(pad=0.2)
            plt.show()
    else:
        items = meta['essentials']['max']
        for j in range(len(keys)):
            plt.subplot(1, 4, j+1)
            if keys[j] in ['essentials','thingsmap']:
                plt.imshow((maps[0,:,:,j]*55/items)+tf.cast(maps[0,:,:,j]>0,tf.float32)*200, cmap='gray')
            else:
                plt.imshow(maps[0,:,:,j], cmap='gray')
            plt.axis('off')
        plt.show()

def normalize_maps(x, map_meta, map_names, use_sigmoid=True):
    """
    Compute the scaling of eve ry map based on their .meta statistics (max and min)
     to bring all values inside (0,1) or (-1,1)
    :param x: the input vector. shape(batch, width, height, len(map_names))
    :param map_names: the name of the feature maps for .meta file lookup
    :param use_sigmoid: if True data will be in range 0,1, if False it will be in -1;1
    :return: a normalized x vector
    """
    fx = tf.cast(x,tf.float32)
    a = 0 if use_sigmoid else -1
    b = 1
    max = tf.constant([map_meta[m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    min = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    min_mat = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.cast(x>0,tf.float32)
    return a + ((fx-min_mat)*(b-a))/(max-min)


def rescale_maps(x, map_meta, map_names, use_sigmoid=True):
    a = 0 if use_sigmoid else -1
    b = 1
    min_mat = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.cast(x>0,tf.float32)
    min = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    max = tf.constant([map_meta[m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    return tf.math.round(((max-min) * (x - a)/(b-a)) + min_mat)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value is tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Created as it is unable to do it inline 
def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def generate_sample(imgs ,keys, save_path = 'dataset/generated/doom/', file_name = 'sample.tfrecords'):
    file_path = save_path + file_name
    with tf.io.TFRecordWriter(file_path) as writer:
        gen_maps = dict()
        for i in range(len(keys)):
            gen_maps[keys[i]] = imgs[0, :, :, i]
        serialized_array = {key: serialize_array(gen_maps[key]) for key in keys}
        feature = {key: _bytes_feature(serialized_array[key]) for key in keys}
        example_message = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_message.SerializeToString())
    file = save_path + 'meta.json'
    with open(file, 'w') as jsonfile:
        json.dump({'keys':keys}, jsonfile)