import tensorflow as tf
import sys, os, json, random, math
from matplotlib import pyplot as plt
# Used for image operations on tensor
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def read_json(save_path = '../dataset/parsed/doom/'):
    file_path = save_path + 'metadata.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as jsonfile:
            map_meta = json.load(jsonfile)
        return map_meta
    else:
        print('No metadata found')
        sys.exit()


def read_record(batch_size=32, save_path='../dataset/parsed/doom/',sample_wgan=False): 
    file_path = save_path + 'data.tfrecords'
    metadata = read_json()
    if not os.path.isfile(file_path):
        print('No dataset record found')
        sys.exit()
    tfr_dataset = tf.data.TFRecordDataset(file_path)
    sample = parse_tfrecord(tfr_dataset,metadata,sample_wgan)
    # view_maps(sample)
    dataset = tfr_dataset.map(_parse_tfr_element)
    train_set, validation_set, val_batches = partition_dataset(dataset,metadata['count'],batch_size)
    # Returns a shuffled training set that is seperated into batches
    return train_set, validation_set, val_batches, metadata['maps_meta'], sample


def parse_tfrecord(record, meta, sample_wgan):
    # If adding the wgan generated maps as the sample
    if sample_wgan:
        save_path = '../dataset/generated/doom/hybrid/'
        file_path = save_path + 'sample.tfrecords'
        if not os.path.isfile(file_path):
            print('No dataset record found')
            sys.exit()
        record = tf.data.TFRecordDataset(file_path)
        map_keys = ['floormap','wallmap','heightmap']
        sample_id = 0
    else:
        map_keys = list(meta['maps_meta'].keys())
        sample_id = random.randrange(meta['count'])
    parse_dic = {
        key: tf.io.FixedLenFeature([], tf.string) for key in map_keys
        }
    features = dict()
    for i,element in enumerate(record):
        if i != sample_id:
            continue
        else:
            example_message = tf.io.parse_single_example(element, parse_dic)
            for key in map_keys:
                b_feature = example_message[key] # get byte string
                feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
                features[key] = feature
            break
    return features


# Read the TF Records
def _parse_tfr_element(element):
    metadata = read_json()
    map_keys = list(metadata['maps_meta'].keys())
    parse_dic = {
        key: tf.io.FixedLenFeature([], tf.string) for key in map_keys
        }
    example_message = tf.io.parse_single_example(element, parse_dic)
    features = dict()
    for key in map_keys:
        b_feature = example_message[key] # get byte string
        feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
        features[key] = feature
    return features


def partition_dataset(ds, ds_size, batch_size, train_split=5/6, val_split=1/6, shuffle=True):
    assert (train_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(ds_size*100, seed=12)
    
    train_size = int(train_split * ds_size)
    val_batches = math.floor((ds_size-train_size)/batch_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size)
    return train_ds.batch(batch_size, drop_remainder=True), val_ds.batch(batch_size, drop_remainder=True), val_batches


def normalize_maps(x, map_meta, map_names, use_sigmoid=True):
    """
    Compute the scaling of eve ry map based on their .meta statistics (max and min)
     to bring all values inside (0,1) or (-1,1)
    :param x: the input vector. shape(batch, width, height, len(map_names))
    :param map_names: the name of the feature maps for .meta file lookup
    :param use_sigmoid: if True data will be in range 0,1, if False it will be in -1;1
    :return: a normalized x vector
    """
    a = 0 if use_sigmoid else -1
    b = 1
    max = tf.constant([map_meta[m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    min = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    min_mat = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.cast(x>0,tf.float32)
    return a + ((x-min_mat)*(b-a))/(max-min)


def rescale_maps(x, map_meta, map_names, use_sigmoid=True):
    a = 0 if use_sigmoid else -1
    b = 1
    min_mat = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.cast(x>0,tf.float32)
    min = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    max = tf.constant([map_meta[m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    return tf.math.round(((max-min) * (x - a)/(b-a)) + min_mat)


def generate_loss_graph(loss, key, location = 'generated_maps/'):
    for i in range(len(loss)):
        plt.figure()
        plt.xlabel('Number of Epochs') if 'validation' in key[i] else plt.xlabel('Number of Steps')
        plt.ylabel('Loss')
        plt.plot(loss[i])
        if max(loss[i])>500:
            plt.ylim(top=500)
        # plt.legend() # must be after labels
        plt.savefig(location + key[i] + '_loss_graph')
        plt.close()


def generate_images(model, seed, epoch, keys, is_p2p = False, is_trad=False, test_input = None, test_keys = None, meta = None):
    prediction = model([test_input, seed], training=True) if is_p2p else model(seed, training=False)
    n = 128
    if is_p2p:
        scaled_pred = rescale_maps(prediction,meta,keys)
        for i in range(len(keys)):
            essentials = scaled_pred[0,:,:,i] if i == 0 else tf.maximum(essentials, scaled_pred[0,:,:,i])
        n = n/meta['essentials']['max']

    title = test_keys + ['essentials'] if is_p2p else keys
    display_list = [test_input[0,:,:,i] for i in range(len(test_keys))] + [essentials] if is_p2p else [prediction[0,:,:,i] for i in range(len(keys))]

    plt.figure(figsize=(8, 8))
    for i in range(len(title)):
        plt.subplot(2, 2, i+1)
        plt.title(title[i] if title[i] != 'essentials' else 'thingsmap')
        if keys[i] in ['thingsmap','essentials']:
            plt.imshow((display_list[i]*n)+(display_list[i]>0).astype(tf.float32)*127, cmap='gray') 
        else:
            plt.imshow(display_list[i], cmap='gray') 
        plt.axis('off')
    loc = 'generated_maps/hybrid/trad_pix2pix/' if is_trad else 'generated_maps/hybrid/pix2pix/' if is_p2p else 'generated_maps/wgan/' if 'essentials' in keys else 'generated_maps/hybrid/wgan/'
    if not os.path.exists(loc):
        os.makedirs(loc)
    plt.savefig(loc + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value is tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Created as it is unable to do it inline 
def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def generate_sample(imgs ,keys, save_path = '../dataset/generated/doom/', file_name = 'sample.tfrecords'):
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


def view_maps(sample):
    plt.subplot(2, 4, 1)
    plt.imshow(sample['floormap'], cmap='gray') 
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plt.imshow(sample['heightmap'], cmap='gray') 
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plt.imshow(sample['wallmap'], cmap='gray') 
    plt.axis('off')
    plt.subplot(2, 4, 4)
    plt.imshow((sample['monsters']*75/19)+(sample['monsters']>0).astype(tf.float32)*180, cmap='gray') 
    plt.axis('off')
    plt.subplot(2, 4, 5)
    plt.imshow((sample['ammunitions']*75/8)+(sample['ammunitions']>0).astype(tf.float32)*180, cmap='gray') 
    plt.axis('off')
    plt.subplot(2, 4, 6)
    plt.imshow((sample['powerups']*75/6)+(sample['powerups']>0).astype(tf.float32)*180, cmap='gray') 
    plt.axis('off')
    plt.subplot(2, 4, 7)
    plt.imshow((sample['artifacts']*75/9)+(sample['artifacts']>0).astype(tf.float32)*180, cmap='gray') 
    plt.axis('off')
    plt.subplot(2, 4, 8)
    plt.imshow((sample['weapons']*75/7)+(sample['weapons']>0).astype(tf.float32)*180, cmap='gray') 
    plt.axis('off')
    plt.show()