import tensorflow as tf
import os, json, random, sys


def read_json(save_path = 'dataset/parsed/doom/'):
    file_path = save_path + 'metadata.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as jsonfile:
            map_meta = json.load(jsonfile)
        return map_meta
    elif os.path.isfile('../'+file_path):
        with open('../'+file_path, 'r') as jsonfile:
            map_meta = json.load(jsonfile)
        return map_meta
    else:
        print('No metadata found')
        sys.exit()


def read_record(batch_size=32, save_path='dataset/parsed/doom/', sample_wgan=False): 
    file_path = save_path + 'data.tfrecords'
    metadata = read_json()
    if not os.path.isfile(file_path):
        print('No dataset record found')
        sys.exit()
    tfr_dataset = tf.data.TFRecordDataset(file_path)
    sample = parse_tfrecord(tfr_dataset,metadata,sample_wgan)
    # view_maps(sample)
    dataset = tfr_dataset.map(_parse_tfr_element)
    train_set, validation_set = partition_dataset(dataset,metadata['count'],batch_size)
    # Returns a shuffled training set that is seperated into batches
    return train_set, validation_set, metadata['maps_meta'], sample


def parse_tfrecord(record, meta, sample_wgan, file_path='dataset/generated/doom/hybrid/sample.tfrecords'):
    # If adding the wgan generated maps as the sample
    if sample_wgan:
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


def partition_dataset(ds, ds_size, batch_size, train_split=0.8, val_split=0.2, shuffle=True):
    assert (train_split + val_split) == 1
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(ds_size*100, seed=12)
    train_size = int(train_split * ds_size)
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size)
    return train_ds.batch(batch_size, drop_remainder=True), val_ds.batch(batch_size, drop_remainder=True)


def downsample(filters, kernel, stride, norm, bias=True, kernel_initializer = 'glorot_uniform'):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding='same', use_bias=bias, kernel_initializer=kernel_initializer))
    if norm == 'batch':
        result.add(tf.keras.layers.BatchNormalization())
    if norm == 'layer':
        result.add(tf.keras.layers.LayerNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, kernel, stride, dropout, bias=True, kernel_initializer = 'glorot_uniform'):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, kernel, strides=stride, padding='same', use_bias=bias, kernel_initializer=kernel_initializer))
    result.add(tf.keras.layers.BatchNormalization())
    if dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.LeakyReLU())
    return result