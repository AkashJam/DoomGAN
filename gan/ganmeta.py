import tensorflow as tf
import sys, os, json, random
from matplotlib import pyplot as plt
import numpy as np
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

def read_record(batch_size=32, save_path='../dataset/parsed/doom/'): 
    file_path = save_path + 'data.tfrecords'
    metadata = read_json()
    if not os.path.isfile(file_path):
        print('No dataset record found')
        sys.exit()
    tfr_dataset = tf.data.TFRecordDataset(file_path)
    sample = parse_tfrecord(tfr_dataset,metadata)
    train_set = tfr_dataset.map(_parse_tfr_element)
    # Returns a shuffled training set that is seperated into batches
    return train_set.shuffle(metadata['count']*100).batch(batch_size, drop_remainder=True), metadata['maps_meta'], sample


# Read TF Records and View the scaled maps
def parse_tfrecord(record,meta):
    dataset = list()
    map_keys = list(meta['maps_meta'].keys())
    map_size = [256,256]
    parse_dic = {
        key: tf.io.FixedLenFeature([], tf.string) for key in map_keys
        }
    features = dict()
    sample_id = random.randrange(meta['count'])
    for i,element in enumerate(record):
        if i != sample_id:
            continue
        else:
            example_message = tf.io.parse_single_example(element, parse_dic)
            for key in map_keys:
                b_feature = example_message[key] # get byte string
                feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
                features[key] = feature
            # plt.figure(figsize=(8, 8))
            # plt.subplot(2, 2, 1)
            # plt.imshow(features['floormap']* 127.5 + 127.5, cmap='gray')
            # plt.axis('off')
            # plt.subplot(2, 2, 2)
            # plt.imshow(features['wallmap']* 127.5 + 127.5, cmap='gray')
            # plt.axis('off')
            # plt.subplot(2, 2, 3)
            # plt.imshow(features['essentials']* 127.5 + 127.5, cmap='gray')
            # plt.axis('off')
            # plt.subplot(2, 2, 4)
            # plt.imshow(features['heightmap']* 127.5 + 127.5, cmap='gray')
            # plt.axis('off')
            # plt.show()
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


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  plt.figure(figsize=(8, 8))
  for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(predictions[0, :, :, i])
    # plt.imshow(predictions[0, :, :, i] * 127.5 + 127.5, cmap='gray') 
    plt.axis('off')

  plt.savefig('generated_maps/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close()


def scaling_maps(x, map_meta, map_names, use_sigmoid=True):
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

    return a + ((x-min)*(b-a))/(max-min)


def generate_loss_graph(d_loss,g_loss,location = 'generated_maps/convergence_graph.png'):
    plt.figure
    plt.title('Convergence Graph')
    plt.xlabel('Number of Steps')
    plt.ylabel('Loss')
    plt.plot(d_loss, label='Dis Loss')
    plt.plot(g_loss, label='Gen Loss')
    plt.legend() # must be after labels
    plt.savefig(location)
    plt.close()
