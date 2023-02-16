import tensorflow as tf
import sys, os, json, math
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def read_json(save_path = '../dataset/parsed/doom/'):
    file_path = save_path + 'metadataset.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as jsonfile:
            map_meta = json.load(jsonfile)
        return map_meta
    else:
        print('No metadata found')
        sys.exit()

def read_record(batch_size=32, save_path='../dataset/parsed/doom/'): 
    file_path = save_path + 'dataset.tfrecords'
    metadata = read_json()
    if not os.path.isfile(file_path):
        print('No dataset record found')
        sys.exit()
    tfr_dataset = tf.data.TFRecordDataset(file_path)
    # train_set = parse_tfrecord(tfr_dataset,metadata)
    train_set = tfr_dataset.map(_parse_tfr_element)
    # Returns a shuffled training set that is seperated into batches
    return train_set.shuffle(metadata['count']*100).batch(batch_size, drop_remainder=True), metadata['maps_meta']


# Read TF Records and View the scaled maps
def parse_tfrecord(record,meta):
    dataset = list()
    map_keys = list(meta['maps_meta'].keys())
    map_size = [256,256]
    parse_dic = {
        key: tf.io.FixedLenFeature([], tf.string) for key in map_keys
        }
    features = dict()
    # print(meta['max_height'],meta['max_width'],meta['min_height'],meta['min_width'])
    for element in record:
        example_message = tf.io.parse_single_example(element, parse_dic)
        for key in map_keys:
            b_feature = example_message[key] # get byte string
            feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
            features[key] = feature
        unscaled_feat = tf.stack([features[key] for key in map_keys], axis=-1)
        scaled_feat = tf.image.resize_with_pad(unscaled_feat, map_size[0], map_size[1], method='area') # resize the maps to specifed pixels
        feats = dict()
        for i,key in enumerate(map_keys):
            feats[key] = scaled_feat[:,:,i]

        feats['floormap'] = (feats['floormap']>0).astype(np.uint8)*255
        feats['wallmap'] = (feats['wallmap']>0).astype(np.uint8)*255
        # tf.map_fn(fn=lambda t: [1 if n>0 else 0 for n in t], elems=feats['wallmap'])
        plt.figure(figsize=(16, 8))
        plt.subplot(2, 3, 1)
        plt.title('thingsmap')
        plt.imshow(features['thingsmap'])
        plt.subplot(2, 3, 2)
        plt.title('wallmap')
        plt.imshow(features['wallmap'])
        plt.subplot(2, 3, 3)
        plt.title('monsters')
        plt.imshow(features['monsters'])
        plt.subplot(2, 3, 4)
        plt.title('weapons')
        plt.imshow(features['weapons'])
        plt.subplot(2, 3, 5)
        plt.title('ammunitions')
        plt.imshow(features['ammunitions'])
        plt.subplot(2, 3, 6)
        plt.title('other')
        plt.imshow(features['other'])
        plt.show()
        dataset.append(scaled_feat)
        break

    train_set = tf.data.Dataset.from_tensors(dataset)
    return train_set


# Read the TF Records
def _parse_tfr_element(element):
    metadata = read_json()
    map_keys = list(metadata['maps_meta'].keys())
    # map_size = [256, 256] 
    parse_dic = {
        key: tf.io.FixedLenFeature([], tf.string) for key in map_keys
        }
    example_message = tf.io.parse_single_example(element, parse_dic)
    features = dict()
    for key in map_keys:
        b_feature = example_message[key] # get byte string
        feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
        features[key] = feature
    # unscaled_feat = tf.stack([features[key] for key in map_keys], axis=-1)
    # scaled_feat = tf.image.resize_with_pad(unscaled_feat, map_size[0], map_size[1], method='area') # resize the maps to specifed pixels
    # for i,key in enumerate(map_keys):
    #     if key in ['floormap','wallmap']:
    #         features[key] = (scaled_feat[:,:,i]>0).astype(np.uint8)*255
    #     else:
    #         features[key] = scaled_feat[:,:,i]
    return features


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(8, 8))

  for i in range(4):
      plt.subplot(2, 2, i+1)
      plt.imshow(predictions[0, :, :, i] * 127.5 + 127.5, cmap='gray')
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
    # plt.show()

