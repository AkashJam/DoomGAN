import tensorflow as tf
import sys, os, json
from matplotlib import pyplot as plt

save_path = '../dataset/parsed/doom/'

def read_json():
    file_path = save_path + 'metadata.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as jsonfile:
            map_meta = json.load(jsonfile)
        return map_meta['key_meta'], map_meta['count']
    else:
        print('No metadata found')
        sys.exit()

def read_record(batch_size=32): 
    file_path = save_path + 'dataset.tfrecords'
    metadata, train_count = read_json()
    if not os.path.isfile(file_path):
        print('No dataset record found')
        sys.exit()
    tfr_dataset = tf.data.TFRecordDataset(file_path)
    train_set = tfr_dataset.map(_parse_tfr_element)
    # Returns a shuffled training set that is seperated into batches
    return train_set.shuffle(train_count*100).batch(batch_size, drop_remainder=True), metadata

# Read TFRecord file
def _parse_tfr_element(element):
    metadata, count = read_json()
    map_size = [128, 128]
    map_keys = list(metadata.keys())
    parse_dic = {
        key: tf.io.FixedLenFeature([], tf.string) for key in map_keys # Note that it is tf.string, not tf.float32
        }
    example_message = tf.io.parse_single_example(element, parse_dic)
    features = dict()
    for key in map_keys:
        b_feature = example_message[key] # get byte string
        feature = tf.io.parse_tensor(b_feature, out_type=tf.uint8) # restore 2D array from byte string
        features[key] = feature
    unscaled_feat = tf.stack([features[key] for key in map_keys], axis=-1)
    scaled_feat = tf.image.resize_with_pad(unscaled_feat, map_size[0], map_size[1])
    for i,key in enumerate(map_keys):
        features[key] = scaled_feat[:,:,i]
    return features

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(6, 4))

  for i in range(6):
      plt.subplot(3, 2, i+1)
      plt.imshow(predictions[0, :, :, i] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('generated_maps/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close()


def scaling_maps(x, map_meta, map_names, use_sigmoid=True):
    """
    Compute the scaling of every map based on their .meta statistics (max and min)
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

def generate_loss_graph(c_loss,g_loss):
    epoch_list = [n+1 for n in range(len(c_loss))]
    plt.figure
    plt.title('Convergence Graph')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.plot(epoch_list,c_loss,label='Critic Loss')
    plt.plot(epoch_list,g_loss,label='Generator Loss')
    plt.savefig('generated_maps/convergence_graph.png')
    plt.close()
    # plt.show()
