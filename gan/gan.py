import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
import os, sys, time, math, json
import numpy as np

save_path = '../dataset/parsed/doom/'

def read_json(path):
    file_path = path + 'metadata.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as jsonfile:
            map_meta = json.load(jsonfile)
        return map_meta['key_meta'], map_meta['count']
    else:
        print('No metadata found')
        return None, None

# Read TFRecord file
def _parse_tfr_element(element):
    map_meta, count = read_json(save_path)
    map_size = [128, 128]
    map_keys = list(map_meta.keys())
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


# Build the network
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((100,), batch_size=32))
    model.add(layers.Dense(8*8*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((8, 8, 1024))) #Check if you can reduce the dense layer and rechape it

    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (32, 16, 16, 512)  # Note: None is the batch size
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (32, 32, 32, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (32, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(6, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    assert model.output_shape == (32, 128, 128, 6)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((128, 128, 6), batch_size=32))
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[128, 128, 6]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    # model.add(layers.LayerNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    # model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # print('CE', tf.shape(cross_entropy(tf.ones_like(fake_output), fake_output)))
    # print('WD',tf.shape(-tf.reduce_mean(fake_output)))
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Train

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

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
    max = tf.constant([map_meta[m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape()) #replace floormap for m in map_names for all 4 maps
    min = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    return a + ((x-min)*(b-a))/(max-min)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, map_meta, seed):
    map_keys = list(map_meta.keys())
    for epoch in range(epochs):
        start = time.time() 

        for i,image_batch in enumerate(dataset):
            for rotation in [0, 90, 180, 270]:
                images = np.stack([image_batch[m] for m in map_keys], axis=-1)
                scaled_images = scaling_maps(images, map_meta, map_keys)
                x_batch = tfa.image.rotate(scaled_images, math.radians(rotation))
                train_step(x_batch)
                x_batch = tf.image.flip_left_right(x_batch)
                train_step(x_batch)

            print ('Time for batch {} is {} sec'.format(i+1, time.time()-start))
            
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                            epochs,
                            seed)


EPOCHS = 200
noise_dim = 100
num_examples_to_generate = 1
batch_size = 32
seed = tf.random.normal([num_examples_to_generate, noise_dim])


file_name = 'dataset.tfrecords'
file_path = save_path + file_name
metadata, train_count = read_json(save_path)
if metadata is None and not os.path.isfile(file_path):
    sys.exit()

tfr_dataset = tf.data.TFRecordDataset(file_path)
train_set = tfr_dataset.map(_parse_tfr_element)
training_set = train_set.shuffle(train_count*100).batch(batch_size)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

train(training_set, EPOCHS, metadata, seed)
