import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
import os, sys, time, math, json
import numpy as np


def read_json(meta_path):
    file_path = meta_path + 'metadata.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as jsonfile:
            map_meta = json.load(jsonfile)
        return map_meta['key_meta'], map_meta['count']
    else:
        print('No metadata found')
        sys.exit()

def read_record(save_path,file_name,batch_size): 
    file_path = save_path + file_name
    metadata, train_count = read_json(save_path)
    if not os.path.isfile(file_path):
        print('No dataset record found')
        sys.exit()
    tfr_dataset = tf.data.TFRecordDataset(file_path)
    train_set = tfr_dataset.map(_parse_tfr_element)
    return train_set.shuffle(train_count*100).batch(batch_size), metadata

# Read TFRecord file
def _parse_tfr_element(element):
    metadata, count = read_json(save_path)
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


# Build the network
def make_generator_model(b_size,z_dim):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((z_dim,), batch_size=b_size))
    model.add(layers.Dense(8*8*1024, use_bias=False, input_shape=(z_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((8, 8, 1024))) #Check if you can reduce the dense layer and rechape it

    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (b_size, 16, 16, 512)  # Note: None is the batch size
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (b_size, 32, 32, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (b_size, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(6, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (b_size, 128, 128, 6)

    return model


def make_discriminator_model(b_size):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((128, 128, 6), batch_size=b_size))
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[128, 128, 6]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LayerNormalization())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LayerNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# This method returns a helper function to compute cross entropy loss
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


# Train

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
    max = tf.constant([map_meta[m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape()) #replace floormap for m in map_names for all 4 maps
    min = tf.constant([map_meta[m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(tf.convert_to_tensor(x).get_shape())
    return a + ((x-min)*(b-a))/(max-min)

def gradient_penalty(batch_size, real_images, fake_images):
    """Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def train_step(real_images,latent_dim):
    discriminator_extra_steps = 3
    gp_weight = 10.0
    if isinstance(real_images, tuple):
        real_images = real_images[0]

    # Get the batch size
    batch_size = tf.shape(real_images)[0]

    # For each batch, we are going to perform the
    # following steps as laid out in the original paper:
    # 1. Train the generator and get the generator loss
    # 2. Train the discriminator and get the discriminator loss
    # 3. Calculate the gradient penalty
    # 4. Multiply this gradient penalty with a constant weight factor
    # 5. Add the gradient penalty to the discriminator loss
    # 6. Return the generator and discriminator losses as a loss dictionary

    # Train the discriminator first. The original paper recommends training
    # the discriminator for `x` more steps (typically 5) as compared to
    # one step of the generator. Here we will train it for 3 extra steps
    # as compared to 5 to reduce the training time.
    for i in range(discriminator_extra_steps):
        # Get the latent vector
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, latent_dim)
        )
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = generator(random_latent_vectors, training=True)
            # Get the logits for the fake images
            fake_logits = discriminator(fake_images, training=True)
            # Get the logits for the real images
            real_logits = discriminator(real_images, training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost = discriminator_loss(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp = gradient_penalty(batch_size, real_images, fake_images)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * gp_weight

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        discriminator_optimizer.apply_gradients(
            zip(d_gradient, discriminator.trainable_variables)
        )

    # Train the generator
    # Get the latent vector
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    with tf.GradientTape() as tape:
        # Generate fake images using the generator
        generated_images = generator(random_latent_vectors, training=True)
        # Get the discriminator logits for fake images
        gen_img_logits = discriminator(generated_images, training=True)
        # Calculate the generator loss
        g_loss = generator_loss(gen_img_logits)

    # Get the gradients w.r.t the generator loss
    gen_gradient = tape.gradient(g_loss, generator.trainable_variables)
    # Update the weights of the generator using the generator optimizer
    generator_optimizer.apply_gradients(
        zip(gen_gradient, generator.trainable_variables)
    )
    return {"d_loss": d_loss, "g_loss": g_loss}

def generate_loss_graph(epochs,c_loss,g_loss):
    epoch_list = [n+1 for n in range(epochs)]
    plt.figure
    plt.title('Convergence Graph')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.plot(epoch_list,c_loss,label='Critic Loss')
    plt.plot(epoch_list,g_loss,label='Generator Loss')
    # plt.savefig('generated_maps/convergence_graph.png')
    # plt.close()
    plt.show()

def train(dataset, map_meta, epochs, z_dim):

    critic_loss = list()
    gen_loss = list()
    # 1 is th enumber of examples to generate
    seed = tf.random.normal([1, z_dim])
    map_keys = list(map_meta.keys())
    # Start training
    for epoch in range(epochs):
        start = time.time() 
        cl_per_batch = list()
        gl_per_batch = list()
        for i,image_batch in enumerate(dataset):
            for flip in [0, 1]:
                for rotation in [0, 90, 180, 270]:
                    images = np.stack([image_batch[m] for m in map_keys], axis=-1)
                    scaled_images = scaling_maps(images, map_meta, map_keys)
                    x_batch = tfa.image.rotate(scaled_images, math.radians(rotation))
                    if flip:
                        x_batch = tf.image.flip_left_right(x_batch)
                    step_loss = train_step(x_batch,z_dim)
                    critic_step_loss = step_loss['g_loss']
                    gen_step_loss = step_loss['d_loss']
                    print(critic_step_loss,gen_step_loss)
                    cl_per_batch.append(critic_step_loss)
                    gl_per_batch.append(gen_step_loss)
            print ('Time for batch {} is {} sec'.format(i+1, time.time()-start))
            
        cl_per_epoch = sum(cl_per_batch)/len(cl_per_batch)
        gl_per_epoch = sum(gl_per_batch)/len(gl_per_batch)
        critic_loss.append(cl_per_epoch)
        gen_loss.append(gl_per_epoch)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


    generate_loss_graph(epochs, critic_loss, gen_loss)

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)


noise_dim = 100
batch_size = 32
save_path = '../dataset/parsed/doom/'
file_name = 'dataset.tfrecords'
training_set, map_meta = read_record(save_path,file_name,batch_size)


generator = make_generator_model(batch_size,noise_dim)
discriminator = make_discriminator_model(batch_size)

generator_optimizer = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
# if os.path.exists(checkpoint_dir):                            
#     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


EPOCHS = 500
train(training_set,map_meta,EPOCHS,noise_dim)
