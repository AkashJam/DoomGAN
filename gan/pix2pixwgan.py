import tensorflow as tf
import tensorflow_addons as tfa
import time, math
import numpy as np
from matplotlib import pyplot as plt
from ganmeta import read_record, scaling_maps

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[128, 128, 2])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 64, 64, 64)
    downsample(128, 4),  # (batch_size, 32, 32, 128)
    downsample(256, 4),  # (batch_size, 16, 16, 256)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(256, 4),  # (batch_size, 16, 16, 512)
    upsample(128, 4),  # (batch_size, 32, 32, 256)
    upsample(64, 4),  # (batch_size, 64, 64, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 128, 128, 1)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, LAMBDA = 100):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[128, 128, 2], name='input_image')
  tar = tf.keras.layers.Input(shape=[128, 128, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 128, 128, 3)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 64, 64, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 32, 32, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 16, 16, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 18, 18, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 15, 15, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def generate_images(model, test_input, tar, epoch, batch):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0,:,:,0],test_input[0,:,:,1], tar[0], prediction[0]]
  title = ['Height Map', 'Wall Map', 'Ground Truth', 'Predicted Image']

  for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig('generated_maps/things_map/image_at_batch_{:04d}_in_epoch_{:04d}.png'.format(batch,epoch))
  plt.close()
#   plt.show()


def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

def train(dataset, map_meta, epochs):
  start = time.time()

  for epoch in range(epochs):
        start = time.time() 
        cl_per_batch = list()
        gl_per_batch = list()
        for i,image_batch in enumerate(dataset):
            for rotation in [0, 90, 180, 270]:
                input = np.stack([image_batch[m] for m in ['heightmap','wallmap']], axis=-1)
                target = np.stack(image_batch['thingsmap']).reshape((32, 128, 128, 1))
                scaled_input = scaling_maps(input, map_meta, ['heightmap','wallmap'])
                scaled_target = scaling_maps(target, map_meta, ['thingsmap'])
                # Rotating images to account for different orientations
                x_input = tfa.image.rotate(scaled_input, math.radians(rotation))
                x_target = tfa.image.rotate(scaled_target, math.radians(rotation))
                for flip in [0, 1]:
                    if flip:
                        x_input = tf.image.flip_left_right(x_input)
                        x_target = tf.image.flip_left_right(x_target)
                    train_step(x_input, x_target)
            
            print ('Time for batch {} of epoch {} is {} sec'.format(i+1, epoch+1, time.time()-start))
            generate_images(generator, x_input, x_target, epoch+1, i+1)

training_set, map_meta = read_record()
train(training_set, map_meta, epochs=1000)