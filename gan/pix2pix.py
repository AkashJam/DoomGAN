import tensorflow as tf
import time, os
from GanMeta import read_record, normalize_maps, generate_images, generate_loss_graph

def downsample(filters, size, stride, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same', kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())
  return result


def upsample(filters, size, stride, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=stride, padding='same', kernel_initializer=initializer, use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])
  noise = tf.keras.layers.Input(shape=[256, 256, 1])

  down_stack = [
    downsample(128, (8,8), (4,4), apply_batchnorm=False),  # (batch_size, 64, 64, 128)
    downsample(256, (8,8), (4,4)),  # (batch_size, 16, 16, 256)
    downsample(512, (8,8), (4,4)),  # (batch_size, 4, 4, 512)
    downsample(1024, (8,8), (4,4)),  # (batch_size, 1, 1, 1024)
  ]

  up_stack = [
    upsample(1024, (8,8), (4,4), apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, (8,8), (4,4), apply_dropout=True),  # (batch_size, 16, 16, 512)
    upsample(256, (8,8), (4,4), apply_dropout=True),  # (batch_size, 64, 64, 256)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(5, (8,8), strides=(4,4), padding='same', kernel_initializer=initializer, activation='relu')  # (batch_size, 256, 256, 5)

  x = tf.keras.layers.concatenate([inputs, noise])

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
  return tf.keras.Model(inputs=[inputs,noise], outputs=x)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target, input_imgs, LAMBDA=10):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Take floormap scaled to 0 or 1, invert the image and use it as a mask for the generated images 
  # to pick up objects generated outside the level bounds and add that to the loss
  floor_bool_mask = tf.cast(input_imgs[:,:,:,input_params.index('floormap')],tf.bool)
  not_floor_mask = tf.reshape(tf.cast(tf.logical_not(floor_bool_mask),tf.float32),[input_imgs.shape[0],input_imgs.shape[1],input_imgs.shape[2],1])
  gen_mask = tf.cast(tf.cast(gen_output,tf.bool),tf.float32)
  
  mask_loss = tf.reduce_sum(not_floor_mask*gen_mask)
  mask_loss = tf.math.log(mask_loss+1)
  
  objs_loss = list()
  for i in range(gen_output.shape[3]):
    ase = tf.abs(tf.reduce_sum(target[:,:,:,i] - gen_output[:,:,:,i]))
    objmap_loss = LAMBDA * tf.math.log(1 + ase/(tf.reduce_sum(target[:,:,:,i]) + 1))
    objs_loss.append(objmap_loss)
  obj_loss = tf.reduce_mean(objs_loss)

  total_gen_loss = gan_loss + mask_loss + obj_loss
  return total_gen_loss, gan_loss, mask_loss, obj_loss


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)
  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 5], name='target_image')
  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, 8)

  down1 = downsample(128, (8,8), (4,4), False)(x)  # (batch_size, 64, 64, 128)
  down2 = downsample(256, (8,8), (4,4))(down1)  # (batch_size, 16, 16, 256)
  down3 = downsample(512, (4,4), (2,2))(down2)  # (batch_size, 4, 4, 512)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 6, 6, 512)
  conv = tf.keras.layers.Conv2D(1024, (4,4), strides=(1,1), kernel_initializer=initializer, use_bias=False)(zero_pad1)  # (batch_size, 3, 3, 1024)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 4, 4, 1024)
  last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (batch_size, 1, 1, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

  
def train_step(input_image, target):
  batch_size = tf.shape(input_image)[0]
  noise_vectors = tf.random.normal(shape=(batch_size, tf.shape(input_image)[1], tf.shape(input_image)[2], 1))
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator([input_image, noise_vectors], training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_mask_loss, gen_obj_loss = generator_loss(disc_generated_output, gen_output, target, input_image)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
  
  return {"d_loss": tf.abs(disc_loss), "g_loss": tf.abs(gen_total_loss), "gan_loss": tf.abs(gen_gan_loss), "mask_loss": tf.abs(gen_mask_loss), "obj_loss": tf.abs(gen_obj_loss)}


def train(epochs):
  start = time.time()
  disc_ts_loss = list()
  gen_ts_loss = list()
  sample_input = tf.stack([sample[m] for m in input_params], axis=-1).reshape((1, 256, 256, len(input_params)))
  scaled_sample_input = normalize_maps(sample_input, map_meta, input_params)
  seed = tf.random.normal([1, tf.shape(sample_input)[1], tf.shape(sample_input)[2], 1])
  for epoch in range(epochs):
    start = time.time()
    for n, image_batch in training_set.enumerate().as_numpy_iterator():
      input = tf.stack([image_batch[m] for m in input_params], axis=-1)
      target = tf.stack([image_batch[m] for m in output_params], axis=-1)
      x_input = normalize_maps(input, map_meta, input_params)
      x_target = normalize_maps(target, map_meta, output_params)
      step_loss = train_step(x_input, x_target)
      disc_ts_loss.append(step_loss['d_loss'])
      gen_ts_loss.append(step_loss['g_loss'])
      print ('Time for batch {} is {} sec. gan_loss: {} mask_loss: {} obj_loss: {}'.format(n, time.time()-start,step_loss['gan_loss'],step_loss['mask_loss'],step_loss['obj_loss']))
    generate_images(generator, seed, epoch+1, output_params, is_p2p = True, test_input = scaled_sample_input, meta = map_meta, test_keys=input_params)

    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      generate_loss_graph([disc_ts_loss, gen_ts_loss], ['disc','gen'], location = 'generated_maps/hybrid/pix2pix/')

    print ('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))

if __name__ == "__main__":
  batch_size = 1
  input_params = ['floormap', 'wallmap', 'heightmap']
  output_params = ['monsters','ammunitions','powerups','artifacts','weapons']

  training_set, map_meta, sample = read_record(batch_size, sample_wgan=True)
  generator = Generator()
  discriminator = Discriminator()
  generator_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5, beta_2=0.999)
  discriminator_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5, beta_2=0.999)

  checkpoint_dir = './training_checkpoints/hybrid/pix2pix'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

  train(101)