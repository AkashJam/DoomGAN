import sys
sys.path.insert(0,'..')
import tensorflow as tf
import time, os
from gan.NetworkArchitecture import topological_maps, object_maps, cGAN_gen, cGAN_disc
from gan.DataProcessing import normalize_maps, read_record
from gan.NNMeta import generate_images, generate_loss_graph, training_metrics
from gan.eval_metrics.metrics import encoding_error, mat_entropy, objs_per_unit_area, oob_error

def downsample(filters, kernel, stride, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding='same', kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())
  return result


def upsample(filters, kernel, stride, apply_dropout=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, kernel, strides=stride, padding='same', kernel_initializer=initializer, use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result


def Generator(n_inp, n_opt):
  inputs = tf.keras.layers.Input(shape=[256, 256, n_inp])
  noise = tf.keras.layers.Input(shape=[256, 256, 1])

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(n_opt, (8,8), strides=(4,4), padding='same', kernel_initializer=initializer, activation='relu')  # (batch_size, 256, 256, 5)

  x = tf.keras.layers.concatenate([inputs, noise])

  # Downsampling through the model
  skips = []
  for layer in cGAN_gen['downstack']:
    x = downsample(layer['n_filters'],layer['kernel_size'],layer['stride'],layer['norm'])(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for layer, skip in zip(cGAN_gen['upstack'], skips):
    x = upsample(layer['n_filters'],layer['kernel_size'],layer['stride'],layer['dropout'])(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  return tf.keras.Model(inputs=[inputs,noise], outputs=x)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target, input_imgs, LAMBDA=10):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  if mod:
    # Take floormap scaled to 0 or 1, invert the image and use it as a mask for the generated images 
    # to pick up objects generated outside the level bounds and add that to the loss
    floor_bool_mask = tf.cast(input_imgs[:,:,:,topological_maps.index('floormap')],tf.bool)
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

  else:
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss 


def Discriminator():
  inp = tf.keras.layers.Input(shape=[256, 256, n_tmaps], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, n_omaps], name='target_image')
  
  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)

  x = tf.keras.layers.concatenate([inp, tar])
  for layer in cGAN_disc:
    x = downsample(layer['n_filters'],layer['kernel_size'],layer['stride'],layer['norm'])(x)

  x = last(x)  # (batch_size, 1, 1, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=x)


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

  
def train_step(input_image, target):
  noisy_img = tf.random.normal(shape=(batch_size, tf.shape(input_image)[1], tf.shape(input_image)[2], 1))
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator([input_image, noisy_img], training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    if mod:
      gen_total_loss, gen_gan_loss, gen_mask_loss, gen_obj_loss = generator_loss(disc_generated_output, gen_output, target, input_image)
    else:
      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, input_image, LAMBDA=100)

    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
  
  return {"d_loss": tf.abs(disc_loss), "g_loss": tf.abs(gen_total_loss), "gan_loss": tf.abs(gen_gan_loss), "mask_loss": tf.abs(gen_mask_loss), 
          "obj_loss": tf.abs(gen_obj_loss)} if mod else  {"d_loss": tf.abs(disc_loss), "g_loss": tf.abs(gen_total_loss), 
                                                          "gan_loss": tf.abs(gen_gan_loss), "l1_loss": tf.abs(gen_l1_loss)}

def v_loss(n_batches):
  v_gloss = list()
  v_dloss = list()
  enc_arr = list()
  ntp_arr = list()
  oob_arr = list()
  obj_arr =list()
  for image_batch in validation_set:
    v_input = tf.stack([image_batch[m] for m in topological_maps], axis=-1)
    v_target = tf.stack([image_batch[m] for m in object_maps], axis=-1)
    x_input = normalize_maps(v_input, map_meta, topological_maps)
    x_target = normalize_maps(v_target, map_meta, object_maps)

    noisy_img = tf.random.normal(shape=(batch_size, tf.shape(x_input)[1], tf.shape(x_input)[2], 1))
    gen_output = generator([x_input, noisy_img], training=True)

    disc_real_output = discriminator([x_input, x_target], training=False)
    disc_generated_output = discriminator([x_input, gen_output], training=False)

    if mod:
      gen_total_loss, gen_gan_loss, gen_mask_loss, gen_obj_loss = generator_loss(disc_generated_output, gen_output, x_target, x_input)
    else:
      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, x_target, x_input, LAMBDA=100)

    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    v_gloss.append(gen_total_loss)
    v_dloss.append(disc_loss)

    enc_arr.append(encoding_error(gen_output, map_meta))
    ntp_arr.append(mat_entropy(gen_output, map_meta))
    id = topological_maps.index('floormap')
    oob_arr.append(oob_error(gen_output, x_input[:,:,:,id]))
    obj_arr.append(objs_per_unit_area(gen_output, x_input[:,:,:,id]))
  training_metrics(n_batches, mod, enc_arr, ntp_arr, oob_arr, obj_arr)

  avg_gloss = sum(v_gloss)/len(v_gloss)
  avg_dloss = sum(v_dloss)/len(v_dloss)
  return  avg_gloss, avg_dloss


def train(epochs):
  start = time.time()
  gen_ts_loss = list()
  disc_ts_loss = list()
  gen_vs_loss = list()
  disc_vs_loss = list()
  sample_input = tf.stack([sample[m] for m in topological_maps], axis=-1).reshape((1, 256, 256, len(topological_maps)))
  scaled_sample_input = normalize_maps(sample_input, map_meta, topological_maps)
  seed = tf.random.normal([1, tf.shape(sample_input)[1], tf.shape(sample_input)[2], 1])

  for epoch in range(epochs):
    start = time.time()
    for n, image_batch in training_set.enumerate().as_numpy_iterator():
      input = tf.stack([image_batch[m] for m in topological_maps], axis=-1)
      target = tf.stack([image_batch[m] for m in object_maps], axis=-1)
      x_input = normalize_maps(input, map_meta, topological_maps)
      x_target = normalize_maps(target, map_meta, object_maps)
      step_loss = train_step(x_input, x_target)
      disc_ts_loss.append(step_loss['d_loss'])
      gen_ts_loss.append(step_loss['g_loss'])
      print ('Time for batch {} is {} sec. d_loss: {} g_loss: {}'.format(n+1, time.time()-start,step_loss['d_loss'],step_loss['g_loss']))
      total_batches = len(disc_ts_loss)
      if total_batches%100 == 0:
        v_dloss, v_gloss= v_loss(total_batches)
        gen_vs_loss.append(v_gloss)
        disc_vs_loss.append(v_dloss)

    if (epoch+1)%10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    loc = 'generated_maps/hybrid/mod_cgan/' if mod else 'generated_maps/hybrid/trad_cgan/'
    generate_images(generator, seed, epoch+1, object_maps, is_cgan=True, is_mod=mod, test_input=scaled_sample_input, meta=map_meta)
    generate_loss_graph(disc_ts_loss, disc_vs_loss, 'discriminator', model,  location = loc)
    generate_loss_graph(gen_ts_loss, gen_vs_loss, 'generator', model, location = loc)
    print ('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))


if __name__ == "__main__":
  batch_size = 1
  mod = True
  n_tmaps = len(topological_maps)
  n_omaps = len(object_maps)
  model = 'Modified cGAN' if mod else 'Traditional cGAN'

  training_set, validation_set, map_meta, sample = read_record(batch_size, sample_wgan=False)
  generator = Generator(n_tmaps,n_omaps)
  discriminator = Discriminator()
  generator_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5, beta_2=0.999)
  discriminator_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5, beta_2=0.999)

  checkpoint_dir = './training_checkpoints/hybrid/mod_cgan' if mod else './training_checkpoints/hybrid/trad_cgan'
  if not os.path.exists(checkpoint_dir+'/'):
    os.makedirs(checkpoint_dir+'/')
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)
  train(100)