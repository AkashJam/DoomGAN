import tensorflow as tf
import time, os
import numpy as np
from matplotlib import pyplot as plt
from GanMeta import read_record, scaling_maps, generate_loss_graph, rescale_maps
# import tensorflow_model_optimization as tfmot

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
    # x = tfmot.sparsity.keras.prune_low_magnitude(up,pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.99, 1,-1, 100))(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  # x = tfmot.sparsity.keras.prune_low_magnitude(last,pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.99, 1,-1, 100))(x)
  return tf.keras.Model(inputs=[inputs,noise], outputs=x)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, input_imgs, LAMBDA=10):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Take floormap scaled to 0 or 1, invert the image and use it as a mask for the generated images 
  # to pick up objects generated outside the level bounds and add that to the loss
  id = input_params.index('floormap')
  floor_bool_mask = tf.cast(input_imgs[:,:,:,id],tf.bool)
  floor_mask = tf.cast(floor_bool_mask,tf.float32)
  floor_mask = tf.reshape(floor_mask,[input_imgs.shape[0],input_imgs.shape[1],input_imgs.shape[2],1])
  not_floor_mask = tf.cast(tf.logical_not(floor_bool_mask),tf.float32)
  not_floor_mask = tf.reshape(not_floor_mask,[input_imgs.shape[0],input_imgs.shape[1],input_imgs.shape[2],1])
  target_mask = tf.cast(tf.cast(target,tf.bool),tf.float32)
  gen_mask = tf.cast(tf.cast(gen_output,tf.bool),tf.float32)
  
  mask_loss = tf.reduce_sum(not_floor_mask*gen_mask)
  mask_loss = tf.math.log(mask_loss+1)
  
  objs_loss = list()
  for i in range(gen_output.shape[3]):
    ase = tf.abs(tf.reduce_sum(target[:,:,:,i] - gen_output[:,:,:,i]))
    objmap_loss = LAMBDA*tf.math.log(1+ase/(tf.reduce_sum(target[:,:,:,i])+1))
    objs_loss.append(objmap_loss)
  obj_loss = tf.reduce_mean(objs_loss)

  print('actual monsters: {} ammunitions: {} powerups: {} artifacts: {} weapons: {} items to size {}'.format(tf.reduce_sum(target_mask[0,:,:,0]),tf.reduce_sum(target_mask[0,:,:,1]),
                                                                                            tf.reduce_sum(target_mask[0,:,:,2]),tf.reduce_sum(target_mask[0,:,:,3]),
                                                                                            tf.reduce_sum(target_mask[0,:,:,4]),tf.reduce_sum(target_mask)*tf.reduce_mean(floor_mask)))
  print('generated monsters: {} ammunitions: {} powerups: {} artifacts: {} weapons: {} items to size {}'.format(tf.reduce_sum(gen_mask[0,:,:,0]),tf.reduce_sum(gen_mask[0,:,:,1]),
                                                                                     tf.reduce_sum(gen_mask[0,:,:,2]),tf.reduce_sum(gen_mask[0,:,:,3]),
                                                                                     tf.reduce_sum(gen_mask[0,:,:,4]),tf.reduce_sum(gen_mask)*tf.reduce_mean(floor_mask)))
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


def generate_images(model, test_input, seed, epoch, keys, meta):
  prediction = model([test_input, seed], training=True)
  plt.figure(figsize=(16, 8))
  # display_list = [test_input[0,:,:,0]]
  # for i in range(len(keys)):
  #   display_list+=[prediction[0,:,:,i]]
  # title = ['floormap'] + keys
  scaled_pred = rescale_maps(prediction,meta,keys)
  for i in range(len(keys)):
      essentials = scaled_pred[0,:,:,i] if i == 0 else tf.maximum(essentials,scaled_pred[0,:,:,i])
  title = ['floormap','essentials']
  display_list = [test_input[0,:,:,0],essentials]
  n_items = meta['essentials']['max']

  for i in range(len(title)):
    plt.subplot(1, 2, i+1)
    plt.title(title[i] if title[i] != 'essentials' else 'thingsmap')
    if title[i] == 'essentials':
      plt.imshow((display_list[i]*128/n_items)+(display_list[i]>0).astype(tf.float32)*127, cmap='gray') 
    else:
      plt.imshow(display_list[i], cmap='gray')
    plt.axis('off')
  plt.savefig('generated_maps/hybrid/pix2pix/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close()

  
def train_step(input_image, target):
  batch_size = tf.shape(input_image)[0]
  noise_vectors = tf.random.normal(shape=(batch_size, tf.shape(input_image)[1], tf.shape(input_image)[2], 1))
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    # gen_output = generator(input_image, training=True)
    gen_output = generator([input_image, noise_vectors], training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_mask_loss, gen_obj_loss = generator_loss(disc_generated_output, gen_output, target, input_image)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  
  return {"d_loss": tf.abs(disc_loss), "g_loss": tf.abs(gen_total_loss), "gan_loss": tf.abs(gen_gan_loss), "mask_loss": tf.abs(gen_mask_loss), "obj_loss": tf.abs(gen_obj_loss)}


def train(dataset, map_meta, sample, inp_param, opt_param, epochs):
  start = time.time()
  disc_ts_loss = list()
  gen_ts_loss = list()
  sample_input = np.stack([sample[m] for m in inp_param], axis=-1).reshape((1, 256, 256, len(inp_param)))
  scaled_sample_input = scaling_maps(sample_input, map_meta, inp_param)
  seed = tf.random.normal([1, tf.shape(sample_input)[1], tf.shape(sample_input)[2], 1])
  # step_callback.on_train_begin()
  for epoch in range(epochs):
    n = 0
    start = time.time()
    for image_batch in dataset:
      input = np.stack([image_batch[m] for m in inp_param], axis=-1)
      target = np.stack([image_batch[m] for m in opt_param], axis=-1)
      x_input = scaling_maps(input, map_meta, inp_param)
      x_target = scaling_maps(target, map_meta, opt_param)
      n+=1
      # step_callback.on_train_batch_begin(batch=n)
      step_loss = train_step(x_input, x_target)
      disc_ts_loss.append(step_loss['d_loss'])
      gen_ts_loss.append(step_loss['g_loss'])
      print ('Time for batch {} is {} sec. gan_loss: {} mask_loss: {} obj_loss: {}'.format(n, time.time()-start,step_loss['gan_loss'],step_loss['mask_loss'],step_loss['obj_loss']))
    generate_images(generator, scaled_sample_input, seed, epoch+1, opt_param, map_meta)

    # step_callback.on_epoch_end(batch=epoch+1)

    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      generate_loss_graph(disc_ts_loss, gen_ts_loss, 'generated_maps/hybrid/pix2pix/')

    print ('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))

if __name__ == "__main__":
  batch_size = 1
  training_set, map_meta, sample = read_record(batch_size, sample_wgan=True)
  generator = Generator()
  # For pruning the Generator
  # generator = tfmot.sparsity.keras.prune_low_magnitude(Generator(),pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.99, 0,-1, 100))
  # step_callback = tfmot.sparsity.keras.UpdatePruningStep()
  # step_callback.set_model(generator)
  discriminator = Discriminator()
  generator_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5, beta_2=0.999)
  discriminator_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5, beta_2=0.999)

  checkpoint_dir = './training_checkpoints/hybrid/pix2pix'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

  # if os.path.exists(checkpoint_dir):                            
  #     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  input_params = ['floormap', 'wallmap', 'heightmap']
  output_params = ['monsters','ammunitions','powerups','artifacts','weapons']
  train(training_set, map_meta, sample, input_params, output_params, epochs=101)