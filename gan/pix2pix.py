import tensorflow as tf
import tensorflow_addons as tfa
import time, math, os
import numpy as np
from matplotlib import pyplot as plt
from ganmeta import read_record, scaling_maps, generate_loss_graph
import tensorflow_model_optimization as tfmot

def downsample(filters, size, stride, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                             kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())
  return result

def upsample(filters, size, stride, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.LeakyReLU())
  return result


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(128, (8,8), (4,4), apply_batchnorm=False),  # (batch_size, 64, 64, 64)
    downsample(256, (8,8), (4,4)),  # (batch_size, 32, 32, 128)
    downsample(512, (8,8), (4,4)),  # (batch_size, 16, 16, 256)
    downsample(1024, (8,8), (4,4)),  # (batch_size, 8, 8, 512)
    # downsample(512, (4,4), (2,2)),  # (batch_size, 4, 4, 512)
    # downsample(512, (4,4), (2,2)),  # (batch_size, 2, 2, 512)
    # downsample(512, (4,4), (2,2)),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    # upsample(512, (4,4), (2,2), apply_dropout=True),  # (batch_size, 2, 2, 1024)
    # upsample(512, (4,4), (2,2), apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(1024, (8,8), (4,4), apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, (8,8), (4,4), apply_dropout=True),  # (batch_size, 16, 16, 512)
    upsample(256, (8,8), (4,4), apply_dropout=True),  # (batch_size, 32, 32, 256)
    # upsample(64, (4,4), (2,2)),  # (batch_size, 64, 64, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(1, (8,8),
                                         strides=(4,4),
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='relu')  # (batch_size, 128, 128, 1)

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
    # x = tfmot.sparsity.keras.prune_low_magnitude(up,pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.99, 1,-1, 100))(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  # x = tfmot.sparsity.keras.prune_low_magnitude(last,pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.99, 1,-1, 100))(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
# generator = tfmot.sparsity.keras.prune_low_magnitude(Generator(),pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.99, 0,-1, 100))

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, input_imgs, LAMBDA = 100):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # UNet - mask loss + Abs Mean error (no. of objs) + Categorical distribution of objs (1/n start + 10% guns + 40% ammo + 50% monsters)



  # Take floormap scaled to 0 or 1, invert the image and use it as a mask for the generated images 
  # to pick up objects generated outside the level bounds and add that to the loss
  loss = list()
  for i in range(gen_output.shape[0]):
    id = input_params.index('floormap')
    mask = (input_imgs[i,:,:,id]<1).astype(tf.float32)
    for j in range(gen_output.shape[3]):
      # obj_map = (gen_output[i,:,:,j]>0).astype(tf.uint8)
      loss.append(tf.reduce_sum(tf.multiply(mask,gen_output[i,:,:,j]))/tf.reduce_sum(mask))
  l2_loss = sum(loss)/len(loss)

  # l2_loss = math.ceil(l2_loss*100)/100

  # Mean absolute error
  # l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  l1_loss = tf.abs(tf.reduce_mean(target - gen_output))
  l_loss = l1_loss + l2_loss
  # total_gen_loss = gan_loss + LAMBDA * (l1_loss + (LAMBDA * l2_loss))
  total_gen_loss = gan_loss/(1 - (l1_loss + l2_loss)/2)

  return total_gen_loss, gan_loss, l_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, 8)

  down1 = downsample(128, (8,8), (4,4), False)(x)  # (batch_size, 64, 64, 128)
  down2 = downsample(256, (8,8), (4,4))(down1)  # (batch_size, 16, 16, 256)
  down3 = downsample(512, (8,8), (4,4))(down2)  # (batch_size, 4, 4, 512)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 6, 6, 512)
  conv = tf.keras.layers.Conv2D(1024, (4,4), strides=(1,1),
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 3, 3, 1024)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 4, 4, 1024)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 1, 1, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def generate_images(model, test_input, tar, epoch):
  prediction = model(test_input, training=False)
  plt.figure(figsize=(20, 8))

  display_list = [test_input[0,:,:,0],tar[0,:,:,0], prediction[0,:,:,0]]
  title = ['Floor Map', 'Ground Truth - essentials', 'Predicted Image - essentials']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.savefig('generated_maps/things_map/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close()
#   plt.show()

def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss, gen_l2_loss = generator_loss(disc_generated_output, gen_output, target, input_image)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  
  return {"d_loss": tf.abs(disc_loss), "g_loss": tf.abs(gen_total_loss), "gan_loss": tf.abs(gen_gan_loss), "l1_loss": tf.abs(gen_l1_loss), "l2_loss": tf.abs(gen_l2_loss)}

# step_callback = tfmot.sparsity.keras.UpdatePruningStep()
# step_callback.set_model(generator)

def train(dataset, map_meta, sample, inp_param, opt_param, epochs, batch_size=32):
  start = time.time()
  # map_keys= list(map_meta.keys())
  disc_ts_loss = list()
  gen_ts_loss = list()
  gan_ts_loss = list()
  l_ts_loss = list()
  sample_input = np.stack([sample[m] for m in inp_param], axis=-1).reshape((1, 256, 256, len(inp_param)))
  scaled_sample_input = scaling_maps(sample_input, map_meta, inp_param)
  if len(opt_param) == 1:
    sample_target = np.stack(sample[opt_param[0]]).reshape((1, 256, 256, 1))
  else:
    sample_target = np.stack([sample[m] for m in opt_param], axis=-1).reshape((1, 256, 256, len(opt_param)))
  if output_params[0] != "essentials":
    scaled_sample_target = scaling_maps(sample_target, map_meta, opt_param)
  else:
    scaled_sample_target = sample_target
  # step_callback.on_train_begin()
  for epoch in range(epochs):
    n = 0
    start = time.time()
    for image_batch in dataset:
      input = np.stack([image_batch[m] for m in inp_param], axis=-1)
      if len(opt_param) == 1:
        target = np.stack(image_batch[opt_param[0]]).reshape((batch_size, 256, 256, 1))
      else:
        target = np.stack([image_batch[m] for m in opt_param], axis=-1)
      scaled_input = scaling_maps(input, map_meta, inp_param)
      if output_params[0] != "essentials":
        scaled_target = scaling_maps(target, map_meta, opt_param)
      else:
        scaled_target = target
      for rotation in [0, 90, 180, 270]:
        x_input = tfa.image.rotate(scaled_input, math.radians(rotation))
        x_target = tfa.image.rotate(scaled_target, math.radians(rotation))
        for flip in [0, 1]:
          if flip:
              x_input = tf.image.flip_left_right(x_input)
              x_target = tf.image.flip_left_right(x_target)
          n+=1
          # step_callback.on_train_batch_begin(batch=n)
          step_loss = train_step(x_input, x_target)
          disc_ts_loss.append(step_loss['d_loss'])
          gen_ts_loss.append(step_loss['g_loss'])  
          gan_ts_loss.append(step_loss['gan_loss'])  
          l_ts_loss.append(step_loss['l_loss'])  
          print ('Time for batch {} is {} sec'.format(n, time.time()-start))
    generate_images(generator, scaled_sample_input, scaled_sample_target, epoch+1)
    generate_loss_graph(disc_ts_loss, gen_ts_loss, gan_ts_loss, l_ts_loss,'generated_maps/things_map/convergence_graph.png')

    # step_callback.on_epoch_end(batch=epoch+1)

    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))

checkpoint_dir = './training_checkpoints/pix2pix'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# if os.path.exists(checkpoint_dir):                            
#     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

b_size = 16
input_params = ['floormap', 'wallmap','heightmap']
output_params = ['essentials']
training_set, map_meta, sample = read_record(batch_size=b_size)
train(training_set, map_meta, sample, input_params, output_params, epochs=200, batch_size=b_size)