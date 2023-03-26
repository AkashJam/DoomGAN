import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import time, math, os
from matplotlib import pyplot as plt
import numpy as np
from GanMeta import read_record, scaling_maps, generate_loss_graph, generate_sample, rescale_maps


# Build the network
def Generator(no_of_maps,b_size,z_dim):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((z_dim,), batch_size=b_size))
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(z_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((8, 8, 256))) 

    model.add(layers.Conv2DTranspose(1024, (4, 4), strides=(2, 2), padding='same', use_bias=False)) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias= False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(no_of_maps, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid')) 
    return model


def Discriminator(no_of_maps,b_size):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((256, 256, no_of_maps), batch_size=b_size))
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[256, 256, no_of_maps]))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(1024, (4, 4), strides=(2, 2), padding= 'same'))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())

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
        pred = discriminator(interpolated, training=True)

    # Calculate the norm of the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# Return the generator and discriminator losses as a loss dictionary
def train_step(real_images, latent_dim, discriminator_extra_steps = 3, gp_weight = 10.0):
    if isinstance(real_images, tuple):
        real_images = real_images[0]

    batch_size = tf.shape(real_images)[0]
    # Train the discriminator first for 3 extra steps.
    for i in range(discriminator_extra_steps):
        # Generate noise vector
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, latent_dim)
        )
        with tf.GradientTape() as tape:
            fake_images = generator(random_latent_vectors, training=True)
            fake_logits = discriminator(fake_images, training=True)
            real_logits = discriminator(real_images, training=True)

            d_cost = discriminator_loss(real_img=real_logits, fake_img=fake_logits)
            gp = gradient_penalty(batch_size, real_images, fake_images)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * gp_weight

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        discriminator_optimizer.apply_gradients(
            zip(d_gradient, discriminator.trainable_variables)
        )


    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    with tf.GradientTape() as tape:
        generated_images = generator(random_latent_vectors, training=True)
        gen_img_logits = discriminator(generated_images, training=True)
        g_loss = generator_loss(gen_img_logits)

    # Get the gradients w.r.t the generator loss
    gen_gradient = tape.gradient(g_loss, generator.trainable_variables)
    # Update the weights of the generator using the generator optimizer
    generator_optimizer.apply_gradients(
        zip(gen_gradient, generator.trainable_variables)
    )
    return {"d_loss": tf.abs(d_loss), "g_loss": tf.abs(g_loss)}


def generate_and_save_images(model, epoch, test_input, keys):
    # `training` is set to False so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(8, 8))
    for i in range(len(keys)):
        plt.subplot(2, 2, i+1)
        plt.title(keys[i] if keys[i] != 'essentials' else 'thingsmap')
        if keys[i] in ['thingsmap','essentials']:
            plt.imshow((predictions[0, :, :, i]*128)+(predictions[0, :, :, i]>0).astype(tf.float32)*127, cmap='gray') 
        else:
            plt.imshow(predictions[0, :, :, i], cmap='gray') 
        plt.axis('off')

    loc = 'generated_maps/wgan/' if 'essentials' in keys else 'generated_maps/hybrid/wgan/'
    plt.savefig(loc + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def train(dataset, map_meta, keys_pref, epochs, z_dim):
    critic_ts_loss = list()
    gen_ts_loss = list()
    seed = tf.random.normal([1, z_dim]) # 1 is the number of examples to generate
    # Start training
    for epoch in range(epochs):
        n = 0
        start = time.time() 
        for image_batch in dataset:
            images = np.stack([image_batch[m] for m in keys_pref], axis=-1)
            scaled_images = scaling_maps(images, map_meta, keys_pref)
            for rotation in [0, 90, 180, 270]:
                # Rotating images to account for different orientations
                x_batch = tfa.image.rotate(scaled_images, math.radians(rotation))                
                for flip in [0, 1]:
                    if flip:
                        x_batch = tf.image.flip_left_right(x_batch)
                    n+=1
                    step_loss = train_step(x_batch,z_dim)
                    critic_ts_loss.append(step_loss['d_loss'])
                    gen_ts_loss.append(step_loss['g_loss'])
                    print ('Time for batch {} is {} sec'.format(n, time.time()-start))
        generate_and_save_images(generator, epoch + 1, seed, keys_pref)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            # Generating sample to view in pix2pix
            sample = generator(seed, training = False)
            scaled_maps = rescale_maps(sample,map_meta,keys_pref)
            gen_maps = tf.cast(scaled_maps,tf.uint8)
            location = 'generated_maps/wgan/' if 'essentials' in keys_pref else 'generated_maps/hybrid/wgan/'
            save_path = '../dataset/generated/doom/wgan/' if 'essentials' in keys_pref else '../dataset/generated/doom/hybrid/'
            generate_loss_graph(critic_ts_loss, gen_ts_loss, location)
            generate_sample(gen_maps, keys_pref, save_path)


        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    

if __name__ == "__main__":
    batch_size = 16
    noise_dim = 100
    map_keys = ['floormap', 'wallmap', 'heightmap']

    training_set, map_meta, sample = read_record(batch_size = batch_size)
    generator = Generator(len(map_keys), batch_size, noise_dim)
    discriminator = Discriminator(len(map_keys), batch_size)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

    checkpoint_dir = './training_checkpoints/wgan' if 'essentials' in map_keys else './training_checkpoints/hybrid/wgan'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    train(training_set, map_meta, map_keys, epochs = 101, z_dim = noise_dim)
