import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import time, math, os
from matplotlib import pyplot as plt
import numpy as np
from ganmeta import read_record, scaling_maps, generate_loss_graph, generate_sample


# Build the network
def Generator(b_size=32,z_dim=100):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((z_dim,), batch_size=b_size))
    model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(z_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((8, 8, 512))) 

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

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid')) 
    assert model.output_shape == (b_size, 256, 256, 3)
    return model


def Discriminator(b_size=32):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((256, 256, 3), batch_size=b_size))
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[256, 256, 3]))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

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


def generate_and_save_images(model, epoch, test_input):
    # `training` is set to False so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(20, 8))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(predictions[0, :, :, i], cmap='gray') 
        plt.axis('off')

    plt.savefig('generated_maps/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def train(dataset, map_meta, keys_pref, epochs, z_dim = 100):
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
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            generate_loss_graph(critic_ts_loss, gen_ts_loss)
            # Generating sample to view in pix2pix
            sample = generator(seed, training = False)
            generate_sample(sample, keys_pref)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)
    

if __name__ == "__main__":
    batch_size = 16
    training_set, map_meta, sample = read_record(batch_size)
    generator = Generator(batch_size)
    discriminator = Discriminator(batch_size)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

    checkpoint_dir = './training_checkpoints/wgan'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    # if os.path.exists(checkpoint_dir):                            
    #     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    map_keys = ['floormap', 'wallmap', 'heightmap']
    train(training_set,map_meta,map_keys,epochs=200)
