import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import time, math, os
import numpy as np
from ganmeta import read_record, scaling_maps, generate_and_save_images, generate_loss_graph



# Build the network
def make_generator_model(b_size=32,z_dim=100):
    # try to convert it for 256^2, which will start from 16*16*512 till 256*256*4 = 131,072 + 65,536 + 131,072 + 262,144 + 65,536 = 655,360 params
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((z_dim,), batch_size=b_size))
    model.add(layers.Dense(4*4*2048, use_bias=False, input_shape=(z_dim,))) # try 8*8*1024 -> 512 -> 256 -> 128 -> 4 with batch size 32 else 16
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((4, 4, 2048))) #Check if you can reduce the dense layer and rechape it

    model.add(layers.Conv2DTranspose(1024, (4, 4), strides=(2, 2), padding='same', use_bias=False)) 
    assert model.output_shape == (b_size, 8, 8, 1024)  # Note: None is the batch size
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)) 
    assert model.output_shape == (b_size, 16, 16, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias= False))
    assert model.output_shape == (b_size, 32, 32, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (b_size, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(4, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) 
    assert model.output_shape == (b_size, 128, 128, 4)
    return model


def make_discriminator_model(b_size=32):
    # from 256*256*4 to 1 = 2,097,152 + 
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((128, 128, 4), batch_size=b_size))
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[128, 128, 4]))
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
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# Return the generator and discriminator losses as a loss dictionary
def train_step(real_images, latent_dim, discriminator_extra_steps = 3, gp_weight = 10.0):
    if isinstance(real_images, tuple):
        real_images = real_images[0]

    # Get the batch size
    batch_size = tf.shape(real_images)[0]

    # Train the discriminator first for 3 extra steps.
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
    return {"d_loss": tf.abs(d_loss), "g_loss": tf.abs(g_loss)}


def train(dataset, map_meta, keys_pref, epochs, z_dim = 100):
    critic_ts_loss = list()
    gen_ts_loss = list()
    seed = tf.random.normal([1, z_dim]) # 1 is the number of examples to generate
    # Start training
    for epoch in range(epochs):
        n = 0
        start = time.time() 
        for image_batch in dataset:
            for rotation in [0, 90, 180, 270]:
                images = np.stack([image_batch[m] for m in keys_pref], axis=-1)
                scaled_images = scaling_maps(images, map_meta, keys_pref)
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

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)


batch_size = 32
training_set, map_meta = read_record(batch_size)
generator = make_generator_model(batch_size)
discriminator = make_discriminator_model(batch_size)
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


EPOCHS = 500

map_keys = ['heightmap', 'wallmap', 'floormap', 'roommap']
train(training_set,map_meta,map_keys,EPOCHS)
