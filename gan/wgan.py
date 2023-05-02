import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import time, math, os
from GanMeta import *

def upsample(x, filters):
    x = layers.Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x) 
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    return x

def downsample(x, filters, use_norm = True):
    x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
    if use_norm:
        x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)
    return x


def Generator(n_maps, noise_dim):
    noise = layers.Input(noise_dim,)
    x = layers.Dense(8*8*256, use_bias=False, input_shape=(noise_dim,))(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)

    x = upsample(x, 1024)
    x = upsample(x, 512)
    x = upsample(x, 256)
    x = upsample(x, 128)

    x = layers.Conv2DTranspose(n_maps, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid')(x)
    model = tf.keras.models.Model(noise, x, name="generator")
    return model


def Discriminator():
    input_img = layers.Input((256, 256, len(map_keys)),)

    x = downsample(input_img, 128, use_norm = False)
    x = downsample(x, 256)
    x = downsample(x, 512)
    x = downsample(x, 1024)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    model = tf.keras.models.Model(input_img, x, name="discriminator")
    return model


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def gradient_penalty(real_images, fake_images):
    # Calculates the gradient penalty on an interpolated image and is added to the discriminator loss.
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
def train_step(real_images, discriminator_extra_steps = 1, gp_weight = 15.0):
    if isinstance(real_images, tuple):
        real_images = real_images[0]

    for i in range(discriminator_extra_steps):
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, z_dim)
        )
        with tf.GradientTape() as tape:
            fake_images = generator(random_latent_vectors, training=True)
            fake_logits = discriminator(fake_images, training=True)
            real_logits = discriminator(real_images, training=True)

            d_cost = discriminator_loss(real_img=real_logits, fake_img=fake_logits)
            gp = gradient_penalty(real_images, fake_images)
            d_loss = d_cost + gp * gp_weight

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        discriminator_optimizer.apply_gradients(
            zip(d_gradient, discriminator.trainable_variables)
        )


    random_latent_vectors = tf.random.normal(shape=(batch_size, z_dim))
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


def train(epochs):
    critic_ts_loss = list()
    gen_ts_loss = list()
    seed = tf.random.normal([1, z_dim]) # 1 is the number of examples to generate
    # Start training
    for epoch in range(epochs):
        n = 0
        start = time.time() 
        for image_batch in training_set:
            images = tf.stack([image_batch[m] for m in map_keys], axis=-1)
            scaled_images = normalize_maps(images, map_meta, map_keys)
            for rotation in [0, 90, 180, 270]:
                # Rotating images to account for different orientations
                x_batch = tfa.image.rotate(scaled_images, math.radians(rotation))                
                for flip in [0, 1]:
                    if flip:
                        x_batch = tf.image.flip_left_right(x_batch)
                    n+=1
                    step_loss = train_step(x_batch)
                    critic_ts_loss.append(step_loss['d_loss'])
                    gen_ts_loss.append(step_loss['g_loss'])
                    print ('Time for batch {} is {} sec'.format(n, time.time()-start))
        generate_images(generator, seed, epoch + 1, map_keys)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            # checkpoint.save(file_prefix = checkpoint_prefix)
            loc = 'generated_maps/wgan/' if 'essentials' in map_keys else 'generated_maps/hybrid/wgan/'
            generate_loss_graph([critic_ts_loss, gen_ts_loss], ['critic','gen'], location = loc)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    # Generating sample to view in pix2pix
    sample = generator(seed, training = False)
    scaled_maps = rescale_maps(sample,map_meta,map_keys)
    gen_maps = tf.cast(scaled_maps,tf.uint8)
    save_path = '../dataset/generated/doom/wgan/' if 'essentials' in map_keys else '../dataset/generated/doom/hybrid/'
    generate_sample(gen_maps, map_keys, save_path)


if __name__ == "__main__":
    batch_size = 16
    z_dim = 100
    map_keys = ['floormap', 'wallmap', 'heightmap']

    training_set, map_meta, sample = read_record(batch_size = batch_size)
    generator = Generator(len(map_keys),z_dim)
    discriminator = Discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

    checkpoint_dir = './training_checkpoints/wgan' if 'essentials' in map_keys else './training_checkpoints/hybrid/wgan'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

    train(101)
