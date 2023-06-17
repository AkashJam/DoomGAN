import sys
sys.path.insert(0,'..')
import tensorflow as tf
import tensorflow_addons as tfa
import time, math, os
from gan.NetworkArchitecture import topological_maps, WGAN_gen, WGAN_disc
from gan.DataProcessing import normalize_maps, rescale_maps, generate_sample, read_record
from gan.NNMeta import generate_images, generate_loss_graph


def upsample(filters, kernel, stride, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, kernel, strides=stride, padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.LeakyReLU())
    return result


def downsample(filters, kernel, stride, use_norm = True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding='same'))
    if use_norm:
        result.add(tf.keras.layers.LayerNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def Generator(n_maps, noise_dim):
    noise = tf.keras.layers.Input(shape=[noise_dim])
    x = tf.keras.layers.Dense(8*8*128, use_bias=False)(noise)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape((8, 8, 128))(x)
    for layer in WGAN_gen:
        x = upsample(layer['n_filters'],layer['kernel_size'],layer['stride'],layer['dropout'])(x)
    x = tf.keras.layers.Conv2DTranspose(n_maps, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid')(x)
    return tf.keras.Model(inputs=noise, outputs=x)


def Discriminator():
    input_img = tf.keras.layers.Input(shape=[256, 256, map_no])
    x = input_img
    for layer in WGAN_disc:
        x = downsample(layer['n_filters'],layer['kernel_size'],layer['stride'],layer['norm'])(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=input_img, outputs=x)


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
def train_step(real_images):
    for i in range(discriminator_extra_steps):
        random_latent_vectors = tf.random.normal(shape=(batch_size, z_dim))
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


def v_loss():
    valid_d_loss = list()
    valid_g_loss = list()
    for image_batch in validation_set:
        images = tf.stack([image_batch[m] for m in map_keys], axis=-1)
        scaled_images = normalize_maps(images, map_meta, map_keys)
        for rotation in [0, 90, 180, 270]:
            # Rotating images to account for different orientations
            real_images = tfa.image.rotate(scaled_images, math.radians(rotation))                
            for flip in [0, 1]:
                if flip:
                    real_images = tf.image.flip_left_right(real_images)
                random_latent_vectors = tf.random.normal(shape=(batch_size, z_dim))
                fake_images = generator(random_latent_vectors, training=False)
                fake_logits = discriminator(fake_images, training=False)
                real_logits = discriminator(real_images, training=False)

                d_cost = discriminator_loss(real_img=real_logits, fake_img=fake_logits)
                gp = gradient_penalty(real_images, fake_images)
                d_loss = d_cost + gp * gp_weight
                g_loss = generator_loss(fake_logits)
                valid_d_loss.append(abs(d_loss))
                valid_g_loss.append(abs(g_loss))
    avg_d_loss = sum(valid_d_loss)/len(valid_d_loss)
    avg_g_loss = sum(valid_g_loss)/len(valid_g_loss)
        
    return avg_d_loss, avg_g_loss


def train(epochs):
    critic_ts_loss = list()
    gen_ts_loss = list()
    critic_vs_loss = list()
    gen_vs_loss = list()
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
                    print ('Time for batch {} is {} sec. Disc loss: {} Gen loss: {}'.format(n, time.time()-start,step_loss['d_loss'],step_loss['g_loss']))
                    if len(critic_ts_loss)%100 == 0:
                        v_closs, v_gloss = v_loss()
                        critic_vs_loss.append(v_closs)
                        gen_vs_loss.append(v_gloss)

        checkpoint.save(file_prefix = checkpoint_prefix)
        loc = 'generated_maps/wgan/' if 'essentials' in map_keys else 'generated_maps/hybrid/wgan/'
        generate_loss_graph(critic_ts_loss, critic_vs_loss, 'critic', model, location = loc)
        generate_loss_graph(gen_ts_loss, gen_vs_loss, 'generator', model, location = loc)
        generate_images(generator, seed, epoch + 1, map_keys)
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
    gen_items = True
    model = "Traditional WGAN" if gen_items else "Hybrid WGAN"
    gp_weight = 10
    discriminator_extra_steps = 3
    map_keys = topological_maps + ['essentials'] if gen_items else topological_maps
    map_no = len(map_keys)

    training_set, validation_set, map_meta, sample = read_record(batch_size = batch_size)
    generator = Generator(map_no,z_dim)
    discriminator = Discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

    checkpoint_dir = './training_checkpoints/wgan' if 'essentials' in map_keys else './training_checkpoints/hybrid/wgan'
    if not os.path.exists(checkpoint_dir+'/'):
        os.makedirs(checkpoint_dir+'/')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

    train(100)
