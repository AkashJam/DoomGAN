import tensorflow as tf
import tensorflow_addons as tfa
import time, math, os, json, sys
from gan.NetworkArchitecture import seed, topological_maps, WGAN_gen, WGAN_disc
from gan.DataProcessing import normalize_maps, generate_images
from gan.ModelEvaluation import generate_loss_graph
from gan.NNMeta import read_record, downsample, upsample


class WGAN_GP:
    def __init__(self, use_hybrid, batch_size=16, z_dim=100, gp_weight = 10, discriminator_extra_steps = 3):
        super().__init__()
        self.batch_size= batch_size
        self.hybrid = use_hybrid
        self.z_dim = z_dim
        self.gp_weight = gp_weight
        self.discriminator_extra_steps = discriminator_extra_steps
        self.map_keys = topological_maps if use_hybrid else topological_maps + ['essentials']
        self.generator = self.generator()
        self.discriminator = self.discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

        self.maps_dir = 'artifacts/generated_maps/hybrid/wgan/' if use_hybrid else 'artifacts/generated_maps/wgan/'
        self.checkpoint_dir = 'artifacts/training_checkpoints/hybrid/wgan' if use_hybrid else 'artifacts/training_checkpoints/wgan'
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer, 
                                             discriminator_optimizer=self.discriminator_optimizer, 
                                             generator=self.generator, discriminator=self.discriminator)
        self.gen_train_step_loss = list()
        self.disc_train_step_loss = list()
        self.gen_valid_loss = list()    
        self.disc_valid_loss = list()


    def generator(self):
        noise = tf.keras.layers.Input(shape=[self.z_dim])
        x = tf.keras.layers.Dense(8*8*128, use_bias=False)(noise)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Reshape((8, 8, 128))(x)
        for layer in WGAN_gen:
            x = upsample(layer['n_filters'],layer['kernel_size'],layer['stride'],layer['dropout'])(x)
        x = tf.keras.layers.Conv2DTranspose(len(self.map_keys), (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid')(x)
        return tf.keras.Model(inputs=noise, outputs=x)
    

    def discriminator(self):
        input_img = tf.keras.layers.Input(shape=[256, 256, len(self.map_keys)])
        x = input_img
        for layer in WGAN_disc:
            x = downsample(layer['n_filters'],layer['kernel_size'],layer['stride'],layer['norm'])(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs=input_img, outputs=x)


    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss


    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)


    def save_metrics(self, save_path="artifacts/eval_metrics/"):
        file_path = save_path + 'training_metrics.json'
        metrics = dict()
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        elif os.path.isfile(file_path):
            with open(file_path, 'r') as jsonfile:
                metrics = json.load(jsonfile)
        WGAN_type = "Hybrid WGAN" if self.hybrid else "Traditional WGAN"
        if WGAN_type not in list(metrics.keys()): metrics[WGAN_type] = dict()
        metrics[WGAN_type]['g_train_loss'] = [float(tl) for tl in self.gen_train_step_loss]
        metrics[WGAN_type]['g_valid_loss'] = [float(vl) for vl in self.gen_valid_loss]
        metrics[WGAN_type]['d_train_loss'] = [float(tl) for tl in self.disc_train_step_loss]
        metrics[WGAN_type]['d_valid_loss'] = [float(vl) for vl in self.disc_valid_loss]
        with open(file_path, 'w') as jsonfile:
            json.dump(metrics, jsonfile)


    def load_metrics(self, save_path="artifacts/eval_metrics/"):
        file_path = save_path + 'training_metrics.json'
        if not os.path.isfile(file_path):
            print('Missing training metrics file')
            sys.exit()
        else:
            with open(file_path, 'r') as jsonfile:
                metrics = json.load(jsonfile)
            WGAN_type = "Hybrid WGAN" if self.hybrid else "Traditional WGAN"
            if WGAN_type not in list(metrics.keys()):
                print('No training metrics found in file')
                sys.exit()
            else:   
                self.gen_train_step_loss = metrics[WGAN_type]['g_train_loss'] 
                self.gen_valid_loss = metrics[WGAN_type]['g_valid_loss'] 
                self.disc_train_step_loss = metrics[WGAN_type]['d_train_loss'] 
                self.disc_valid_loss = metrics[WGAN_type]['d_valid_loss']

    
    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir+'/'):
            print('Training checkpoint directory does not exist,')
            sys.exit()
        else:
            # self.load_metrics()
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial()


    def gradient_penalty(self, real_images, fake_images):
        # Code from https://keras.io/examples/generative/wgan_gp/#create-the-wgangp-model
        # Calculates the gradient penalty on an interpolated image and is added to the discriminator loss.
        alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        # Calculate the norm of the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    # Return the generator and discriminator losses as a loss dictionary
    def train_step(self, real_images):
        # Code from https://keras.io/examples/generative/wgan_gp/#create-the-wgangp-model
        for i in range(self.discriminator_extra_steps):
            random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.z_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                d_cost = self.discriminator_loss(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.discriminator_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.z_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.generator_loss(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.gen_train_step_loss.append(tf.abs(g_loss))
        self.disc_train_step_loss.append(tf.abs(d_loss))


    def validation(self, validation_set, map_meta):
        gen_valid_step_loss= list()
        critic_valid_step_loss = list()
        for image_batch in validation_set:
            images = tf.stack([image_batch[m] for m in self.map_keys], axis=-1)
            scaled_images = normalize_maps(images, map_meta, self.map_keys)
            for rotation in [0, 90, 180, 270]:
                # Rotating images to account for different orientations
                real_images = tfa.image.rotate(scaled_images, math.radians(rotation))                
                for flip in [0, 1]:
                    if flip:
                        real_images = tf.image.flip_left_right(real_images)
                    random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.z_dim))
                    fake_images = self.generator(random_latent_vectors, training=False)
                    fake_logits = self.discriminator(fake_images, training=False)
                    real_logits = self.discriminator(real_images, training=False)

                    d_cost = self.discriminator_loss(real_img=real_logits, fake_img=fake_logits)
                    gp = self.gradient_penalty(real_images, fake_images)
                    d_loss = d_cost + gp * self.gp_weight
                    g_loss = self.generator_loss(fake_logits)

                    gen_valid_step_loss.append(abs(g_loss))
                    critic_valid_step_loss.append(abs(d_loss))
                    self.gen_valid_loss.append(sum(gen_valid_step_loss)/len(gen_valid_step_loss))
                    self.disc_valid_loss.append(sum(critic_valid_step_loss)/len(critic_valid_step_loss))


    def train(self, epochs):
        noise_vector = tf.random.normal([1, self.z_dim], seed=seed) # 1 is the number of examples to generate
        training_set, validation_set, map_meta, sample = read_record(batch_size = self.batch_size)
        if not os.path.exists(self.checkpoint_dir+'/'):
            os.makedirs(self.checkpoint_dir+'/')
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        for epoch in range(epochs):
            n = 0
            start = time.time()
            for image_batch in training_set:
                images = tf.stack([image_batch[m] for m in self.map_keys], axis=-1)
                scaled_images = normalize_maps(images, map_meta, self.map_keys)
                for rotation in [0, 90, 180, 270]:
                    # Rotating images to account for different orientations
                    x_batch = tfa.image.rotate(scaled_images, math.radians(rotation))                
                    for flip in [0, 1]:
                        if flip: x_batch = tf.image.flip_left_right(x_batch)
                        self.train_step(x_batch)
                        n+=1
                        print ('Time for batch {} is {} sec.'.format(n, time.time()-start))
                        if (len(self.gen_train_step_loss)+1)%100 == 0: 
                            self.validation(validation_set, map_meta)

            self.checkpoint.save(file_prefix = checkpoint_prefix)
            self.save_metrics()
            generate_loss_graph(self.gen_train_step_loss, self.gen_valid_loss, 'generator', location = self.maps_dir)
            generate_loss_graph(self.disc_train_step_loss, self.disc_valid_loss, 'critic', location = self.maps_dir)
            generate_images(self.generator, noise_vector, epoch + 1, self.map_keys)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))