import tensorflow as tf
import time, os, json, sys
from gan.NetworkArchitecture import topological_maps, object_maps, cGAN_gen, cGAN_disc
from gan.DataProcessing import normalize_maps, generate_images
from gan.NNMeta import read_record, downsample, upsample
from gan.ModelEvaluation import generate_loss_graph
from gan.metrics import encoding_error, mat_entropy, objs_per_unit_area, oob_error


class cGAN:
    def __init__(self, use_mod, batch_size=1):
        super().__init__()
        self.batch_size= batch_size
        self.mod = use_mod
        self.use_bias = False
        self.generator = self.generator()
        self.discriminator = self.discriminator()
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5, beta_2=0.999)

        self.maps_dir = 'artifacts/generated_maps/hybrid/mod_cgan/' if use_mod else 'artifacts/generated_maps/hybrid/trad_cgan/'
        self.checkpoint_dir = 'artifacts/training_checkpoints/hybrid/mod_cgan' if use_mod else 'artifacts/training_checkpoints/hybrid/trad_cgan'
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer, 
                                              discriminator_optimizer=self.discriminator_optimizer, 
                                              generator=self.generator, discriminator=self.discriminator)
        self.gen_train_step_loss = list()
        self.disc_train_step_loss = list()
        self.gen_valid_loss = list()
        self.disc_valid_loss = list()
        self.enc_err = list()
        self.entropy = list()
        self.oob_err = list()
        self.obj_arr =list()


    def generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, len(topological_maps)])
        noise = tf.keras.layers.Input(shape=[256, 256, 1])
        x = tf.keras.layers.concatenate([inputs, noise])
        initializer = tf.random_normal_initializer(0., 0.02)
        # Downsampling through the model
        skips = []
        for layer in cGAN_gen['downstack']:
            x = downsample(layer['n_filters'], layer['kernel_size'], layer['stride'], layer['norm'], bias=self.use_bias, 
                           kernel_initializer = initializer)(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for layer, skip in zip(cGAN_gen['upstack'], skips):
            x = upsample(layer['n_filters'], layer['kernel_size'], layer['stride'], layer['dropout'], bias=self.use_bias, 
                           kernel_initializer = initializer)(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv2DTranspose(len(object_maps), (8,8), strides=(4,4), padding='same', 
                                            kernel_initializer = initializer, activation='relu')(x)
        return tf.keras.Model(inputs=[inputs,noise], outputs=x)


    def generator_loss(self, disc_generated_output, gen_output, target, input_imgs, LAMBDA=100):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        if self.mod:
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
        else :
            l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
            total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss


    def discriminator(self):
        inp = tf.keras.layers.Input(shape=[256, 256, len(topological_maps)], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, len(object_maps)], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])
        initializer = tf.random_normal_initializer(0., 0.02)
        for layer in cGAN_disc:
            x = downsample(layer['n_filters'], layer['kernel_size'], layer['stride'], layer['norm'], bias=self.use_bias, 
                           kernel_initializer = initializer)(x)
        x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)  # (batch_size, 1, 1, 1)
        return tf.keras.Model(inputs=[inp, tar], outputs=x)


    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss


    def save_metrics(self, save_path="artifacts/eval_metrics/"):
        file_path = save_path + 'training_metrics.json'
        metrics = dict()
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        elif os.path.isfile(file_path):
            with open(file_path, 'r') as jsonfile:
                metrics = json.load(jsonfile)
        cGAN_type = 'Modified cGAN' if self.mod else 'Traditional cGAN'
        if cGAN_type not in list(metrics.keys()): metrics[cGAN_type] = dict()
        metrics[cGAN_type]['entropy'] = [float(met) for met in self.entropy]
        metrics[cGAN_type]['encoding_err'] = [float(met) for met in self.enc_err]
        metrics[cGAN_type]['out_of_bounds_err'] = [float(met) for met in self.oob_err]
        metrics[cGAN_type]['objs_per_area'] = [float(met) for met in self.obj_arr]
        metrics[cGAN_type]['g_train_loss'] = [float(tl) for tl in self.gen_train_step_loss]
        metrics[cGAN_type]['g_valid_loss'] = [float(vl) for vl in self.gen_valid_loss]
        metrics[cGAN_type]['d_train_loss'] = [float(tl) for tl in self.disc_train_step_loss]
        metrics[cGAN_type]['d_valid_loss'] = [float(vl) for vl in self.disc_valid_loss]
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
            cGAN_type = 'Modified cGAN' if self.mod else 'Traditional cGAN'
            if cGAN_type not in list(metrics.keys()):
                print('No training metrics found in file')
                sys.exit()
            else:   
                self.entropy = metrics[cGAN_type]['entropy'] 
                self.enc_err = metrics[cGAN_type]['encoding_err'] 
                self.oob_err = metrics[cGAN_type]['out_of_bounds_err'] 
                self.obj_arr = metrics[cGAN_type]['objs_per_area'] 
                self.gen_train_step_loss = metrics[cGAN_type]['g_train_loss'] 
                self.gen_valid_loss = metrics[cGAN_type]['g_valid_loss'] 
                self.disc_train_step_loss = metrics[cGAN_type]['d_train_loss'] 
                self.disc_valid_loss = metrics[cGAN_type]['d_valid_loss']


    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir+'/'):
            print('Training checkpoint directory does not exist,')
            sys.exit()
        else:
            # self.load_metrics()
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial()


    def training_metrics(self, gen_output, x_input, map_meta):
        id = topological_maps.index('floormap')
        self.entropy.append(mat_entropy(gen_output, map_meta))
        self.enc_err.append(encoding_error(gen_output, map_meta))
        self.oob_err.append(oob_error(gen_output, x_input[:,:,:,id]))
        self.obj_arr.append(objs_per_unit_area(gen_output, x_input[:,:,:,id]))

    
    def train_step(self, input_image, target):
        noisy_img = tf.random.normal(shape=(self.batch_size, tf.shape(input_image)[1], tf.shape(input_image)[2], 1))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator([input_image, noisy_img], training=True)
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)
            g_loss = self.generator_loss(disc_generated_output, gen_output, target, input_image)
            d_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
        generator_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        self.gen_train_step_loss.append(g_loss)
        self.disc_train_step_loss.append(d_loss)
    

    def validation(self, validation_set, map_meta):
        gen_valid_step_loss = list()
        disc_valid_step_loss = list()
        for image_batch in validation_set:
            v_input = tf.stack([image_batch[m] for m in topological_maps], axis=-1)
            v_target = tf.stack([image_batch[m] for m in object_maps], axis=-1)
            x_input = normalize_maps(v_input, map_meta, topological_maps)
            x_target = normalize_maps(v_target, map_meta, object_maps)

            noisy_img = tf.random.normal(shape=(self.batch_size, tf.shape(x_input)[1], tf.shape(x_input)[2], 1))
            gen_output = self.generator([x_input, noisy_img], training=True)

            disc_real_output = self.discriminator([x_input, x_target], training=False)
            disc_generated_output = self.discriminator([x_input, gen_output], training=False)
            g_loss = self.generator_loss(disc_generated_output, gen_output, x_target, x_input)
            d_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
            gen_valid_step_loss.append(g_loss)
            disc_valid_step_loss.append(d_loss)
            self.training_metrics(gen_output, x_input, map_meta)

        self.gen_valid_loss.append(sum(gen_valid_step_loss)/len(gen_valid_step_loss))
        self.disc_valid_loss.append(sum(disc_valid_step_loss)/len(disc_valid_step_loss))


    def train(self, epochs):
        training_set, validation_set, map_meta, sample = read_record(self.batch_size, sample_wgan=False)
        sample_input = tf.stack([sample[m] for m in topological_maps], axis=-1).reshape((1, 256, 256, len(topological_maps)))
        scaled_sample_input = normalize_maps(sample_input, map_meta, topological_maps)
        seed = tf.random.normal([1, tf.shape(sample_input)[1], tf.shape(sample_input)[2], 1])
        if not os.path.exists(self.checkpoint_dir+'/'):
            os.makedirs(self.checkpoint_dir+'/')
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        for epoch in range(epochs):
            start = time.time()
            for n, image_batch in training_set.enumerate().as_numpy_iterator():
                input = tf.stack([image_batch[m] for m in topological_maps], axis=-1)
                target = tf.stack([image_batch[m] for m in object_maps], axis=-1)
                x_input = normalize_maps(input, map_meta, topological_maps)
                x_target = normalize_maps(target, map_meta, object_maps)
                self.train_step(x_input, x_target)
                print ('Time for batch {} is {} sec.'.format(n+1, time.time()-start))
                if len(self.gen_train_step_loss)%100 == 0: self.validation(validation_set, map_meta)

            print ('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))
            if (epoch+1)%10 == 0:
                self.checkpoint.save(file_prefix = checkpoint_prefix)
                self.save_metrics()
                generate_images(self.generator, seed, epoch+1, object_maps, is_cgan=True, is_mod=self.mod, test_input=scaled_sample_input, meta=map_meta)
                generate_loss_graph(self.disc_train_step_loss, self.disc_valid_loss, 'discriminator',  location = self.maps_dir)
                generate_loss_graph(self.gen_train_step_loss, self.gen_valid_loss, 'generator', location = self.maps_dir)