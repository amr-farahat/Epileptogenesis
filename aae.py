
import tensorflow as tf
import time
tf.enable_eager_execution()

import pdb
import numpy as np
from utils import predict_validation_samples, plot_samples, plot_latent_space, sample_data

random_seed = 42
tf.random.set_random_seed(random_seed)
np.random.seed(random_seed)

class AAE(tf.keras.Model):
    
    def __init__(self, input_size, h_dim, z_dim, run_logdir):
        super(AAE, self).__init__()
        self.input_size = input_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.kernel_size = 5

        self.run_logdir = run_logdir
        self.n_critic_iterations = 1
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.accuracy_z = tf.keras.metrics.BinaryAccuracy()
        self.accuracy_x = tf.keras.metrics.BinaryAccuracy()

        self.base_lr = 0.0002
        self.augment_samples = 5
        self.augment_weight = 0.8

        self.norm_params = np.array([0])
        self.std = 0.1

        self.ae_optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr, beta_1=0.5)
        self.dc_z_optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr, beta_1=0.5)
        self.dc_x_optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr, beta_1=0.5)
        self.gen_z_optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr, beta_1=0.5)
        self.gen_x_optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr, beta_1=0.5)

        self.ae_loss_weight = 0.5
        self.reg_loss_weight = 0.2
        self.gen_z_loss_weight = 0.2
        self.gen_x_loss_weight = 0.1
        self.dc_loss_weight = 1.0
                
        self.encoder = self.make_encoder_model()
        self.decoder = self.make_decoder_model()
        self.discriminator_z = self.make_discriminator_z_model()
        self.discriminator_x = self.make_discriminator_x_model()

    def make_encoder_model(self):
        input = tf.keras.layers.Input(shape=(self.input_size,1))

        conv1 = tf.keras.layers.Conv1D(8, self.kernel_size, strides=2, padding='same', dilation_rate=1)(input)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.ReLU()(conv1)

        conv2 = tf.keras.layers.Conv1D(16, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.ReLU()(conv2)

        conv3 = tf.keras.layers.Conv1D(32, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.ReLU()(conv3)

        conv4 = tf.keras.layers.Conv1D(64, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv3)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.ReLU()(conv4)

        conv5 = tf.keras.layers.Conv1D(128, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv4)
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv5 = tf.keras.layers.ReLU()(conv5)

        code = tf.keras.layers.Conv1D(1,1, activation='linear')(conv5)
        code = tf.keras.layers.BatchNormalization()(code)

        model = tf.keras.Model(inputs=input, outputs=code)
        return model 
    
    def make_decoder_model(self):
        encoded = tf.keras.Input(shape=(self.z_dim,1))
        reshaped_input = tf.keras.layers.Reshape((self.z_dim,1,1))(encoded)

        deconv1 = tf.keras.layers.Conv2DTranspose(128, (self.kernel_size,1), strides=(2,1),  padding='same', dilation_rate=1)(reshaped_input)
        deconv1 = tf.keras.layers.BatchNormalization()(deconv1)
        deconv1 = tf.keras.layers.ReLU()(deconv1)

        deconv2 = tf.keras.layers.Conv2DTranspose(64, (self.kernel_size,1), strides=(2,1), padding='same', dilation_rate=1)(deconv1)
        deconv2 = tf.keras.layers.BatchNormalization()(deconv2)
        deconv2 = tf.keras.layers.ReLU()(deconv2)  

        deconv3 = tf.keras.layers.Conv2DTranspose(32, (self.kernel_size,1), strides=(2,1),  padding='same', dilation_rate=1)(deconv2)
        deconv3 = tf.keras.layers.BatchNormalization()(deconv3)
        deconv3 = tf.keras.layers.ReLU()(deconv3) 

        deconv4 = tf.keras.layers.Conv2DTranspose(16, (self.kernel_size,1), strides=(2,1),  padding='same', dilation_rate=1)(deconv3)
        deconv4 = tf.keras.layers.BatchNormalization()(deconv4)
        deconv4 = tf.keras.layers.ReLU()(deconv4)      

        deconv5 = tf.keras.layers.Conv2DTranspose(8, (self.kernel_size,1), strides=(2,1),  padding='same', dilation_rate=1)(deconv4)
        deconv5 = tf.keras.layers.BatchNormalization()(deconv5)
        deconv5 = tf.keras.layers.ReLU()(deconv5)     

        decoded = tf.keras.layers.Conv2DTranspose(1, 1, padding='same', activation='linear', dilation_rate=1)(deconv5)
        decoded = tf.keras.layers.Reshape((self.input_size,1))(decoded)



        model = tf.keras.Model(inputs=encoded, outputs=decoded)
        return model

    def make_discriminator_x_model(self):
        input = tf.keras.layers.Input(shape=(self.input_size,1))

        conv1 = tf.keras.layers.Conv1D(8, self.kernel_size, strides=2, padding='same', dilation_rate=1)(input)
        # conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
        # conv1 = tf.keras.layers.Dropout(0.3)(conv1)

        conv2 = tf.keras.layers.Conv1D(16, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv1)
        # conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)
        # conv2 = tf.keras.layers.Dropout(0.3)(conv2)

        conv3 = tf.keras.layers.Conv1D(32, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv2)
        # conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv3)
        # conv3 = tf.keras.layers.Dropout(0.3)(conv3)

        conv4 = tf.keras.layers.Conv1D(64, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv3)
        # conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv4)
        # conv4 = tf.keras.layers.Dropout(0.3)(conv4)

        conv5 = tf.keras.layers.Conv1D(128, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv4)
        # conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv5 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv5)
        # conv5 = tf.keras.layers.Dropout(0.3)(conv5)

        # conv6 = tf.keras.layers.Conv1D(256, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv5)
        # # conv6 = tf.keras.layers.BatchNormalization()(conv6)
        # conv6 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv6)
        # # conv6 = tf.keras.layers.Dropout(0.3)(conv6)

        flat = tf.keras.layers.Flatten()(conv5)

        prediction = tf.keras.layers.Dense(1)(flat)

        model = tf.keras.Model(inputs=input, outputs=[prediction, flat])
        return model

    def make_discriminator_z_model(self):
        encoded = tf.keras.Input(shape=(self.z_dim,1))
        flattened = tf.keras.layers.Flatten()(encoded)
        x = tf.keras.layers.Dense(self.h_dim)(flattened)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=[prediction, x])
        return model


    def autoencoder_loss(self, inputs, reconstruction, loss_weight):
        return loss_weight * self.mse(inputs, reconstruction)

    def discriminator_loss(self, real_output, fake_output, loss_weight):
        loss_real = self.cross_entropy(tf.ones_like(real_output), real_output)
        loss_fake = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return loss_real, loss_fake, loss_weight * 0.5 * (loss_real + loss_fake)

    def generator_loss(self, fake_output, loss_weight):
        return loss_weight * self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def predict(self, sample):
        z = self.encoder(sample)
        x = self.decoder(z)
        return x

    def set_lr(self, decay, epoch):
        return self.base_lr * (1 / (1 + decay * epoch))

    def print_status_bar(self, iteration, epoch_finished, losses):
        metrics = " - ".join(["{}:{:.5f}".format(m.name, m.result()) for m in losses])
        end = "\n" if epoch_finished else ""
        print("\r{} - ".format(iteration)+metrics, end=end)

    def print_trainable_weights_count(self):
        print('Encoder : ')
        print(self.encoder.summary())
        print('Decoder : ')
        print(self.decoder.summary())
        print('Discriminator Z : ')
        print(self.discriminator_z.summary())
        print('Discriminator X : ')
        print(self.discriminator_x.summary())

    def plot_models(self):
        tf.keras.utils.plot_model(self.encoder, to_file=self.run_logdir+'/encoder.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.decoder, to_file=self.run_logdir+'/decoder.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.discriminator_z, to_file=self.run_logdir+'/discriminator_z.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.discriminator_x, to_file=self.run_logdir+'/discriminator_x.png', show_shapes=True, show_layer_names=True)

    def save(self):
        self.encoder.save(self.run_logdir+'/encoder.h5')
        self.decoder.save(self.run_logdir+'/decoder.h5')
        self.discriminator_z.save(self.run_logdir+'/discriminator_z.h5')
        self.discriminator_x.save(self.run_logdir+'/discriminator_x.h5')

    def train_step(self, batch_x):

        # autoencoder
        with tf.GradientTape(persistent=False) as ae_tape:
            encoder_output = self.encoder(batch_x, training=True)

            decoder_output = self.decoder(encoder_output, training=True)
            ae_loss = self.autoencoder_loss(batch_x, decoder_output, self.ae_loss_weight)

            dc_z_fake = self.discriminator_z(encoder_output, training=True)[0]
            gen_z_loss = self.generator_loss(dc_z_fake, self.gen_z_loss_weight)

            weights = np.ones(len(self.norm_params), dtype=np.float64) / len(self.norm_params)
            mixture_idx = np.random.choice(len(weights), size=batch_x.shape[0], replace=True, p=weights)
            real_distribution = tf.convert_to_tensor([np.random.normal(self.norm_params[idx], self.std, size=(self.z_dim,1)) for idx in mixture_idx], dtype=tf.float32)

            # real_distribution = tf.random.normal([tf.cast(batch_x.shape[0], dtype=tf.int32), self.z_dim, 1], mean=0.0, stddev=1.0)
            generator_output = self.decoder(real_distribution, training=True)
            dc_x_fake = self.discriminator_x(generator_output, training=True)[0]
            gen_x_loss = self.generator_loss(dc_x_fake, self.gen_x_loss_weight)                       

            new_batch_x = []
            for i in range(batch_x.shape[0]): 
                idx = tf.random.uniform([self.augment_samples], minval=0, maxval=batch_x.shape[0], dtype=tf.int32)
                new_batch_x.append(self.augment_weight*batch_x[i] + (1-self.augment_weight)*(tf.reduce_mean(tf.gather(batch_x, idx), 0)))
            new_batch_x = tf.stack(new_batch_x)
            encoder_output_augmented = self.encoder(new_batch_x, training=True)
            reg_loss = self.autoencoder_loss(encoder_output, encoder_output_augmented, self.reg_loss_weight)

            final_loss = ae_loss + reg_loss + gen_x_loss + gen_z_loss

        ae_grads = ae_tape.gradient(final_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.ae_optimizer.apply_gradients(zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        for _ in range(self.n_critic_iterations):
            # Discriminator z
            with tf.GradientTape(persistent=False) as dc_z_tape:

                # real_distribution = tf.random.normal([tf.cast(batch_x.shape[0], dtype=tf.int32), self.z_dim, 1], mean=0.0, stddev=1.0)
 
                weights = np.ones(len(self.norm_params), dtype=np.float64) / len(self.norm_params)
                mixture_idx = np.random.choice(len(weights), size=batch_x.shape[0], replace=True, p=weights)
                real_distribution = tf.convert_to_tensor([np.random.normal(self.norm_params[idx], self.std, size=(self.z_dim,1)) for idx in mixture_idx], dtype=tf.float32)

                
                encoder_output = self.encoder(batch_x, training=True)
                dc_z_real = self.discriminator_z(real_distribution, training=True)[0]
                dc_z_fake = self.discriminator_z(encoder_output, training=True)[0]
                dc_z_loss_real, dc_z_loss_fake, dc_z_loss = self.discriminator_loss(dc_z_real, dc_z_fake, self.dc_loss_weight)
                dc_z_acc = self.accuracy_z(tf.concat([tf.ones_like(dc_z_real), tf.zeros_like(dc_z_fake)], axis=0), tf.sigmoid(tf.concat([dc_z_real, dc_z_fake], axis=0)))
            dc_z_grads = dc_z_tape.gradient(dc_z_loss, self.discriminator_z.trainable_variables)            
            self.dc_z_optimizer.apply_gradients(zip(dc_z_grads, self.discriminator_z.trainable_variables))

            # Discriminator x
            with tf.GradientTape(persistent=False) as dc_x_tape:
                # encoder_output = self.encoder(batch_x, training=True)

                weights = np.ones(len(self.norm_params), dtype=np.float64) / len(self.norm_params)
                mixture_idx = np.random.choice(len(weights), size=batch_x.shape[0], replace=True, p=weights)
                real_distribution = tf.convert_to_tensor([np.random.normal(self.norm_params[idx], self.std, size=(self.z_dim,1)) for idx in mixture_idx], dtype=tf.float32)

                # real_distribution = tf.random.normal([tf.cast(batch_x.shape[0], dtype=tf.int32), self.z_dim, 1], mean=0.0, stddev=1.0)
                decoder_output = self.decoder(real_distribution, training=True)
                dc_x_real = self.discriminator_x(batch_x, training=True)[0]
                dc_x_fake = self.discriminator_x(decoder_output, training=True)[0]
                dc_x_loss_real, dc_x_loss_fake, dc_x_loss = self.discriminator_loss(dc_x_real, dc_x_fake, self.dc_loss_weight)
                dc_x_acc = self.accuracy_x(tf.concat([tf.ones_like(dc_x_real), tf.zeros_like(dc_x_fake)], axis=0), tf.sigmoid(tf.concat([dc_x_real, dc_x_fake], axis=0)))
            dc_x_grads = dc_x_tape.gradient(dc_x_loss, self.discriminator_x.trainable_variables)
            self.dc_x_optimizer.apply_gradients(zip(dc_x_grads, self.discriminator_x.trainable_variables))

        # # Generator z(Encoder)
        # with tf.GradientTape() as gen_z_tape:
        #     encoder_output = self.encoder(batch_x, training=True)
        #     dc_z_fake = self.discriminator_z(encoder_output, training=True)[0]

        #     # dc_z_fake = self.discriminator_z(encoder_output, training=True)[1]
        #     # real_distribution = tf.random.normal([tf.cast(batch_x.shape[0], dtype=tf.int32), self.z_dim, 1], mean=0.0, stddev=1.0)
        #     # dc_z_real = self.discriminator_z(real_distribution, training=True)[1]
        #     # gen_z_loss = self.autoencoder_loss(dc_z_real, dc_z_fake, self.gen_z_loss_weight)

        #     gen_z_loss = self.generator_loss(dc_z_fake, self.gen_z_loss_weight)
        # gen_z_grads = gen_z_tape.gradient(gen_z_loss, self.encoder.trainable_variables)
        # self.gen_z_optimizer.apply_gradients(zip(gen_z_grads, self.encoder.trainable_variables))

        # # Generator x(Encoder)
        # with tf.GradientTape() as gen_x_tape:
        #     # encoder_output = self.encoder(batch_x, training=True)
        #     # real_distribution = tf.random.normal([tf.cast(batch_x.shape[0], dtype=tf.int32), self.z_dim, 1], mean=0.0, stddev=1.0)
        #     weights = np.ones(len(self.norm_params), dtype=np.float64) / len(self.norm_params)
        #     mixture_idx = np.random.choice(len(weights), size=batch_x.shape[0], replace=True, p=weights)
        #     real_distribution = tf.convert_to_tensor([np.random.normal(self.norm_params[idx], self.std, size=(self.z_dim,1)) for idx in mixture_idx], dtype=tf.float32)

        #     # norm_prior = tf.random.normal([tf.cast(batch_x.shape[0], dtype=tf.int32), 40-self.z_dim, 1], mean=0.0, stddev=1.0)
        #     # decoder_input = tf.concat([encoder_output, norm_prior], 1)

        #     decoder_output = self.decoder(real_distribution, training=True)
        #     dc_x_fake = self.discriminator_x(decoder_output, training=True)[0]

        #     # dc_x_fake = self.discriminator_x(decoder_output, training=True)[1]
        #     # dc_x_real = self.discriminator_x(batch_x, training=True)[1]
        #     # gen_x_loss = self.autoencoder_loss(dc_x_real, dc_x_fake, self.gen_x_loss_weight) 

        #     gen_x_loss = self.generator_loss(dc_x_fake, self.gen_x_loss_weight)

        # gen_x_grads = gen_x_tape.gradient(gen_x_loss, self.decoder.trainable_variables)
        # self.gen_x_optimizer.apply_gradients(zip(gen_x_grads, self.decoder.trainable_variables))

        self.accuracy_z.reset_states()
        self.accuracy_x.reset_states()

        return ae_loss, dc_z_loss, dc_z_acc, dc_x_loss, dc_x_acc, gen_z_loss, gen_x_loss, dc_z_loss_real, dc_z_loss_fake, dc_x_loss_real, dc_x_loss_fake

    def train(self, n_epochs, train_set, valid_set):
        metrics = {key:[] for key in ["ae_losses", "dc_z_losses", "dc_z_accs", "gen_z_losses", \
            "dc_x_losses", "dc_x_accs", "gen_x_losses", "dc_z_losses_real", "dc_z_losses_fake", "dc_x_losses_real", "dc_x_losses_fake" ]}

        for epoch in range(n_epochs):
            start = time.time()

            epoch_ae_loss_avg = tf.compat.v2.metrics.Mean(name='Reconst loss')
            epoch_dc_x_loss_avg = tf.compat.v2.metrics.Mean(name='Disc loss_x')
            epoch_dc_z_acc_avg = tf.compat.v2.metrics.Mean(name='Disc acc_z')
            epoch_dc_x_acc_avg = tf.compat.v2.metrics.Mean(name='Disc acc_x')
            epoch_gen_x_loss_avg = tf.compat.v2.metrics.Mean(name='Gen loss_x')
            epoch_gen_z_loss_avg = tf.compat.v2.metrics.Mean(name='Gen loss_z')
            epoch_dc_z_loss_avg = tf.compat.v2.metrics.Mean(name='Disc loss_z')

            for batch, (batch_x) in enumerate(train_set):

                ae_loss, dc_z_loss, dc_z_acc, dc_x_loss, dc_x_acc, gen_z_loss, gen_x_loss, dc_z_loss_real, dc_z_loss_fake, dc_x_loss_real, dc_x_loss_fake = self.train_step(batch_x)

                metrics['ae_losses'].append(ae_loss)
                metrics['dc_z_losses'].append(dc_z_loss)
                metrics['dc_z_accs'].append(dc_z_acc)
                metrics['gen_z_losses'].append(gen_z_loss)
                metrics['dc_x_losses'].append(dc_x_loss)
                metrics['dc_x_accs'].append(dc_x_acc)
                metrics['gen_x_losses'].append(gen_x_loss)
                metrics['dc_z_losses_real'].append(dc_z_loss_real)
                metrics['dc_z_losses_fake'].append(dc_z_loss_fake)
                metrics['dc_x_losses_real'].append(dc_x_loss_real)
                metrics['dc_x_losses_fake'].append(dc_x_loss_fake)

                epoch_ae_loss_avg(ae_loss)
                epoch_dc_z_loss_avg(dc_z_loss)
                epoch_dc_z_acc_avg(dc_z_acc)
                epoch_dc_x_loss_avg(dc_x_loss)
                epoch_dc_x_acc_avg(dc_x_acc)
                epoch_gen_z_loss_avg(gen_z_loss)
                epoch_gen_x_loss_avg(gen_x_loss)

                self.print_status_bar(batch, False,  [epoch_ae_loss_avg, epoch_dc_z_loss_avg,
                                                    epoch_dc_z_acc_avg, epoch_dc_x_loss_avg, epoch_dc_x_acc_avg, epoch_gen_z_loss_avg, epoch_gen_x_loss_avg ])

            epoch_time = time.time() - start
            self.print_status_bar('Epoch :' + str(epoch+1)+' Time: '+str(round(epoch_time)), True,  [epoch_ae_loss_avg, epoch_dc_z_loss_avg,
                                                    epoch_dc_z_acc_avg, epoch_dc_x_loss_avg, epoch_dc_x_acc_avg, epoch_gen_z_loss_avg, epoch_gen_x_loss_avg ])

            if (epoch+1) % 1 == 0:
                original_data, reconstructions = predict_validation_samples(self, valid_set, no_samples=10)
                plot_samples(original_data, reconstructions, self.run_logdir, epoch+1)
        
            if (epoch+1) % 1 == 0:
                sample_data(self.decoder, self.z_dim, self.run_logdir, self.norm_params, self.std, epoch+1, no_samples=10)      

            # if (epoch+1) % 1 == 0:
            #     plot_latent_space(self.encoder, valid_set, self.run_logdir, epoch+1)                              

        return metrics






