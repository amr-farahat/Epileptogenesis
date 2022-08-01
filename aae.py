
import tensorflow as tf
import time
# tf.enable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
from utils import predict_validation_samples, plot_samples, sample_data

random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

class AAE(tf.keras.Model):
    
    def __init__(self, input_size, h_dim, z_dim, run_logdir):
        super(AAE, self).__init__()
        self.input_size = input_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.kernel_size = 5

        self.es_delta = 0.0001
        self.es_patience = 2

        self.run_logdir = run_logdir
        self.n_critic_iterations = 2
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
        self.gen_z_optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr, beta_1=0.5)

        self.ae_loss_weight = 0.99
        self.gen_z_loss_weight = 0.01
        self.dc_loss_weight = 1.0
        
        self.kernels = [3,3,3,3]
        self.strides = [2,1,2,1]
        self.filters = [64,128,256,512]
        self.num_RUs = [2,2,2,2]
        self.activation = 'relu'

        self.encoder = self.create_res_encoder()
        self.decoder = self.create_res_decoder()
        self.discriminator_z = self.make_discriminator_z_model()
        self.discriminator_x = self.make_discriminator_x_model()

    def create_residual_unit(self, inputs, filters, i, j, strides=1, kernel_size=3, change_dims=False, transpose=False, batch_norm=True):
        if transpose:
            conv = tf.keras.layers.Conv2DTranspose
        else:
            conv = tf.keras.layers.Conv2D
        
        if batch_norm:
            bn = tf.keras.layers.BatchNormalization
        else:
            bn = tf.keras.layers.BatchNormalization
            
        main_layers = [
            conv(filters, (3,1), strides=(strides,1), padding="same", use_bias=False, dilation_rate=1, name='conv_'+str(i)+'_'+str(j)+'_1'),
            bn(),
            tf.keras.layers.Activation(self.activation, name='relu_'+str(i)+'_'+str(j)+'_1'),

            conv(filters, (kernel_size,1), strides=(1,1), padding="same", use_bias=False, dilation_rate=1, name='conv_'+str(i)+'_'+str(j)+'_2'),
            bn(),
            ]

        skip_layers = []
        if strides > 1 or change_dims:
            skip_layers = [
                conv(filters, (1,1), strides=(strides,1),
                padding="same", use_bias=False),
                bn()
                ]

        Z = inputs
        for layer in main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in skip_layers:
            skip_Z = layer(skip_Z)
        return tf.keras.layers.Activation(self.activation, name='relu_'+str(i)+'_'+str(j)+'_2')(Z + skip_Z)


    def create_res_decoder(self):
        first_filter = self.filters[0]
        inputs = tf.keras.layers.Input(shape=(self.z_dim, 1))
        x = tf.keras.layers.Reshape((self.z_dim, 1, 1))(inputs)
        x = tf.keras.layers.Conv2DTranspose(first_filter, (3,1), strides=(1,1), padding="same", dilation_rate=1, use_bias=False, name='conv_0')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(self.activation, name='relu_0')(x)
        num_RUs = self.num_RUs
        kernels = self.kernels
        filters = self.filters
        strides = self.strides
        for i in range(len(num_RUs)):
            for j in range(num_RUs[i]):
                if filters[i] != first_filter:
                    change_dims = True
                else:
                    change_dims = False
                if j==0:
                    x = self.create_residual_unit(x, filters[i], i+1, j+1, strides=strides[i], kernel_size=kernels[i],
                                                  change_dims=change_dims, transpose=True, batch_norm=False)
                else:
                    x = self.create_residual_unit(x, filters[i], i+1, j+1, strides=1, kernel_size=1, transpose=True, batch_norm=False)
                first_filter = filters[i]
        x = tf.keras.layers.Conv2D(1, (1,1), strides=(1,1), padding="same", use_bias=False)(x)
        x = tf.keras.layers.Reshape((self.input_size,1))(x)
        model = tf.keras.models.Model(inputs, x)
        return model
    
    def create_res_encoder(self):
        first_filter = self.filters[0]
        inputs = tf.keras.layers.Input(shape=(self.input_size, 1))
        x = tf.keras.layers.Reshape((self.input_size, 1, 1))(inputs)
        x = tf.keras.layers.Conv2D(first_filter, (3,1), strides=(1,1), padding="same", use_bias=False, dilation_rate=1, name='conv_0')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(self.activation, name='relu_0')(x)
        num_RUs = self.num_RUs
        kernels = self.kernels
        filters = self.filters
        strides = self.strides
        for i in range(len(num_RUs)):
            for j in range(num_RUs[i]):
                if filters[i] != first_filter:
                    change_dims = True
                else:
                    change_dims = False
                if j==0:
                    x = self.create_residual_unit(x, filters[i], i+1, j+1, strides=strides[i], kernel_size=kernels[i], change_dims=change_dims)
                else:
                    x = self.create_residual_unit(x, filters[i], i+1, j+1, strides=1, kernel_size=1)
                first_filter = filters[i]

        x = tf.keras.layers.Conv2D(1, (1,1), strides=(1,1), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Reshape((self.z_dim,))(x)
        model = tf.keras.models.Model(inputs, x)
        return model



    def make_discriminator_x_model(self):
        input = tf.keras.layers.Input(shape=(self.input_size,1))

        conv1 = tf.keras.layers.Conv1D(16, self.kernel_size, strides=2, padding='same', dilation_rate=1)(input)
        conv1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)

        conv2 = tf.keras.layers.Conv1D(32, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv1)
        conv2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)

        conv3 = tf.keras.layers.Conv1D(64, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv2)
        conv3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv3)

        conv4 = tf.keras.layers.Conv1D(128, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv3)
        conv4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv4)

        conv5 = tf.keras.layers.Conv1D(256, self.kernel_size, strides=2, padding='same', dilation_rate=1)(conv4)
        conv5 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv5)

        flat = tf.keras.layers.Flatten()(conv5)

        prediction = tf.keras.layers.Dense(1)(flat)

        model = tf.keras.Model(inputs=input, outputs=[prediction, flat])
        
        print('Discriminator X : ')
        print(model.summary(line_length=50))
        
        return model

    def make_discriminator_z_model(self):
        encoded = tf.keras.Input(shape=(self.z_dim,1))
        flattened = tf.keras.layers.Flatten()(encoded)
        x = tf.keras.layers.Dense(self.h_dim)(flattened)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=[prediction, x])

        print('Discriminator Z : ')
        print(model.summary(line_length=50))
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
            final_loss = ae_loss

        ae_grads = ae_tape.gradient(final_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.ae_optimizer.apply_gradients(zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        for _ in range(self.n_critic_iterations):
            # Discriminator z
            with tf.GradientTape(persistent=False) as dc_z_tape:

                real_distribution = tf.random.normal([tf.cast(batch_x.shape[0], dtype=tf.int32), self.z_dim, 1], mean=0.0, stddev=self.std)

                encoder_output = self.encoder(batch_x, training=True)
                dc_z_real = self.discriminator_z(real_distribution, training=True)[0]
                dc_z_fake = self.discriminator_z(encoder_output, training=True)[0]
                dc_z_loss_real, dc_z_loss_fake, dc_z_loss = self.discriminator_loss(dc_z_real, dc_z_fake, self.dc_loss_weight)
                dc_z_acc = self.accuracy_z(tf.concat([tf.ones_like(dc_z_real), tf.zeros_like(dc_z_fake)], axis=0), tf.sigmoid(tf.concat([dc_z_real, dc_z_fake], axis=0)))
            dc_z_grads = dc_z_tape.gradient(dc_z_loss, self.discriminator_z.trainable_variables)            
            self.dc_z_optimizer.apply_gradients(zip(dc_z_grads, self.discriminator_z.trainable_variables))




        # # Generator z(Encoder)
        with tf.GradientTape() as gen_z_tape:
            encoder_output = self.encoder(batch_x, training=True)
            dc_z_fake = self.discriminator_z(encoder_output, training=True)[0]

            gen_z_loss = self.generator_loss(dc_z_fake, self.gen_z_loss_weight)
        gen_z_grads = gen_z_tape.gradient(gen_z_loss, self.encoder.trainable_variables)
        self.gen_z_optimizer.apply_gradients(zip(gen_z_grads, self.encoder.trainable_variables))

        self.accuracy_z.reset_states()
        self.accuracy_x.reset_states()

        return ae_loss, dc_z_loss, dc_z_acc, gen_z_loss, dc_z_loss_real, dc_z_loss_fake

    def train(self, n_epochs, train_set, valid_set):
        metrics = {key:[] for key in ["ae_losses", "dc_z_losses", "dc_z_accs", "gen_z_losses", \
            "dc_x_losses", "dc_x_accs", "gen_x_losses", "dc_z_losses_real", "dc_z_losses_fake", "dc_x_losses_real", "dc_x_losses_fake" ]}

        wait = 0
        best = np.Inf

        for epoch in range(n_epochs):
            start = time.time()

            epoch_ae_loss_avg = tf.compat.v2.metrics.Mean(name='Reconst loss')
            epoch_dc_z_acc_avg = tf.compat.v2.metrics.Mean(name='Disc acc_z')
            epoch_gen_z_loss_avg = tf.compat.v2.metrics.Mean(name='Gen loss_z')
            epoch_dc_z_loss_avg = tf.compat.v2.metrics.Mean(name='Disc loss_z')

            for batch, (batch_x) in enumerate(train_set):

                ae_loss, dc_z_loss, dc_z_acc, gen_z_loss, dc_z_loss_real, dc_z_loss_fake = self.train_step(batch_x)

                metrics['ae_losses'].append(ae_loss)
                metrics['dc_z_losses'].append(dc_z_loss)
                metrics['dc_z_accs'].append(dc_z_acc)
                metrics['gen_z_losses'].append(gen_z_loss)

                metrics['dc_z_losses_real'].append(dc_z_loss_real)
                metrics['dc_z_losses_fake'].append(dc_z_loss_fake)


                epoch_ae_loss_avg(ae_loss)
                epoch_dc_z_loss_avg(dc_z_loss)
                epoch_dc_z_acc_avg(dc_z_acc)
                epoch_gen_z_loss_avg(gen_z_loss)


                self.print_status_bar(batch, False,  [epoch_ae_loss_avg, epoch_dc_z_loss_avg,
                                                    epoch_dc_z_acc_avg, epoch_gen_z_loss_avg ])

            epoch_time = time.time() - start
            self.print_status_bar('Epoch :' + str(epoch+1)+' Time: '+str(round(epoch_time)), True,  [epoch_ae_loss_avg, epoch_dc_z_loss_avg,
                                                    epoch_dc_z_acc_avg, epoch_gen_z_loss_avg ])

            if (epoch+1) % 1 == 0:
                original_data, reconstructions = predict_validation_samples(self, valid_set, no_samples=10)
                plot_samples(original_data, reconstructions, self.run_logdir, epoch+1)
        
            if (epoch+1) % 1 == 0:
                sample_data(self.decoder, self.z_dim, self.run_logdir, self.norm_params, self.std, epoch+1, no_samples=10)      

            # if (epoch+1) % 1 == 0:
            #     plot_latent_space(self.encoder, valid_set, self.run_logdir, epoch+1)     

            current = epoch_ae_loss_avg.result()
            if current + self.es_delta < best:
                best = current
                wait = 0
                self.save()
            else:
                wait +=1
                if wait >= self.es_patience:
                    return metrics

                                     

        return metrics
    
    def clear_model(self):
        """clear current session. Reinitialize a new model"""
        tf.keras.backend.clear_session()






