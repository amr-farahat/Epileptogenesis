
import tensorflow as tf
import time
if tf.__version__ != '2.0.0':
    tf.enable_eager_execution()

import pdb
import numpy as np
from utils import predict_validation_samples, plot_samples

random_seed = 42
tf.random.set_random_seed(random_seed)
np.random.seed(random_seed)

class AAE(tf.keras.Model):
    
    def __init__(self, input_size, h_dim, z_dim, run_logdir):
        super(AAE, self).__init__()
        self.input_size = input_size
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.run_logdir = run_logdir

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.accuracy_z = tf.keras.metrics.BinaryAccuracy()
        self.accuracy_x = tf.keras.metrics.BinaryAccuracy()
        self.lr = 0.0001
        self.ae_optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        self.dc_z_optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        self.dc_x_optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        self.gen_z_optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        self.gen_x_optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)

        self.ae_loss_weight = 1.0
        self.gen_z_loss_weight = 0.01
        self.gen_x_loss_weight = 0.001
        self.dc_loss_weight = 1.0
                
        self.encoder = self.make_encoder_model()
        self.decoder = self.make_decoder_model()
        self.discriminator_z = self.make_discriminator_z_model()
        self.discriminator_x = self.make_discriminator_x_model()

    def make_encoder_model(self):
        input = tf.keras.layers.Input(shape=(self.input_size,1))

        conv1 = tf.keras.layers.Conv1D(16, 100, strides=4, padding='same', dilation_rate=1)(input)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.ELU()(conv1)

        conv2 = tf.keras.layers.Conv1D(32, 25, strides=4, padding='same', dilation_rate=1)(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.ELU()(conv2)

        conv3 = tf.keras.layers.Conv1D(64, 7, strides=4, padding='same', dilation_rate=1)(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.ELU()(conv3)

        code = tf.keras.layers.Conv1D(1,1, activation='linear')(conv3)
        code = tf.keras.layers.BatchNormalization()(code)

        model = tf.keras.Model(inputs=input, outputs=code)
        return model 
    
    def make_decoder_model(self,):
        encoded = tf.keras.Input(shape=(self.z_dim,1))

        reshaped_input = tf.keras.layers.Reshape((self.z_dim,1,1))(encoded)

        deconv1 = tf.keras.layers.Conv2DTranspose(64, (7,1), strides=(4,1),  padding='same', dilation_rate=1)(reshaped_input)
        deconv1 = tf.keras.layers.BatchNormalization()(deconv1)
        deconv1 = tf.keras.layers.ELU()(deconv1)

        deconv2 = tf.keras.layers.Conv2DTranspose(32, (7,1), strides=(4,1), padding='same', dilation_rate=1)(deconv1)
        deconv2 = tf.keras.layers.BatchNormalization()(deconv2)
        deconv2 = tf.keras.layers.ELU()(deconv2)  

        deconv3 = tf.keras.layers.Conv2DTranspose(16, (7,1), strides=(4,1),  padding='same', dilation_rate=1)(deconv2)
        deconv3 = tf.keras.layers.BatchNormalization()(deconv3)
        deconv3 = tf.keras.layers.ELU()(deconv3)   

        decoded = tf.keras.layers.Conv2DTranspose(1, 1, padding='same', dilation_rate=1)(deconv3)
        decoded = tf.keras.layers.Reshape((self.input_size,1))(decoded)

        model = tf.keras.Model(inputs=encoded, outputs=decoded)
        return model

    def make_discriminator_x_model(self):
        input = tf.keras.layers.Input(shape=(self.input_size,1))

        conv1 = tf.keras.layers.Conv1D(16, 100, strides=1, padding='same', dilation_rate=1)(input)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.ELU()(conv1)
        conv1 = tf.keras.layers.MaxPool1D(4)(conv1)

        conv2 = tf.keras.layers.Conv1D(32, 25, strides=1, padding='same', dilation_rate=1)(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.ELU()(conv2)
        conv2 = tf.keras.layers.MaxPool1D(4)(conv2)

        conv3 = tf.keras.layers.Conv1D(64, 7, strides=1, padding='same', dilation_rate=1)(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.ELU()(conv3)
        conv3 = tf.keras.layers.MaxPool1D(4)(conv3)

        conv4 = tf.keras.layers.Conv1D(64, 7, strides=1, padding='same', dilation_rate=1)(conv3)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.ELU()(conv4)
        conv4 = tf.keras.layers.MaxPool1D(4)(conv4)

        output = tf.keras.layers.Dense(1)(conv4)

        model = tf.keras.Model(inputs=input, outputs=output)
        return model

    def make_discriminator_z_model(self):
        encoded = tf.keras.Input(shape=(self.z_dim,1))
        flattened = tf.keras.layers.Flatten()(encoded)
        x = tf.keras.layers.Dense(self.h_dim)(flattened)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=prediction)
        return model


    def autoencoder_loss(self, inputs, reconstruction, loss_weight):
        return loss_weight * self.mse(inputs, reconstruction)

    def discriminator_loss(self, real_output, fake_output, loss_weight):
        loss_real = self.cross_entropy(tf.ones_like(real_output), real_output)
        loss_fake = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return loss_weight * 0.5 * (loss_fake + loss_real)

    def generator_loss(self, fake_output, loss_weight):
        return loss_weight * self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def predict(self, sample):
        z = self.encoder(sample)
        x = self.decoder(z)
        return x

    def set_lr(self, decay, epoch):
        return self.lr * (1 / (1 + decay * epoch))

    # @tf.function
    def train_step(self, batch_x):
       

        # autoencoder
        with tf.GradientTape() as ae_tape:
            encoder_output = self.encoder(batch_x, training=True)
            decoder_output = self.decoder(encoder_output, training=True)
            ae_loss = self.autoencoder_loss(batch_x, decoder_output, self.ae_loss_weight)

            dc_z_fake = self.discriminator_z(encoder_output, training=True)
            gen_z_loss = self.generator_loss(dc_z_fake, self.gen_z_loss_weight)

            # d_x_fake = self.discriminator_x(decoder_output, training=True)
            # gen_x_loss = self.generator_loss(d_x_fake, self.gen_x_loss_weight)

            final_loss = ae_loss + gen_z_loss

        ae_grads = ae_tape.gradient(final_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.ae_optimizer.apply_gradients(zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))


        # Discriminator z
        with tf.GradientTape() as dc_z_tape:
            real_distribution = tf.random.normal([tf.cast(batch_x.shape[0], dtype=tf.int32), self.z_dim], mean=0.0, stddev=1.0)
            # indices = tf.random.uniform([tf.cast(batch_x.shape[0] // 2, dtype=tf.int32)], minval=0, maxval=batch_x.shape[0], dtype=tf.dtypes.int32)
            # encoder_output = self.encoder(tf.gather(batch_x, indices), training=True)
            encoder_output = self.encoder(batch_x, training=True)
            dc_z_real = self.discriminator_z(real_distribution, training=True)
            dc_z_fake = self.discriminator_z(encoder_output, training=True)
            dc_z_loss = self.discriminator_loss(dc_z_real, dc_z_fake, self.dc_loss_weight)
            dc_z_acc = self.accuracy_z(tf.concat([tf.ones_like(dc_z_real), tf.zeros_like(dc_z_fake)], axis=0), tf.sigmoid(tf.concat([dc_z_real, dc_z_fake], axis=0)))
        dc_z_grads = dc_z_tape.gradient(dc_z_loss, self.discriminator_z.trainable_variables)
        self.dc_z_optimizer.apply_gradients(zip(dc_z_grads, self.discriminator_z.trainable_variables))

        # Discriminator x
        with tf.GradientTape() as dc_x_tape:
            encoder_output = self.encoder(batch_x, training=True)
            decoder_output = self.decoder(encoder_output, training=True)

            dc_x_real = self.discriminator_x(batch_x, training=True)
            dc_x_fake = self.discriminator_x(decoder_output, training=True)
            dc_x_loss = self.discriminator_loss(dc_x_real, dc_x_fake, self.dc_loss_weight)
            dc_x_acc = self.accuracy_x(tf.concat([tf.ones_like(dc_x_real), tf.zeros_like(dc_x_fake)], axis=0), tf.sigmoid(tf.concat([dc_x_real, dc_x_fake], axis=0)))
        dc_x_grads = dc_x_tape.gradient(dc_x_loss, self.discriminator_x.trainable_variables)
        self.dc_x_optimizer.apply_gradients(zip(dc_x_grads, self.discriminator_x.trainable_variables))


        # Generator z(Encoder)
        # with tf.GradientTape() as gen_z_tape:
        #     encoder_output = self.encoder(batch_x, training=True)
        #     dc_z_fake = self.discriminator_z(encoder_output, training=True)
        #     gen_z_loss = self.generator_loss(dc_z_fake, self.gen_z_loss_weight)
        # gen_z_grads = gen_z_tape.gradient(gen_z_loss, self.encoder.trainable_variables)
        # self.gen_z_optimizer.apply_gradients(zip(gen_z_grads, self.encoder.trainable_variables))

        # Generator x(Encoder)
        with tf.GradientTape() as gen_x_tape:
            encoder_output = self.encoder(batch_x, training=True)
            decoder_output = self.decoder(encoder_output, training=True)
            dc_x_fake = self.discriminator_x(decoder_output, training=True)
            gen_x_loss = self.generator_loss(dc_x_fake, self.gen_x_loss_weight)
        gen_x_grads = gen_x_tape.gradient(gen_x_loss, self.decoder.trainable_variables)
        self.gen_x_optimizer.apply_gradients(zip(gen_x_grads, self.decoder.trainable_variables))


        self.accuracy_z.reset_states()
        self.accuracy_x.reset_states()

        return ae_loss, dc_z_loss, dc_z_acc, dc_x_loss, dc_x_acc, gen_z_loss, gen_x_loss

    def print_status_bar(self, iteration, epoch_finished, losses):
        metrics = " - ".join(["{}:{:.3f}".format(m.name, m.result()) for m in losses])
        end = "\n" if epoch_finished else ""
        print("\r{} - ".format(iteration)+metrics, end=end)

    def print_trainable_weights_count(self):
        print('Encoder : '+str(len(self.encoder.trainable_variables)))
        print('Decoder : '+str(len(self.decoder.trainable_variables)))
        print('Discriminator Z : '+str(len(self.discriminator_z.trainable_variables)))
        print('Discriminator X : '+str(len(self.discriminator_x.trainable_variables)))




    def train(self, n_epochs, train_set, valid_set):
        metrics = {key:[] for key in ["ae_losses", "dc_z_losses", "dc_z_accs", "gen_z_losses", \
            "dc_x_losses", "dc_x_accs", "gen_x_losses",\
        "v_ae_losses", "v_dc_z_losses", "v_gen_z_losses", "v_gen_x_losses", "v_dc_z_accs", "v_dc_x_losses", "v_dc_x_accs" ]}

        for epoch in range(n_epochs):
            start = time.time()

            new_lr = self.set_lr(0.02, epoch)
            self.dc_z_optimizer.lr = new_lr
            self.dc_x_optimizer.lr = new_lr
            self.ae_optimizer.lr = new_lr
            self.gen_z_optimizer.lr = new_lr
            self.gen_x_optimizer.lr = new_lr

            # Learning rate schedule
        #     if epoch in [60, 100, 300]:
        #         base_lr = base_lr / 2
        #         max_lr = max_lr / 2
        #         step_size = step_size / 2
        #         print('learning rate changed!')

            epoch_ae_loss_avg = tf.compat.v2.metrics.Mean(name='Reconst loss')
            epoch_dc_x_loss_avg = tf.compat.v2.metrics.Mean(name='Disc loss_x')
            epoch_dc_z_acc_avg = tf.compat.v2.metrics.Mean(name='Disc acc_z')
            epoch_dc_x_acc_avg = tf.compat.v2.metrics.Mean(name='Disc acc_x')
            epoch_gen_x_loss_avg = tf.compat.v2.metrics.Mean(name='Gen loss_x')
            epoch_gen_z_loss_avg = tf.compat.v2.metrics.Mean(name='Gen loss_z')
            epoch_dc_z_loss_avg = tf.compat.v2.metrics.Mean(name='Disc loss_z')

            for batch, (batch_x) in enumerate(train_set):
                # Calculate cyclic learning rate
        #         global_step = global_step + 1
        #         cycle = np.floor(1 + global_step / (2 * step_size))
        #         x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        #         clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
        #         ae_optimizer.lr = clr
        #         dc_optimizer.lr = clr
        #         gen_optimizer.lr = clr

                ae_loss, dc_z_loss, dc_z_acc, dc_x_loss, dc_x_acc, gen_z_loss, gen_x_loss = self.train_step(batch_x)
                epoch_ae_loss_avg(ae_loss)
                epoch_dc_z_loss_avg(dc_z_loss)
                epoch_dc_z_acc_avg(dc_z_acc)
                epoch_dc_x_loss_avg(dc_x_loss)
                epoch_dc_x_acc_avg(dc_x_acc)
                epoch_gen_z_loss_avg(gen_z_loss)
                epoch_gen_x_loss_avg(gen_x_loss)

                self.print_status_bar(batch, False,  [epoch_ae_loss_avg, epoch_dc_z_loss_avg,
                                                    epoch_dc_z_acc_avg, epoch_dc_x_loss_avg, epoch_dc_x_acc_avg, epoch_gen_z_loss_avg, epoch_gen_x_loss_avg ])

            epoch_ae_loss_avg_valid = tf.compat.v2.metrics.Mean(name='Reconst loss')
            epoch_dc_x_loss_avg_valid = tf.compat.v2.metrics.Mean(name='Disc loss_x')
            epoch_dc_z_acc_avg_valid = tf.compat.v2.metrics.Mean(name='Disc acc_z')
            epoch_dc_x_acc_avg_valid = tf.compat.v2.metrics.Mean(name='Disc accu_x')
            epoch_gen_x_loss_avg_valid = tf.compat.v2.metrics.Mean(name='Gen loss_x')
            epoch_gen_z_loss_avg_valid = tf.compat.v2.metrics.Mean(name='Gen loss_z')
            epoch_dc_z_loss_avg_valid = tf.compat.v2.metrics.Mean(name='Disc loss_z')
            
            for _, (batch_x) in enumerate(valid_set):
                encoder_output = self.encoder(batch_x)
                decoder_output = self.decoder(encoder_output)
                ae_loss = self.autoencoder_loss(batch_x, decoder_output, self.ae_loss_weight)
                
                real_distribution = tf.random.normal([batch_x.shape[0], self.z_dim], mean=0.0, stddev=1.0)
                dc_z_real = self.discriminator_z(real_distribution)
                dc_z_fake = self.discriminator_z(encoder_output)
                dc_z_loss = self.discriminator_loss(dc_z_real, dc_z_fake, self.dc_loss_weight)

                dc_z_acc = self.accuracy_z(tf.concat([tf.ones_like(dc_z_real), tf.zeros_like(dc_z_fake)], axis=0),
                            tf.concat([dc_z_real, dc_z_fake], axis=0))

                dc_x_real = self.discriminator_x(batch_x, training=True)
                dc_x_fake = self.discriminator_x(decoder_output, training=True)
                dc_x_loss = self.discriminator_loss(dc_x_real, dc_x_fake, self.dc_loss_weight)
                dc_x_acc = self.accuracy_x(tf.concat([tf.ones_like(dc_x_real), tf.zeros_like(dc_x_fake)], axis=0), tf.sigmoid(tf.concat([dc_x_real, dc_x_fake], axis=0)))
                self.accuracy_z.reset_states()
                self.accuracy_x.reset_states()
                gen_z_loss = self.generator_loss(dc_z_fake, self.gen_z_loss_weight)
                gen_x_loss = self.generator_loss(dc_x_fake, self.gen_x_loss_weight)

                epoch_ae_loss_avg_valid(ae_loss)
                epoch_dc_z_loss_avg_valid(dc_z_loss)
                epoch_dc_z_acc_avg_valid(dc_z_acc)
                epoch_dc_x_loss_avg_valid(dc_x_loss)
                epoch_dc_x_acc_avg_valid(dc_x_acc)
                epoch_gen_z_loss_avg_valid(gen_z_loss)
                epoch_gen_x_loss_avg_valid(gen_x_loss)


            epoch_time = time.time() - start

            metrics['ae_losses'].append(epoch_ae_loss_avg.result().numpy())
            metrics['dc_z_losses'].append(epoch_dc_z_loss_avg.result().numpy())
            metrics['dc_z_accs'].append(epoch_dc_z_acc_avg.result().numpy())
            metrics['gen_z_losses'].append(epoch_gen_z_loss_avg.result().numpy())
            metrics['dc_x_losses'].append(epoch_dc_x_loss_avg.result().numpy())
            metrics['dc_x_accs'].append(epoch_dc_x_acc_avg.result().numpy())
            metrics['gen_x_losses'].append(epoch_gen_x_loss_avg.result().numpy())

            metrics['v_ae_losses'].append(epoch_ae_loss_avg_valid.result().numpy())
            metrics['v_dc_z_losses'].append(epoch_dc_z_loss_avg_valid.result().numpy())
            metrics['v_dc_z_accs'].append(epoch_dc_z_acc_avg_valid.result().numpy())
            metrics['v_gen_z_losses'].append(epoch_gen_z_loss_avg_valid.result().numpy())  
            metrics['v_dc_x_losses'].append(epoch_dc_x_loss_avg_valid.result().numpy())
            metrics['v_dc_x_accs'].append(epoch_dc_x_acc_avg_valid.result().numpy())
            metrics['v_gen_x_losses'].append(epoch_gen_x_loss_avg_valid.result().numpy())         

            self.print_status_bar('Epoch :' + str(epoch+1)+' Time: '+str(round(epoch_time)), True,  [epoch_ae_loss_avg, epoch_dc_z_loss_avg,
                                                    epoch_dc_z_acc_avg, epoch_dc_x_loss_avg, epoch_dc_x_acc_avg, epoch_gen_z_loss_avg, epoch_gen_x_loss_avg ])
            self.print_status_bar('Validation set   ', True,  [epoch_ae_loss_avg_valid, epoch_dc_z_loss_avg_valid,
                                                    epoch_dc_z_acc_avg_valid, epoch_dc_x_loss_avg_valid, epoch_dc_x_acc_avg_valid, epoch_gen_z_loss_avg_valid, 
                                                    epoch_gen_x_loss_avg_valid ]) 


            if (epoch+1) % 5 == 0:
                original_data, reconstructions = predict_validation_samples(self, valid_set, no_samples=10)
                plot_samples(original_data, reconstructions, self.run_logdir, epoch+1)                                          

        return metrics

    def save(self):
        self.encoder.save(self.run_logdir+'/encoder.h5')
        self.decoder.save(self.run_logdir+'/decoder.h5')
        self.discriminator_z.save(self.run_logdir+'/discriminator_z.h5')
        self.discriminator_x.save(self.run_logdir+'/discriminator_x.h5')




