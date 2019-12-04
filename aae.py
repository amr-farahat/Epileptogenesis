
import tensorflow as tf
import time
if tf.__version__ != '2.0.0':
    tf.enable_eager_execution()

import pdb
import numpy as np


class AAE(tf.keras.Model):
    
    def __init__(self, input_size, h_dim, z_dim):
        super(AAE, self).__init__()
        self.input_size = input_size
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.accuracy = tf.keras.metrics.BinaryAccuracy()

        self.ae_optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
        self.dc_optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.1)
        self.gen_optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.1)

        self.ae_loss_weight = 1.
        self.gen_loss_weight = 1.
        self.dc_loss_weight = 1.
                
        self.encoder = self.make_encoder_model()
        self.decoder = self.make_decoder_model()
        self.discriminator = self.make_discriminator_model()

    def make_encoder_model(self):
        inputs = tf.keras.Input(shape=(self.input_size,))
        x = tf.keras.layers.Dropout(0.2)(inputs)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.ELU()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.ELU()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        encoded = tf.keras.layers.Dense(self.z_dim)(x)
        model = tf.keras.Model(inputs=inputs, outputs=encoded)
        return model

    def make_decoder_model(self):
        encoded = tf.keras.Input(shape=(self.z_dim,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.ELU()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.ELU()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        reconstruction = tf.keras.layers.Dense(self.input_size)(x)
        model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
        return model

    def make_discriminator_model(self):
        encoded = tf.keras.Input(shape=(self.z_dim,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.ELU()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.ELU()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=prediction)
        return model

    def autoencoder_loss(self, inputs, reconstruction, loss_weight):
        return loss_weight * self.mse(inputs, reconstruction)

    def discriminator_loss(self, real_output, fake_output, loss_weight):
        loss_real = self.cross_entropy(tf.ones_like(real_output), real_output)
        loss_fake = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return loss_weight * (loss_fake + loss_real)

    def generator_loss(self, fake_output, loss_weight):
        return loss_weight * self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def predict(self, sample):
        z = self.encoder(sample)
        x = self.decoder(z)
        return x

    # @tf.function
    def train_step(self, batch_x):
        # autoencoder
        with tf.GradientTape() as ae_tape:
            encoder_output = self.encoder(batch_x, training=True)
            decoder_output = self.decoder(encoder_output, training=True)
            ae_loss = self.autoencoder_loss(batch_x, decoder_output, self.ae_loss_weight)
        ae_grads = ae_tape.gradient(ae_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.ae_optimizer.apply_gradients(zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))
        
        # Discriminator
        with tf.GradientTape() as dc_tape:
            real_distribution = tf.random.normal([tf.cast(batch_x.shape[0] // 2, dtype=tf.int32), self.z_dim], mean=0.0, stddev=1.0)
            indices = tf.random.uniform([tf.cast(batch_x.shape[0] // 2, dtype=tf.int32)], minval=0, maxval=batch_x.shape[0], dtype=tf.dtypes.int32)
            encoder_output = self.encoder(tf.gather(batch_x, indices), training=True)
            dc_real = self.discriminator(real_distribution, training=True)
            dc_fake = self.discriminator(encoder_output, training=True)
            dc_loss = self.discriminator_loss(dc_real, dc_fake, self.dc_loss_weight)
            dc_acc = self.accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                            tf.concat([dc_real, dc_fake], axis=0))
        dc_grads = dc_tape.gradient(dc_loss, self.discriminator.trainable_variables)
        self.dc_optimizer.apply_gradients(zip(dc_grads, self.discriminator.trainable_variables))

        # Generator (Encoder)
        with tf.GradientTape() as gen_tape:
            encoder_output = self.encoder(batch_x, training=True)
            dc_fake = self.discriminator(encoder_output, training=False)
            gen_loss = self.generator_loss(dc_fake, self.gen_loss_weight)
        gen_grads = gen_tape.gradient(gen_loss, self.encoder.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.encoder.trainable_variables))

        return ae_loss, dc_loss, dc_acc, gen_loss

    def print_status_bar(self, iteration, epoch_finished, losses):
        metrics = " - ".join(["{}:{:.4f}".format(m.name, m.result()) for m in losses])
        end = "\n" if epoch_finished else ""
        print("\r{} - ".format(iteration)+metrics, end=end)


    def train(self, n_epochs, train_set, valid_set):
        metrics = {key:[] for key in ["ae_losses", "dc_losses", "dc_accs", "gen_losses", \
        "v_ae_losses", "v_dc_losses", "v_dc_accs", "v_gen_losses" ]}

        for epoch in range(n_epochs):
            start = time.time()

            # Learning rate schedule
        #     if epoch in [60, 100, 300]:
        #         base_lr = base_lr / 2
        #         max_lr = max_lr / 2
        #         step_size = step_size / 2
        #         print('learning rate changed!')

            epoch_ae_loss_avg = tf.compat.v2.metrics.Mean(name='Reconstruction loss')
            epoch_dc_loss_avg = tf.compat.v2.metrics.Mean(name='Discriminator loss')
            epoch_dc_acc_avg = tf.compat.v2.metrics.Mean(name='Discriminator accuracy')
            epoch_gen_loss_avg = tf.compat.v2.metrics.Mean(name='Generator loss')

            for batch, (batch_x) in enumerate(train_set):
                # Calculate cyclic learning rate
        #         global_step = global_step + 1
        #         cycle = np.floor(1 + global_step / (2 * step_size))
        #         x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        #         clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
        #         ae_optimizer.lr = clr
        #         dc_optimizer.lr = clr
        #         gen_optimizer.lr = clr

                ae_loss, dc_loss, dc_acc, gen_loss = self.train_step(batch_x)
                epoch_ae_loss_avg(ae_loss)
                epoch_dc_loss_avg(dc_loss)
                epoch_dc_acc_avg(dc_acc)
                epoch_gen_loss_avg(gen_loss)
                self.print_status_bar(batch, False,  [epoch_ae_loss_avg, epoch_dc_loss_avg,
                                                    epoch_dc_acc_avg, epoch_gen_loss_avg])

            epoch_ae_loss_avg_valid = tf.compat.v2.metrics.Mean(name='Reconstruction loss')
            epoch_dc_loss_avg_valid = tf.compat.v2.metrics.Mean(name='Discriminator loss')
            epoch_dc_acc_avg_valid = tf.compat.v2.metrics.Mean(name='Discriminator accuracy')
            epoch_gen_loss_avg_valid = tf.compat.v2.metrics.Mean(name='Generator loss')
            
            for _, (batch_x) in enumerate(valid_set):
                encoder_output = self.encoder(batch_x)
                decoder_output = self.decoder(encoder_output)
                ae_loss = self.autoencoder_loss(batch_x, decoder_output, self.ae_loss_weight)
                
                real_distribution = tf.random.normal([batch_x.shape[0], self.z_dim], mean=0.0, stddev=1.0)
                dc_real = self.discriminator(real_distribution)
                dc_fake = self.discriminator(encoder_output)
                dc_loss = self.discriminator_loss(dc_real, dc_fake, self.dc_loss_weight)
                
                gen_loss = self.generator_loss(dc_fake, self.gen_loss_weight)
                
                epoch_ae_loss_avg_valid(ae_loss)
                epoch_dc_loss_avg_valid(dc_loss)
                epoch_dc_acc_avg_valid(dc_acc)
                epoch_gen_loss_avg_valid(gen_loss)

            epoch_time = time.time() - start

            metrics['ae_losses'].append(epoch_ae_loss_avg.result().numpy())
            metrics['dc_losses'].append(epoch_dc_loss_avg.result().numpy())
            metrics['dc_accs'].append(epoch_dc_acc_avg.result().numpy())
            metrics['gen_losses'].append(epoch_gen_loss_avg.result().numpy())

            metrics['v_ae_losses'].append(epoch_ae_loss_avg_valid.result().numpy())
            metrics['v_dc_losses'].append(epoch_dc_loss_avg_valid.result().numpy())
            metrics['v_dc_accs'].append(epoch_dc_acc_avg_valid.result().numpy())
            metrics['v_gen_losses'].append(epoch_gen_loss_avg_valid.result().numpy())           

            self.print_status_bar('Epoch :' + str(epoch+1)+' Time: '+str(round(epoch_time)), True,  [epoch_ae_loss_avg, epoch_dc_loss_avg,
                                                        epoch_dc_acc_avg, epoch_gen_loss_avg])
            self.print_status_bar('Validation set   ', True,  [epoch_ae_loss_avg_valid, epoch_dc_loss_avg_valid,
                                                        epoch_dc_acc_avg_valid, epoch_gen_loss_avg_valid])                                           

        return metrics

    def save(self, run_logdir):
        self.encoder.save(run_logdir+'/encoder.h5')
        self.decoder.save(run_logdir+'/decoder.h5')
        self.discriminator.save(run_logdir+'/discriminator.h5')




