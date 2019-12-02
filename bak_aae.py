
import tensorflow as tf
import time
if tf.__version__ != '2.0.0':
    tf.enable_eager_execution()

import pdb

ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()
accuracy = tf.keras.metrics.BinaryAccuracy()

ae_optimizer = tf.keras.optimizers.Adam(lr=0.001)
dc_optimizer = tf.keras.optimizers.Adam(lr=0.001)
gen_optimizer = tf.keras.optimizers.Adam(lr=0.001)

input_size = 2560
h_dim = 1000
z_dim = 2

# model subparts

def make_encoder_model():
    inputs = tf.keras.Input(shape=(input_size,))
    x = tf.keras.layers.Dense(h_dim)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    encoded = tf.keras.layers.Dense(z_dim)(x)
    model = tf.keras.Model(inputs=inputs, outputs=encoded)
    return model

def make_decoder_model():
    encoded = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(h_dim)(encoded)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    reconstruction = tf.keras.layers.Dense(input_size, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    return model

def make_discriminator_model():
    encoded = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(h_dim)(encoded)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    prediction = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model

# losses

def autoencoder_loss(inputs, reconstruction, loss_weight):
    

    return loss_weight * mse(inputs, reconstruction)

def discriminator_loss(real_output, fake_output, loss_weight):
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_weight * (loss_fake + loss_real)

def generator_loss(fake_output, loss_weight):
    return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)

# train step

@tf.function
def train_step(batch_x, model):
    # -------------------------------------------------------------------------------------------------------------
    # Autoencoder
    
    encoder, decoder, discriminator = model
    with tf.GradientTape() as ae_tape:
        encoder_output = encoder(batch_x, training=True)
        decoder_output = decoder(encoder_output, training=True)

        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)
    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))
    # -------------------------------------------------------------------------------------------------------------
    # Discriminator
    with tf.GradientTape() as dc_tape:

        real_distribution = tf.random.normal([tf.cast(batch_x.shape[0], dtype=tf.int32), z_dim], mean=0.0, stddev=1.0)
        encoder_output = encoder(batch_x, training=True)
        

        dc_real = discriminator(real_distribution, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                          tf.concat([dc_real, dc_fake], axis=0))

    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Generator (Encoder)
    with tf.GradientTape() as gen_tape:
        encoder_output = encoder(batch_x, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Generator loss
        gen_loss = generator_loss(dc_fake, gen_loss_weight)

    gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

    return ae_loss, dc_loss, dc_acc, gen_loss



def train_aae(n_epochs, train_set, valid_set):

    encoder = make_encoder_model()
    decoder = make_decoder_model()
    discriminator = make_discriminator_model()

    model = (encoder, decoder, discriminator)

    for epoch in range(n_epochs):

        start = time.time()

        # Learning rate schedule
    #     if epoch in [60, 100, 300]:
    #         base_lr = base_lr / 2
    #         max_lr = max_lr / 2
    #         step_size = step_size / 2

    #         print('learning rate changed!')

        epoch_ae_loss_avg = tf.compat.v2.metrics.Mean()
        epoch_dc_loss_avg = tf.compat.v2.metrics.Mean()
        epoch_dc_acc_avg = tf.compat.v2.metrics.Mean()
        epoch_gen_loss_avg = tf.compat.v2.metrics.Mean()


    for _, (batch_x) in enumerate(train_set):
        # -------------------------------------------------------------------------------------------------------------
        # Calculate cyclic learning rate
#         global_step = global_step + 1
#         cycle = np.floor(1 + global_step / (2 * step_size))
#         x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
#         clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
#         ae_optimizer.lr = clr
#         dc_optimizer.lr = clr
#         gen_optimizer.lr = clr

        ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x, model)

        epoch_ae_loss_avg(ae_loss)
        epoch_dc_loss_avg(dc_loss)
        epoch_dc_acc_avg(dc_acc)
        epoch_gen_loss_avg(gen_loss)
        

    epoch_ae_loss_avg_valid = tf.compat.v2.metrics.Mean()
    epoch_dc_loss_avg_valid = tf.compat.v2.metrics.Mean()
    epoch_dc_acc_avg_valid = tf.compat.v2.metrics.Mean()
    epoch_gen_loss_avg_valid = tf.compat.v2.metrics.Mean()
        
    
    for _, (batch_x) in enumerate(valid_set):
        
        encoder_output = encoder(batch_x)
        decoder_output = decoder(encoder_output)
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)
        
        real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
        dc_real = discriminator(real_distribution)
        dc_fake = discriminator(encoder_output)
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)
        
        gen_loss = generator_loss(dc_fake, gen_loss_weight)
        
        
        epoch_ae_loss_avg_valid(ae_loss)
        epoch_dc_loss_avg_valid(dc_loss)
        epoch_dc_acc_avg_valid(dc_acc)
        epoch_gen_loss_avg_valid(gen_loss)
        

    epoch_time = time.time() - start
    print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f} \n' \
          .format(epoch, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_ae_loss_avg.result(),
                  epoch_dc_loss_avg.result(),
                  epoch_dc_acc_avg.result(),
                  epoch_gen_loss_avg.result()))
    
    print('            Validation AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f} \n' \
          .format(epoch_ae_loss_avg_valid.result(),
                  epoch_dc_loss_avg_valid.result(),
                  epoch_dc_acc_avg_valid.result(),
                  epoch_gen_loss_avg_valid.result()))