import tensorflow as tf

def create_conv_AE_big(input_shape = (2560, 1), af='elu'):
        input = tf.keras.layers.Input(shape=input_shape)
        conv1 = tf.keras.layers.Conv1D(16, 55, activation='elu', padding='same')(input)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPool1D(2)(bn1)
        conv2 = tf.keras.layers.Conv1D(16, 55, activation='elu', padding='same')(pool1)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPool1D(2)(bn2)
        conv3 = tf.keras.layers.Conv1D(32, 55, activation='elu', padding='same')(pool2)
        bn3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPool1D(2)(bn3)   
        conv4 = tf.keras.layers.Conv1D(32,55, activation='elu', padding='same')(pool3)
        bn4 = tf.keras.layers.BatchNormalization()(conv4)
        pool4 = tf.keras.layers.MaxPool1D(2)(bn4)   
        conv5 = tf.keras.layers.Conv1D(64, 55, activation='elu', padding='same')(pool4)
        bn5 = tf.keras.layers.BatchNormalization()(conv5)
        pool5 = tf.keras.layers.MaxPool1D(2)(bn5)
        conv6 = tf.keras.layers.Conv1D(64, 55, activation='elu', padding='same')(pool5)
        bn6 = tf.keras.layers.BatchNormalization()(conv6)
        pool6 = tf.keras.layers.MaxPool1D(2)(bn6)
        conv7 = tf.keras.layers.Conv1D(1, 1, activation='linear')(pool6)
        bn7 = tf.keras.layers.BatchNormalization()(conv7)

        

        upsamp1 = tf.keras.layers.UpSampling1D(2)(bn7)
        deconv1 = tf.keras.layers.Conv1D(64, 55, activation='elu', padding='same')(upsamp1)
        upsamp2 = tf.keras.layers.UpSampling1D(2)(deconv1)   
        deconv2 = tf.keras.layers.Conv1D(32, 55, activation='elu', padding='same')(upsamp2)
        upsamp3 = tf.keras.layers.UpSampling1D(2)(deconv2)
        deconv3 = tf.keras.layers.Conv1D(32, 55, activation='elu', padding='same')(upsamp3)
        upsamp4 = tf.keras.layers.UpSampling1D(2)(deconv3)
        deconv4 = tf.keras.layers.Conv1D(16, 55, activation='elu', padding='same')(upsamp4)
        upsamp5 = tf.keras.layers.UpSampling1D(2)(deconv4)   
        deconv5 = tf.keras.layers.Conv1D(16, 55, activation='elu', padding='same')(upsamp5)
        upsamp6 = tf.keras.layers.UpSampling1D(2)(deconv5)
        deconv6 = tf.keras.layers.Conv1D(1, 55, activation='linear')(upsamp6)

        model = tf.keras.Model(inputs=input, outputs=deconv6)



        return model 


def create_conv_AE(input_shape = (2560, 1), af='elu'):
    
    input = keras.layers.Input(shape=input_shape)
    
    conv1 = keras.layers.Conv1D(16, 6, activation=af, padding='same')(input)
    pool1 = keras.layers.MaxPool1D(4)(conv1)
    
    conv2 = keras.layers.Conv1D(32, 6, activation=af, padding='same')(pool1)
    pool2 = keras.layers.MaxPool1D(4)(conv2)

    conv3 = keras.layers.Conv1D(64, 6, activation=af, padding='same')(pool2)
    pool3 = keras.layers.MaxPool1D(4)(conv2)
    
    conv4 = keras.layers.Conv1D(128, 6, activation=af, padding='same')(pool3)
    pool4 = keras.layers.MaxPool1D(4)(conv4)
        
    code = keras.layers.Conv1D(1,1, activation=af)(pool4)
    
    upsamp1 = keras.layers.UpSampling1D(4)(code)
    deconv1 = keras.layers.Conv1D(64, 6, activation=af, padding='same')(upsamp1)
    
    upsamp2 = keras.layers.UpSampling1D(4)(deconv1)   
    deconv2 = keras.layers.Conv1D(32, 6, activation=af, padding='same')(upsamp2)

    upsamp3 = keras.layers.UpSampling1D(4)(deconv2)
    deconv3 = keras.layers.Conv1D(1, 6, activation='linear', padding='same')(upsamp3)
    
    model = keras.models.Model(inputs=[input], outputs=[deconv3])
    
    return model  


def create_dense_AE(input_shape = (2560, ), af='elu'):
    
    input = keras.layers.Input(shape=input_shape)
    
    dense1 = keras.layers.Dense(1024, activation=af)(input)
    dense2 = keras.layers.Dense(512, activation=af)(dense1)
    dense3 = keras.layers.Dense(128, activation=af)(dense2)
    code = keras.layers.Dense(64, activation=af)(dense3)
    dense4 = keras.layers.Dense(128, activation=af)(code)
    dense5 = keras.layers.Dense(512, activation=af)(dense4)
    dense6 = keras.layers.Dense(1024, activation=af)(dense5)
    dense7 = keras.layers.Dense(2560, activation='linear')(dense6)  
    model = keras.models.Model(inputs=[input], outputs=[dense7])
    
    return model  