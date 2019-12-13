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


def create_conv_AE(input_shape = (2560, 1), dilation_rate=1):
    
        input = tf.keras.layers.Input(shape=input_shape)
        conv1 = tf.keras.layers.Conv1D(64, 55, activation='elu', padding='same', dilation_rate=dilation_rate)(input)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPool1D(4)(bn1)
        conv2 = tf.keras.layers.Conv1D(128, 55, activation='elu', padding='same', dilation_rate=dilation_rate)(pool1)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPool1D(4)(bn2)
        conv3 = tf.keras.layers.Conv1D(256, 55, activation='elu', padding='same', dilation_rate=dilation_rate)(pool2)
        bn3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPool1D(4)(bn3)   
        code = tf.keras.layers.Conv1D(1,1, activation='linear')(pool3)
        bn4 = tf.keras.layers.BatchNormalization()(code)

        bn4 = tf.keras.layers.Reshape((40,1,1))(bn4)

        deconv1 = tf.keras.layers.Conv2DTranspose(256, (55,1), strides=(4,1), activation='elu', padding='same')(bn4)
        # upsamp1 = tf.keras.layers.UpSampling1D(4)(deconv1)
        deconv2 = tf.keras.layers.Conv2DTranspose(128, (55,1), strides=(4,1) , activation='elu', padding='same')(deconv1)
        # upsamp2 = tf.keras.layers.UpSampling1D(4)(deconv2)   
        deconv3 = tf.keras.layers.Conv2DTranspose(64, (55,1), strides=(4,1) , activation='elu', padding='same')(deconv2)
        # upsamp3 = tf.keras.layers.UpSampling1D(4)(deconv3)
        decoded = tf.keras.layers.Conv2DTranspose(1, (55,1), strides=(1,1) , activation='linear', padding='same')(deconv3)
        decoded = tf.keras.layers.Reshape(input_shape)(decoded)
        model = tf.keras.Model(inputs=input, outputs=decoded)
    
        return model  

def branched(input_shape = (2560, 1), dilation_rate=1):
    
        input = tf.keras.layers.Input(shape=input_shape)

        conv1 = tf.keras.layers.Conv1D(16, 55, activation='elu', padding='same', dilation_rate=dilation_rate)(input)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPool1D(4)(bn1)

        conv2 = tf.keras.layers.Conv1D(16, 105, activation='elu', padding='same', dilation_rate=dilation_rate)(input)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPool1D(4)(bn2)

        conv3 = tf.keras.layers.Conv1D(16, 210, activation='elu', padding='same', dilation_rate=dilation_rate)(input)
        bn3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPool1D(4)(bn3) 

        conc1 = tf.keras.layers.concatenate([pool1, pool2, pool3])
        conc1 = tf.keras.layers.Conv1D(1,1, activation='elu')(conc1)

        conv10 = tf.keras.layers.Conv1D(16, 15, activation='elu', padding='same', dilation_rate=dilation_rate)(conc1)
        bn10 = tf.keras.layers.BatchNormalization()(conv10)
        pool10 = tf.keras.layers.MaxPool1D(4)(bn10)

        conv20 = tf.keras.layers.Conv1D(16, 31, activation='elu', padding='same', dilation_rate=dilation_rate)(conc1)
        bn20 = tf.keras.layers.BatchNormalization()(conv20)
        pool20 = tf.keras.layers.MaxPool1D(4)(bn20)

        conv30 = tf.keras.layers.Conv1D(16, 63, activation='elu', padding='same', dilation_rate=dilation_rate)(conc1)
        bn30 = tf.keras.layers.BatchNormalization()(conv30)
        pool30 = tf.keras.layers.MaxPool1D(4)(bn30) 

        conc10 = tf.keras.layers.concatenate([pool10, pool20, pool30])
        conc10 = tf.keras.layers.Conv1D(1,1, activation='elu')(conc10)

        conv100 = tf.keras.layers.Conv1D(16, 3, activation='elu', padding='same', dilation_rate=dilation_rate)(conc10)
        bn100 = tf.keras.layers.BatchNormalization()(conv100)
        pool100 = tf.keras.layers.MaxPool1D(4)(bn100)

        conv200 = tf.keras.layers.Conv1D(16, 7, activation='elu', padding='same', dilation_rate=dilation_rate)(conc10)
        bn200 = tf.keras.layers.BatchNormalization()(conv200)
        pool200 = tf.keras.layers.MaxPool1D(4)(bn200)

        conv300 = tf.keras.layers.Conv1D(16, 15, activation='elu', padding='same', dilation_rate=dilation_rate)(conc10)
        bn300 = tf.keras.layers.BatchNormalization()(conv300)
        pool300 = tf.keras.layers.MaxPool1D(4)(bn300) 

        conc100 = tf.keras.layers.concatenate([pool100, pool200, pool300])
        
        code = tf.keras.layers.Conv1D(1,1, activation='linear')(conc100)
        code = tf.keras.layers.BatchNormalization()(code)

        # deconv1 = tf.keras.layers.Conv1D(256, 55, activation='elu', padding='valid')(code)
        upsamp1 = tf.keras.layers.UpSampling1D(4)(code)
        deconv2 = tf.keras.layers.Conv1D(16, 7, activation='elu', padding='same')(upsamp1)
        upsamp2 = tf.keras.layers.UpSampling1D(4)(deconv2)   
        deconv3 = tf.keras.layers.Conv1D(16, 7, activation='elu', padding='same', dilation_rate=2)(upsamp2)
        upsamp3 = tf.keras.layers.UpSampling1D(4)(deconv3)
        decoded = tf.keras.layers.Conv1D(1, 7, activation='linear', padding='same', dilation_rate=4)(upsamp3)
        model = tf.keras.Model(inputs=input, outputs=decoded)
    
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