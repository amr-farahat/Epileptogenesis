#%%
from input_pipeline import csv_reader_dataset, compute_data_parameters, get_data_files
from utils import get_run_logdir, save_results
from models import create_conv_AE, create_dense_AE, create_conv_AE_big
import numpy as np
from aae import train_aae
import pickle
from tensorflow import keras
import tensorflow as tf
if tf.__version__ != '2.0.0':
    tf.enable_eager_execution()

#%%
animal_path = '/home/farahat/Documents/data/1227/'
root_logdir = '/home/farahat/Documents/my_logs'

# train_files, valid_files = get_data_files(path)
# with open("../data/1227/train_files.txt", "wb") as fp:
#     pickle.dump(train_files, fp)
# with open("../data/1227/valid_files.txt", "wb") as fp:
#     pickle.dump(valid_files, fp)


with open(animal_path+"train_files.txt", "rb") as fp:
    train_files = pickle.load(fp)[:20]
with open(animal_path+"valid_files.txt", "rb") as fp:
    valid_files = pickle.load(fp)[:10]

# mean, std = compute_data_parameters(train_files)
# np.savetxt("../data/1227/mean_1227.csv", mean, delimiter=",")
# np.savetxt("../data/1227/std_122v.csv", std, delimiter=",")

mean = np.genfromtxt(animal_path+"mean_1227.csv", delimiter=',')[:-1]
# mean = np.expand_dims(mean, 1)
std = np.genfromtxt(animal_path+"std_122v.csv", delimiter=',')[:-1]
# std = np.expand_dims(std, 1)



train_set = csv_reader_dataset(train_files, mean, std, batch_size=128)

valid_set = csv_reader_dataset(valid_files, mean, std, batch_size=128)


model = create_conv_AE_big()
opt = keras.optimizers.Adam(lr=0.005)
model.compile(loss="mean_squared_error", optimizer=opt)
model.summary()
model = keras.models.load_model('/home/farahat/Documents/my_logs/run_2019_11_28-14_11_04/the_model.h5')

run_logdir = get_run_logdir(root_logdir)

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(train_set, epochs=10, validation_data=(valid_set))


note = '''
continuation of previous model
'''

save_results(history, model, valid_set, note, run_logdir, no_samples=6)


