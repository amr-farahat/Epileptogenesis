#%%
from input_pipeline import csv_reader_dataset, compute_data_parameters, get_data_files
from utils import get_run_logdir, save_results, predict_validation_samples, plot_samples, plot_dict_loss
from models import branched
import numpy as np
from aae import AAE
import pickle
from importlib import reload
from tensorflow import keras
import tensorflow as tf
if tf.__version__ != '2.0.0':
    tf.enable_eager_execution()


animal_path = '/home/farahat/Documents/data/1227/'
root_logdir = '/home/farahat/Documents/my_logs'

# train_files, valid_files = get_data_files(path)
# with open("../data/1227/train_files.txt", "wb") as fp:
#     pickle.dump(train_files, fp)
# with open("../data/1227/valid_files.txt", "wb") as fp:
#     pickle.dump(valid_files, fp)


with open(animal_path+"train_files.txt", "rb") as fp:
    train_files = pickle.load(fp)[:5]
with open(animal_path+"valid_files.txt", "rb") as fp:
    valid_files = pickle.load(fp)[:5]

# mean, std = compute_data_parameters(train_files)
# np.savetxt("../data/1227/mean_1227_5files.csv", mean, delimiter=",")
# np.savetxt("../data/1227/std_1227_5files.csv", std, delimiter=",")

mean = np.genfromtxt(animal_path+"mean_1227_5files.csv", delimiter=',')[:-1]
mean = np.expand_dims(mean, 1)
std = np.genfromtxt(animal_path+"std_1227_5files.csv", delimiter=',')[:-1]
std = np.expand_dims(std, 1)

batch_size = 256

train_set = csv_reader_dataset(train_files, mean, std, batch_size=batch_size)

valid_set = csv_reader_dataset(valid_files, mean, std, batch_size=batch_size)


run_logdir = get_run_logdir(root_logdir)

# save_results(history, model, valid_set, note, run_logdir, no_samples=6)


n_epochs=20


model = branched()
opt = keras.optimizers.Adam(lr=0.01)
model.compile(loss="mean_squared_error", optimizer=opt)
model.summary()
# model = keras.models.load_model('/home/farahat/Documents/my_logs/run_2019_11_28-14_11_04/the_model.h5')

# tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(train_set, epochs=n_epochs, validation_data=(valid_set))


note = '''
continuation of previous model
'''

save_results(history, model, valid_set, note, run_logdir, no_samples=6)


