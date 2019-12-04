
from input_pipeline import csv_reader_dataset, compute_data_parameters, get_data_files
from utils import get_run_logdir, predict_validation_samples, plot_samples, plot_dict_loss
from models import create_conv_AE, create_dense_AE, create_conv_AE_big
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
    train_files = pickle.load(fp)[:20]
with open(animal_path+"valid_files.txt", "rb") as fp:
    valid_files = pickle.load(fp)[:10]

# mean, std = compute_data_parameters(train_files)
# np.savetxt("../data/1227/mean_1227_20files.csv", mean, delimiter=",")
# np.savetxt("../data/1227/std_1227_20files.csv", std, delimiter=",")

mean = np.genfromtxt(animal_path+"mean_1227_20files.csv", delimiter=',')[:-1]
# mean = np.expand_dims(mean, 1)
std = np.genfromtxt(animal_path+"std_1227_20files.csv", delimiter=',')[:-1]
# std = np.expand_dims(std, 1)



train_set = csv_reader_dataset(train_files, mean, std, batch_size=128)

valid_set = csv_reader_dataset(valid_files, mean, std, batch_size=128)


run_logdir = get_run_logdir(root_logdir)

# save_results(history, model, valid_set, note, run_logdir, no_samples=6)

input_size = 2560
h_dim = 1000
z_dim = 32
n_epochs=1000
model = AAE(input_size, h_dim, z_dim)

# model.encoder.load_weights(root_logdir+'/run_2019_12_02-17_46_16/encoder.h5')
# model.discriminator.load_weights(root_logdir+'/run_2019_12_02-17_46_16/discriminator.h5')
# model.decoder.load_weights(root_logdir+'/run_2019_12_02-17_46_16/decoder.h5')

metrics = model.train(n_epochs, train_set, valid_set)
with open(run_logdir+'/metrics.pickle', 'wb') as handle:
    pickle.dump(metrics, handle)
plot_dict_loss(metrics, run_logdir)
model.save(run_logdir)

original_data, reconstructions = predict_validation_samples(model, valid_set, no_samples=10)
plot_samples(original_data, reconstructions, run_logdir)


