
from input_pipeline import csv_reader_dataset, get_data_files
from utils import get_run_logdir, plot_dict_loss
import numpy as np
from aae import AAE
from os import listdir
import pickle


data_path = '/home/farahat/Documents/data/'
root_logdir = '/home/farahat/Documents/my_logs'
input_size = 2560
h_dim = 200
z_dim = 80
n_epochs=15
batch_size = 256


animals = sorted([f for f in listdir(data_path)])[0:1]

for animal in animals:
    
    animal_path = data_path + animal

    train_files, valid_files = get_data_files(animal_path+'/BL/')
    train_set = csv_reader_dataset(train_files, batch_size=batch_size)
    valid_set = csv_reader_dataset(valid_files, batch_size=batch_size)

    run_logdir = get_run_logdir(root_logdir, animal)

    model = AAE(input_size, h_dim, z_dim, run_logdir)
    model.print_trainable_weights_count()
    model.plot_models()
    metrics = model.train(n_epochs, train_set, valid_set)
    with open(run_logdir+'/metrics.pickle', 'wb') as handle:
        pickle.dump(metrics, handle)
    plot_dict_loss(metrics, run_logdir)
    model.save()





