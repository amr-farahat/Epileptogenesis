from input_pipeline import csv_reader_dataset, get_data_files, get_all_data_files
from utils import get_run_logdir
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import itertools
import os
import scipy
import math
import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd

random_seed = 42
tf.random.set_random_seed(random_seed)
np.random.seed(random_seed)


def plotting(batch, folder, counter, error_value):
    batch = tf.expand_dims(batch, axis=0)
    z = encoder(batch)
    x_hat = decoder(z)
    fig = plt.figure(figsize=(20,5))
    plt.plot(batch.numpy()[0,:,0], c='red', label='original',  linewidth=2)
    plt.plot(x_hat.numpy()[0,:,0], c='black', label='reconstructed',  linewidth=2, alpha=0.5)
    plt.title(str(counter) + ' - error_value : ' + str(error_value))
    plt.savefig(folder + str(counter)+'.png')    
    plt.close()



LOO = True
data_path = '/home/farahat/Documents/data/'
root_logdir = '/home/farahat/Documents/my_logs/final3/'
batch_size = 1024
models = sorted([f for f in os.listdir(root_logdir)])
z_dim = 80

for model_name in models[:]:
    print('working on: '+model_name)

    animal = model_name[24:]
    animal_path = data_path+animal

    run_logdir = root_logdir + model_name
    output_directory = run_logdir +  '/stats/'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    valid_high = output_directory + 'valid_data/high_examples/'
    valid_low = output_directory + 'valid_data/low_examples/'
    epg_high = output_directory + 'epg_data/high_examples/'
    epg_low = output_directory + 'epg_data/low_examples/'
    example_folders = [valid_high, valid_low, epg_high, epg_low]
    for folder in example_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

    if LOO:
        epg_files = get_data_files(animal_path+'/EPG/', train_valid_split=False)
        valid_files = get_data_files(animal_path+'/BL/', train_valid_split=False)
        train_files = get_all_data_files(data_path, animal, train_valid_split=False)
    else:
        epg_files = get_data_files(animal_path+'/EPG/', train_valid_split=False)
        train_files, valid_files = get_data_files(animal_path+'/BL/')

    epg_set = csv_reader_dataset(epg_files, batch_size=batch_size, shuffle=False)
    valid_set = csv_reader_dataset(valid_files, batch_size=batch_size, shuffle=False)
    train_set = csv_reader_dataset(train_files, batch_size=batch_size, shuffle=False)

    encoder = tf.keras.models.load_model(run_logdir+'/encoder.h5')
    decoder = tf.keras.models.load_model(run_logdir+'/decoder.h5')
    
    whole_segment_t_errors = np.load(output_directory+'train_data/whole_segment_t_errors.npy')
    whole_segment_v_errors = np.load(output_directory+'valid_data/whole_segment_v_errors.npy')
    whole_segment_e_errors = np.load(output_directory+'epg_data/whole_segment_e_errors.npy')
    
    highest_100_epg = list(np.argsort(whole_segment_e_errors)[-100:])
    lowest_100_epg = list(np.argsort(whole_segment_e_errors)[:100])
    highest_100_valid = list(np.argsort(whole_segment_v_errors)[-100:])
    lowest_100_valid = list(np.argsort(whole_segment_v_errors)[:100])
    
    highest_counter = 1
    lowest_counter = 1
    for i, batch in enumerate(valid_set):
        for j in range(batch.shape[0]):
            idx = i * batch_size + j
            if idx in highest_100_valid:
                # import pdb; pdb.set_trace()
                plotting(batch[j], valid_high, highest_counter, whole_segment_v_errors[idx] )
                highest_counter+=1
                highest_100_valid.remove(idx)
            if idx in lowest_100_valid:
                
                plotting(batch[j], valid_low, lowest_counter, whole_segment_v_errors[idx])
                lowest_counter+=1
                lowest_100_valid.remove(idx)
            if not (highest_100_valid and lowest_100_valid):
                break
            
    highest_counter = 1
    lowest_counter = 1      
    for i, batch in enumerate(epg_set):
        for j in range(batch.shape[0]):
            idx = i * batch_size + j
            if idx in highest_100_epg:
                plotting(batch[j], epg_high, highest_counter, whole_segment_e_errors[idx])
                highest_counter+=1
                highest_100_epg.remove(idx)
            if idx in lowest_100_epg:
                plotting(batch[j], epg_low, lowest_counter, whole_segment_e_errors[idx])
                lowest_counter+=1
                lowest_100_epg.remove(idx)
            if not (highest_100_epg and lowest_100_epg):
                break
            