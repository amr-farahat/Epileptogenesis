from input_pipeline import csv_reader_dataset, get_train_val_files, get_data_files_LOO, get_all_data_files, get_data_files_from_folder
from utils import get_run_logdir
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import os
import scipy
import math
import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd


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


LOO = False
data_path = "C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/PPS"    #'/home/farahat/Documents/data'
root_logdir = "C:/Users/LDY/Desktop/EPG/EPG_data/results"  #'/home/farahat/Documents/my_logs/final6'
batch_size = 1024
models = sorted([f for f in os.listdir(root_logdir)])
z_dim = 80
no_samples = 20

for model_name in models[:]:
    print('working on: ' + model_name)

    animal = "1227"  #model_name[24:]
    animal_path = os.path.join(data_path, animal, animal)

    run_logdir = os.path.join(root_logdir, model_name)
    output_directory = os.path.join(run_logdir, 'stats')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    # use os.path.join is more flexible
    valid_high = os.path.join(output_directory, 'valid_data', 'high_examples')
    valid_low = os.path.join(output_directory, 'valid_data', 'low_examples')
    epg_high = os.path.join(output_directory, 'epg_data', 'high_examples')
    epg_low = os.path.join(output_directory, 'epg_data', 'low_examples')
    valid_random = os.path.join(output_directory, 'valid_data', 'random_examples')
    epg_random = os.path.join(output_directory, 'epg_data', 'random_examples')

    example_folders = [valid_high, valid_low, epg_high, epg_low, valid_random, epg_random]
    for folder in example_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if LOO:
        epg_files = get_data_files_from_folder(os.path.join(animal_path, 'EPG'), train_valid_split=False)
        valid_files = get_data_files_from_folder(os.path.join(animal_path, 'BL'), train_valid_split=False)
        # train_files = get_all_data_files(data_path, animal, train_valid_split=False)
        train_files = get_data_files_LOO(data_path, train_valid_split=False,
                               LOO_ID=animal)
    else:
        epg_files = get_data_files_from_folder(os.path.join(animal_path, 'EPG'), train_valid_split=False)
        train_files, valid_files = get_data_files_from_folder(os.path.join(animal_path, 'BL'), train_valid_split=True)

    epg_set = csv_reader_dataset(epg_files, batch_size=batch_size, shuffle=False)
    valid_set = csv_reader_dataset(valid_files, batch_size=batch_size, shuffle=False)
    train_set = csv_reader_dataset(train_files, batch_size=batch_size, shuffle=False)

    encoder = tf.keras.models.load_model(os.path.join(run_logdir,'encoder.h5'))
    decoder = tf.keras.models.load_model(os.path.join(run_logdir,'decoder.h5'))
    
    whole_segment_t_errors = np.load(os.path.join(output_directory, 'train_data', 'whole_segment_t_errors.npy'))
    whole_segment_v_errors = np.load(os.path.join(output_directory, 'valid_data', 'whole_segment_v_errors.npy'))
    whole_segment_e_errors = np.load(os.path.join(output_directory, 'epg_data', 'whole_segment_e_errors.npy'))
    
    highest_100_epg = list(np.argsort(whole_segment_e_errors)[-no_samples:])
    lowest_100_epg = list(np.argsort(whole_segment_e_errors)[:no_samples])
    highest_100_valid = list(np.argsort(whole_segment_v_errors)[-no_samples:])
    lowest_100_valid = list(np.argsort(whole_segment_v_errors)[:no_samples])
    
    ###############################

    no_samples_90_percentile_e = len(whole_segment_e_errors)//10
    no_samples_90_percentile_v = len(whole_segment_v_errors)//10

    highest_10_percentile_epg = np.argsort(whole_segment_e_errors)[-no_samples_90_percentile_e:]
    highest_10_percentile_valid = np.argsort(whole_segment_v_errors)[-no_samples_90_percentile_v:]

    random_indices_valid = list(np.random.choice(highest_10_percentile_valid, size=(100,), replace=False))
    random_indices_epg = list(np.random.choice(highest_10_percentile_epg, size=(100,), replace=False))
    print('random indices generated!')
    random_counter = 1
    for i, batch in enumerate(valid_set):
        for j in range(batch.shape[0]):
            idx = i * batch_size + j
            if idx in random_indices_valid:
                # import pdb; pdb.set_trace()
                plotting(batch[j], valid_random, random_counter, whole_segment_v_errors[idx] )
                random_counter+=1
                random_indices_valid.remove(idx)

            if not random_indices_valid:
                break
            
    random_counter = 1
    for i, batch in enumerate(epg_set):
        for j in range(batch.shape[0]):
            idx = i * batch_size + j
            if idx in random_indices_epg:
                # import pdb; pdb.set_trace()
                plotting(batch[j], epg_random, random_counter, whole_segment_e_errors[idx] )
                random_counter+=1
                random_indices_epg.remove(idx)

            if not random_indices_epg:
                break
            
###########################################################################################

    # highest_counter = 1
    # lowest_counter = 1
    # for i, batch in enumerate(valid_set):
    #     for j in range(batch.shape[0]):
    #         idx = i * batch_size + j
    #         if idx in highest_100_valid:
    #             # import pdb; pdb.set_trace()
    #             plotting(batch[j], valid_high, highest_counter, whole_segment_v_errors[idx] )
    #             highest_counter+=1
    #             highest_100_valid.remove(idx)
    #         if idx in lowest_100_valid:
                
    #             plotting(batch[j], valid_low, lowest_counter, whole_segment_v_errors[idx])
    #             lowest_counter+=1
    #             lowest_100_valid.remove(idx)
    #         if not (highest_100_valid and lowest_100_valid):
    #             break
            
    # highest_counter = 1
    # lowest_counter = 1      
    # for i, batch in enumerate(epg_set):
    #     for j in range(batch.shape[0]):
    #         idx = i * batch_size + j
    #         if idx in highest_100_epg:
    #             plotting(batch[j], epg_high, highest_counter, whole_segment_e_errors[idx])
    #             highest_counter+=1
    #             highest_100_epg.remove(idx)
    #         if idx in lowest_100_epg:
    #             plotting(batch[j], epg_low, lowest_counter, whole_segment_e_errors[idx])
    #             lowest_counter+=1
    #             lowest_100_epg.remove(idx)
    #         if not (highest_100_epg and lowest_100_epg):
    #             break
            