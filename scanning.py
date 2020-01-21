from input_pipeline import csv_reader_dataset, get_data_files
from utils import get_run_logdir
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import itertools
import os
import scipy
import math
import seaborn as sns
sns.set(style="darkgrid")
import pandas as pd

random_seed = 42
tf.random.set_random_seed(random_seed)
np.random.seed(random_seed)

data_path = '/home/farahat/Documents/data/'
root_logdir = '/home/farahat/Documents/my_logs/final/'
batch_size = 512
models = sorted([f for f in os.listdir(root_logdir)])
z_dim = 80

for model_name in models[5:]:
    print('working on: '+model_name)

    animal_path = data_path+model_name[24:]

    run_logdir = root_logdir + model_name
    output_directory = run_logdir +  '/stats/'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    epg_files = get_data_files(animal_path+'/EPG/', train_valid_split=False)
    train_files, valid_files = get_data_files(animal_path+'/BL/')

    epg_set = csv_reader_dataset(epg_files, batch_size=batch_size, shuffle=False)
    valid_set = csv_reader_dataset(valid_files, batch_size=batch_size, shuffle=False)
    train_set = csv_reader_dataset(train_files, batch_size=batch_size, shuffle=False)

    encoder = tf.keras.models.load_model(run_logdir+'/encoder.h5')
    decoder = tf.keras.models.load_model(run_logdir+'/decoder.h5')
    disc_x = tf.keras.models.load_model(run_logdir+'/discriminator_x.h5')


    def compute_batch_distance(z):
        distance = []
        for i in range(z.shape[0]):
            distance.append(scipy.spatial.distance.euclidean(z[i].numpy(),np.zeros(z_dim)))
        return np.array(distance)


    def compute_distros(dataset, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        errors = np.array([])
        probilities = np.array([])
        distances = np.array([])
        z_all = np.zeros(z_dim)

        for i, batch in enumerate(dataset):
            z = encoder(batch)
            z_all = np.vstack((z_all,z[:,:,0].numpy()))

            x_hat = decoder(z)
            prob = scipy.special.expit(disc_x(x_hat)[0]).ravel()
            probilities = np.concatenate((probilities,prob),axis=0)


            loss = np.square(batch-x_hat)[:,:,0]
            error = loss.reshape(loss.shape[0]*loss.shape[1])
            errors = np.concatenate((errors,error),axis=0)

            distance = compute_batch_distance(z[:,:,0])
            distances = np.concatenate((distances,distance),axis=0)

            if (i+1) % 10 == 0:
                print('finished: '+str(i)+' batches')
        np.save(directory+'/errors.npy', errors)
        np.save(directory+'/probilities.npy', probilities)
        np.save(directory+'/distances.npy', distances)
        np.save(directory+'/z.npy', z_all[1:,:])       

    compute_distros(train_set, output_directory+'train_data')
    compute_distros(valid_set, output_directory+'valid_data')
    compute_distros(epg_set, output_directory+'epg_data')

    ####################################################################

    t_errors = np.load(output_directory+'train_data'+'/errors.npy')
    t_probilities = np.load(output_directory+'train_data'+'/probilities.npy')
    t_distances = np.load(output_directory+'train_data'+'/distances.npy')
    t_z_all = np.load(output_directory+'train_data'+'/z.npy')

    v_errors = np.load(output_directory+'valid_data'+'/errors.npy')
    v_probilities = np.load(output_directory+'valid_data'+'/probilities.npy')
    v_distances = np.load(output_directory+'valid_data'+'/distances.npy')
    v_z_all = np.load(output_directory+'valid_data'+'/z.npy')

    e_errors = np.load(output_directory+'epg_data'+'/errors.npy')
    e_probilities = np.load(output_directory+'epg_data'+'/probilities.npy')
    e_distances = np.load(output_directory+'epg_data'+'/distances.npy')
    e_z_all = np.load(output_directory+'epg_data'+'/z.npy')

    fig = plt.figure(figsize=(10,10))
    sns.distplot(e_errors, kde=False, norm_hist=True, label='epg errors')
    sns.distplot(t_errors, kde=False, norm_hist=True, label='train errors')
    sns.distplot(v_errors, kde=False, norm_hist=True, label='valid errors')
    plt.yscale('log')
    plt.legend()
    plt.savefig(output_directory+'errors.png')
    plt.close()

    fig = plt.figure(figsize=(10,10))
    sns.distplot(e_probilities, kde=False, norm_hist=True, label='epg probabilities')
    sns.distplot(t_probilities, kde=False, norm_hist=True, label='train probabilities')
    sns.distplot(v_probilities, kde=False, norm_hist=True, label='valid probabilities')
    plt.legend()
    plt.savefig(output_directory+'probabilities.png')
    plt.close()

    fig = plt.figure(figsize=(10,10))
    sns.distplot(e_distances, kde=False, norm_hist=True, label='epg distances')
    sns.distplot(t_distances, kde=False, norm_hist=True, label='train distances')
    sns.distplot(v_distances, kde=False, norm_hist=True, label='valid distances')
    plt.legend()
    plt.savefig(output_directory+'distances.png')
    plt.close()

    reshaped_t_errors = np.reshape(t_errors, (int(t_errors.shape[0]/2560),2560))
    whole_segment_t_errors = np.mean(reshaped_t_errors, axis = 1)
    np.save(output_directory+'train_data'+'/whole_segment_t_errors.npy', whole_segment_t_errors)
    # whole_segment_t_errors = np.load(output_directory+'train_data'+'/whole_segment_t_errors.npy')
    moving_average = pd.Series(whole_segment_t_errors).rolling(720*12).mean()
    moving_std = pd.Series(whole_segment_t_errors).rolling(720*12).std()
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(int(whole_segment_t_errors.shape[0])),whole_segment_t_errors, color='orange', label='errors', marker='.', s=1)  
    plt.plot(moving_average, linewidth=2, color='black', label='moving average')
    plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
    plt.ylim([0,1.2])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_t_errors.shape[0], 720*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1))
    plt.legend()
    plt.savefig(output_directory+'t_whole_segment_errors.png')
    plt.close()

    th90 = np.percentile(whole_segment_t_errors, 90)
    th95 = np.percentile(whole_segment_t_errors, 95)
    th99 = np.percentile(whole_segment_t_errors, 99)

    reshaped_v_errors = np.reshape(v_errors, (int(v_errors.shape[0]/2560),2560))
    whole_segment_v_errors = np.mean(reshaped_v_errors, axis = 1)
    np.save(output_directory+'valid_data'+'/whole_segment_v_errors.npy', whole_segment_v_errors)
    # whole_segment_v_errors = np.load(output_directory+'valid_data'+'/whole_segment_v_errors.npy')
    moving_average = pd.Series(whole_segment_v_errors).rolling(720).mean()
    moving_std = pd.Series(whole_segment_v_errors).rolling(720).std()
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(int(whole_segment_v_errors.shape[0])),whole_segment_v_errors, color='orange', label='errors', marker='.', s=1)  
    plt.plot(moving_average, linewidth=2, color='black', label='moving average')
    plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
    plt.hlines([th90, th95, th99], 0, len(whole_segment_v_errors), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
    plt.ylim([0,1.2])
    plt.xlabel('Time in hours')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_v_errors.shape[0], 720)
    plt.xticks(ticks, np.arange(0,len(ticks), 1))
    plt.legend()
    plt.savefig(output_directory+'v_whole_segment_errors.png')
    plt.close()

    reshaped_e_errors = np.reshape(e_errors, (int(e_errors.shape[0]/2560),2560))
    whole_segment_e_errors = np.mean(reshaped_e_errors, axis = 1)
    np.save(output_directory+'epg_data'+'/whole_segment_e_errors.npy', whole_segment_e_errors)
    # whole_segment_e_errors = np.load(output_directory+'epg_data'+'/whole_segment_e_errors.npy')
    moving_average = pd.Series(whole_segment_e_errors).rolling(720*12).mean()
    moving_std = pd.Series(whole_segment_e_errors).rolling(720*12).std()
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(int(whole_segment_e_errors.shape[0])),whole_segment_e_errors, color='orange', label='errors', marker='.', s=1)
    plt.plot(moving_average, linewidth=2, color='black', label='moving average')
    plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
    plt.hlines([th90, th95, th99], 0, len(whole_segment_e_errors), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
    plt.ylim([0,1.2])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_e_errors.shape[0], 720*24)
    plt.xticks(ticks, np.arange(0,len(ticks)*24, 1)) 
    plt.legend()
    plt.savefig(output_directory+'e_whole_segment_errors.png')
    plt.close()

    fig = plt.figure(figsize=(10,10))
    sns.distplot(whole_segment_e_errors, kde=False, norm_hist=True, label='epg errors')
    sns.distplot(whole_segment_t_errors, kde=False, norm_hist=True, label='train errors')
    sns.distplot(whole_segment_v_errors, kde=False, norm_hist=True, label='valid errors')
    plt.legend()
    plt.savefig(output_directory+'whole_segment_errors.png')
    plt.close()

    th_whole_segment_t_errors = np.copy(whole_segment_t_errors)
    th_whole_segment_t_errors[th_whole_segment_t_errors < th99] = 0
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(len(th_whole_segment_t_errors)),th_whole_segment_t_errors, marker='.', color='orange', s=1)
    plt.hlines(th99, 0, len(whole_segment_t_errors), colors='green', linewidth=1, linestyles='dashed', label='99th percentiles')
    plt.ylim([0,1.2])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_t_errors.shape[0], 720*24)
    plt.xticks(ticks, np.arange(0,len(ticks)*24, 1)) 
    plt.savefig(output_directory+'th_t_whole_segment_errors.png')
    plt.close()

    th_whole_segment_v_errors = np.copy(whole_segment_v_errors)
    th_whole_segment_v_errors[th_whole_segment_v_errors < th99] = 0
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(len(th_whole_segment_v_errors)),th_whole_segment_v_errors, marker='.', color='orange', s=1)
    plt.hlines(th99, 0, len(whole_segment_v_errors), colors='green', linewidth=1, linestyles='dashed', label='99th percentiles')
    plt.ylim([0,1.2])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_v_errors.shape[0], 720*24)
    plt.xticks(ticks, np.arange(0,len(ticks)*24, 1)) 
    plt.savefig(output_directory+'th_v_whole_segment_errors.png')
    plt.close()

    th_whole_segment_e_errors = np.copy(whole_segment_e_errors)
    th_whole_segment_e_errors[th_whole_segment_e_errors < th99] = 0
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(len(th_whole_segment_e_errors)),th_whole_segment_e_errors, marker='.', color='orange', s=1)
    plt.hlines(th99, 0, len(whole_segment_e_errors), colors='green', linewidth=1, linestyles='dashed', label='99th percentiles')
    plt.ylim([0,1.2])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_e_errors.shape[0], 720*24)
    plt.xticks(ticks, np.arange(0,len(ticks)*24, 1)) 
    plt.savefig(output_directory+'th_e_whole_segment_errors.png')
    plt.close()

    test_window_in_minutes = [1,5,15,30,60]
    for window_in_minutes in test_window_in_minutes:
        window = int((window_in_minutes*60)/5)
        r = np.reshape(th_whole_segment_t_errors[:len(th_whole_segment_t_errors)//window*window], (-1, window))
        frequency_t = np.sum(np.where(r>0, 1, 0),axis=1)    
        fig = plt.figure(figsize=(20,10))
        plt.plot(np.arange(len(frequency_t)),frequency_t)
        plt.xlabel('Time in days')
        plt.ylabel('#suprathrehold segments per '+str(window_in_minutes)+' minutes')
        ticks = np.arange(0,frequency_t.shape[0], (720/window)*24)
        plt.xticks(ticks, np.arange(0,len(ticks)*24, 1)) 
        plt.savefig(output_directory+'frequency_t_'+str(window_in_minutes)+' minutes.png')
        plt.close()

        th99_frequency = np.percentile(frequency_t, 99)

        r = np.reshape(th_whole_segment_v_errors[:len(th_whole_segment_v_errors)//window*window], (-1, window))
        frequency_v = np.sum(np.where(r>0, 1, 0),axis=1)    
        fig = plt.figure(figsize=(20,10))
        plt.plot(np.arange(len(frequency_v)),frequency_v)
        if np.any(frequency_v > th99_frequency):
            plt.axvline(np.where(frequency_v > th99_frequency)[0][0], c='r', linewidth=3, linestyle='dashed')
        plt.xlabel('Time in days')
        plt.ylabel('#suprathrehold segments per '+str(window_in_minutes)+' minutes')
        ticks = np.arange(0,frequency_v.shape[0], (720/window)*24)
        plt.xticks(ticks, np.arange(0,len(ticks)*24, 1)) 
        plt.savefig(output_directory+'frequency_v_'+str(window_in_minutes)+' minutes.png')
        plt.close()

        r = np.reshape(th_whole_segment_e_errors[:len(th_whole_segment_e_errors)//window*window], (-1, window))
        frequency_e = np.sum(np.where(r>0, 1, 0),axis=1)    
        fig = plt.figure(figsize=(20,10))
        plt.plot(np.arange(len(frequency_e)),frequency_e)
        if np.any(frequency_e > th99_frequency):
            plt.axvline(np.where(frequency_e > th99_frequency)[0][0], c='r', linewidth=3, linestyle='dashed')
        plt.xlabel('Time in days')
        plt.ylabel('#suprathrehold segments per'+str(window_in_minutes)+' minutes')
        ticks = np.arange(0,frequency_e.shape[0], (720/window)*24)
        plt.xticks(ticks, np.arange(0,len(ticks)*24, 1)) 
        plt.savefig(output_directory+'frequency_e_'+str(window_in_minutes)+' minutes.png')
        plt.close()

    #############################################################################
