from input_pipeline import csv_reader_dataset, get_data_files, get_all_data_files
from utils import get_run_logdir, KnuthMorrisPratt
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
from sklearn import metrics


random_seed = 42
tf.random.set_random_seed(random_seed)
np.random.seed(random_seed)

def compute_no_sequences(whole_errors, th99, window = 60, sequence=[1,1,1]):
    random_start = np.random.randint(0,len(whole_errors)-window*12)
    random_one_window = whole_errors[random_start:random_start+window*12]
    th = np.where(random_one_window>th99)[0]
    subs = [y - x for x,y in zip(th,th[1:])]
    return len(list(KnuthMorrisPratt(subs, sequence)))

def plot_roc(fpr, tpr, roc_auc, animal, title, output_directory):
    # plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label=animal+' (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC \n'+title)
    plt.legend(loc="lower right")
    # plt.savefig(output_directory+title+'.png')
    # plt.close()


data_path = '/home/farahat/Documents/data/'
root_logdir = '/home/farahat/Documents/my_logs/final3/'
models = sorted([f for f in os.listdir(root_logdir)])
no_samples = 100
window_in_minutes = 60

for model_name in models[:]:
    print('working on: '+model_name)
    animal = model_name[24:]
    run_logdir = root_logdir + model_name
    output_directory = run_logdir +  '/stats/'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)


    whole_segment_t_errors = np.load(output_directory+'train_data/whole_segment_t_errors.npy')
    whole_segment_v_errors = np.load(output_directory+'valid_data/whole_segment_v_errors.npy')
    whole_segment_e_errors = np.load(output_directory+'epg_data/whole_segment_e_errors.npy')

    d_t = np.load(output_directory+'train_data/distances.npy')
    d_v = np.load(output_directory+'valid_data/distances.npy')
    d_e = np.load(output_directory+'epg_data/distances.npy')


    th99 = np.percentile(whole_segment_t_errors, 99)
    th99_d = np.percentile(d_t, 99)

    no_v_seq = []
    no_e_seq = []

    for i in range(no_samples):
        no_v_seq.append(compute_no_sequences(whole_segment_v_errors, th99, window=window_in_minutes, sequence=[1,1,1]))
        no_e_seq.append(compute_no_sequences(whole_segment_e_errors, th99, window=window_in_minutes, sequence=[1,1,1]))
    no_v_seq = np.array(no_v_seq)
    no_e_seq = np.array(no_e_seq)

    
    window = int((window_in_minutes*60)/5)

    r = np.where(whole_segment_v_errors>th99, 1, 0)
    frequency_v = pd.Series(r).rolling(window).sum().dropna().values
    r = np.where(whole_segment_e_errors>th99, 1, 0)
    frequency_e = pd.Series(r).rolling(window).sum().dropna().values

    r = np.where(d_v>th99_d, 1, 0)
    frequency_v_d = pd.Series(r).rolling(window).sum().dropna().values
    r = np.where(d_e>th99_d, 1, 0)
    frequency_e_d = pd.Series(r).rolling(window).sum().dropna().values

    idx_v = np.random.choice(frequency_v.shape[0], no_samples)
    idx_e = np.random.choice(frequency_e.shape[0], no_samples)

    no_v_re = frequency_v[idx_v]
    no_e_re = frequency_e[idx_e]
    no_v_d = frequency_v_d[idx_v]
    no_e_d = frequency_e_d[idx_e]



    re_contrib = 0.8
    no_v = list(re_contrib*no_v_re + (1 - re_contrib)*no_v_d)
    no_e = list(re_contrib*no_e_re + (1 - re_contrib)*no_e_d)


    re_contrib = 0.5
    no_v_seq_re = list(re_contrib*no_v_re + (1 - re_contrib)*no_v_seq)
    no_e_seq_re = list(re_contrib*no_e_re + (1 - re_contrib)*no_e_seq)

    y = list(np.zeros(no_samples))+list(np.ones(no_samples))


    # fpr, tpr, thresholds = metrics.roc_curve(y, no_v+no_e)
    # roc_auc = metrics.auc(fpr, tpr)
    # print(thresholds)
    # plot_roc(fpr, tpr, roc_auc, animal, 'Average reconstruction errors and distance', output_directory)

    # fpr, tpr, thresholds = metrics.roc_curve(y, list(no_v_re)+list(no_e_re))
    # roc_auc = metrics.auc(fpr, tpr)
    # print(thresholds)
    # plot_roc(fpr, tpr, roc_auc, animal, 'Only reconstruction errors', output_directory)

    # fpr, tpr, thresholds = metrics.roc_curve(y, list(no_v_d)+list(no_e_d))
    # roc_auc = metrics.auc(fpr, tpr)
    # print(thresholds)
    # plot_roc(fpr, tpr, roc_auc, animal, 'Only distance', output_directory)

    # fpr, tpr, thresholds = metrics.roc_curve(y, list(no_v_seq)+list(no_e_seq))
    # roc_auc = metrics.auc(fpr, tpr)
    # print(thresholds)
    # plot_roc(fpr, tpr, roc_auc, animal, 'Only subsequent suprathreshold events', output_directory)

    fpr, tpr, thresholds = metrics.roc_curve(y, list(no_v_seq_re)+list(no_e_seq_re))
    roc_auc = metrics.auc(fpr, tpr)
    print(thresholds)
    plot_roc(fpr, tpr, roc_auc, animal, 'Average subsequent suprathreshold events and Reconstruction errors', output_directory)

plt.savefig('5.png')
plt.close()