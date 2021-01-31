from input_pipeline import csv_reader_dataset, get_train_val_files, get_data_files_LOO
from utils import get_run_logdir, KnuthMorrisPratt
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import scipy
import math
import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd
from sklearn import metrics


def compute_no_sequences(whole_errors, th99, window = 60, sequence=[1,1,1]):
    random_start = np.random.randint(0,len(whole_errors)-window*12)
    random_one_window = whole_errors[random_start:random_start+window*12]
    th = np.where(random_one_window>th99)[0]
    subs = [y - x for x,y in zip(th,th[1:])]
    return len(list(KnuthMorrisPratt(subs, sequence)))

def plot_roc(fpr, tpr, roc_auc, animal, title):
    lw = 2
    plt.plot(fpr, tpr,
             lw=lw, label=animal+' (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC \n'+title)
    plt.legend(loc="lower right")



root_logdir = '/home/farahat/Documents/my_logs/final_icml/'
models = sorted([f for f in os.listdir(root_logdir) if os.path.isdir(os.path.join(root_logdir, f))])
no_samples = 100
window_in_minutes = 60
sequence = [1,1]


y_s = []
no_v_re_s = []
no_e_re_s = []
no_v_d_s = []
no_e_d_s = []

for model_name in models[:]:
    print('working on: '+model_name)
    animal = model_name[40:]
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


    
    window = int((window_in_minutes*60)/1)

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

    no_v_re_s.append(no_v_re)
    no_e_re_s.append(no_e_re)
    no_v_d_s.append(no_v_d)
    no_e_d_s.append(no_e_d)


    y = list(np.zeros(no_samples))+list(np.ones(no_samples))
    y_s.append(y)


for k in range(len(models)):
    no_v_re = no_v_re_s[k]
    no_e_re = no_e_re_s[k]

    y = y_s[k]

    fpr, tpr, thresholds = metrics.roc_curve(y, list(no_v_re)+list(no_e_re))
    roc_auc = metrics.auc(fpr, tpr)
    print(thresholds)
    plot_roc(fpr, tpr, roc_auc, animal, 'Only reconstruction errors')

plt.savefig(root_logdir+'Only reconstruction errors.png')
plt.close()

for k in range(len(models)):

    no_v_d = no_v_d_s[k]
    no_e_d = no_e_d_s[k]
    y = y_s[k]

    fpr, tpr, thresholds = metrics.roc_curve(y, list(no_v_d)+list(no_e_d))
    roc_auc = metrics.auc(fpr, tpr)
    print(thresholds)
    plot_roc(fpr, tpr, roc_auc, animal, 'Only probability w.r.t the latent space distribution')

plt.savefig(root_logdir+'Only probability w.r.t the latent space distribution.png')
plt.close()

for i in np.arange(0,1,0.1):
    for k in range(len(models)):
        no_v_re = no_v_re_s[k]
        no_e_re = no_e_re_s[k]
        no_v_d = no_v_d_s[k]
        no_e_d = no_e_d_s[k]
        y = y_s[k]

        
        re_contrib = i
        no_v = re_contrib*no_v_re + (1 - re_contrib)*no_v_d
        no_e = re_contrib*no_e_re + (1 - re_contrib)*no_e_d

        fpr, tpr, thresholds = metrics.roc_curve(y, list(no_v)+list(no_e))
        roc_auc = metrics.auc(fpr, tpr)
        print(thresholds)
        plot_roc(fpr, tpr, roc_auc, animal, 'Average reconstruction errors and probability w.r.t the latent space distribution_'+str(i))


    plt.savefig(root_logdir+'Average reconstruction errors and probability w.r.t the latent space distribution_'+str(i)+'.png')
    plt.close()