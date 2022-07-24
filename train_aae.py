import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pickle

from aae import AAE
from input_pipeline import csv_reader_dataset, get_train_val_files, get_data_files_LOO
from utils import get_run_logdir, plot_dict_loss

LOO = True # False
paths_platforms = {
                   "FIAS_cluster": {"PPS_data_path": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/PPS-Rats",
                                    "Ctrl_data_path": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/Control-Rats",
                                    "root_logdir": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/resultsl-7rats"
                                    }                   
                   }

platform = "FIAS_cluster"
PPS_data_path = paths_platforms[platform]["PPS_data_path"]
Ctrl_data_path = paths_platforms[platform]["Ctrl_data_path"]
root_logdir = paths_platforms["Farahat"]["root_logdir"]
#
n_sec_per_sample = 1
sampling_rate = 512
input_size = n_sec_per_sample * sampling_rate
h_dim = 384
z_dim = 64
n_epochs=100
batch_size = 128

# LOO_animals = ["1275", "1276", "32140", "3267", "3263"]
LOO_animals = ["1227", "1237", "3266", "32141", "1270"]



# LOO_animals = ["1275", "1276"]
# LOO_animals = ["32140", "32141"]
# LOO_animals = ["3263", "3266"]
# LOO_animals = ["1227", "1237"]
# LOO_animals = ["3267", "1270"]


n_files2use = 5


for LOO_animal in LOO_animals:
    run_logdir = get_run_logdir(root_logdir, LOO_animal)
    train_files, valid_files = [], []
    if_LOO_ctrl = True if "326" in LOO_animal else False  # determines whether it is in the LOO control rats case, then we need to get all PPS data and only do LOO in control
    if LOO:
        PPS_train_files, PPS_valid_files = get_data_files_LOO(PPS_data_path,
                                                              train_valid_split=True,
                                                              train_percentage=0.9, num2use=30,
                                                              LOO_ID=LOO_animal,
                                                              if_LOO_ctrl=if_LOO_ctrl,
                                                              current_folder="PPS")
        Ctrl_train_files, Ctrl_valid_files = get_data_files_LOO(Ctrl_data_path,
                                                                train_valid_split=True,
                                                                train_percentage=0.9,
                                                                num2use=60,
                                                                LOO_ID=LOO_animal,
                                                                if_LOO_ctrl=if_LOO_ctrl,
                                                                current_folder="Ctrl")
        train_files.extend(PPS_train_files)
        train_files.extend(Ctrl_train_files)
        valid_files.extend(PPS_valid_files)
        valid_files.extend(Ctrl_valid_files)
    else:
        train_files, valid_files = get_train_val_files(PPS_data_path, train_valid_split=True, train_percentage=0.8, num2use=n_files2use)
    np.savetxt(os.path.join(run_logdir, "picked_train_files_{}.csv".format(len(train_files))), np.array(train_files), fmt="%s", delimiter=",")
    np.savetxt(os.path.join(run_logdir, "picked_val_files_{}.csv".format(len(valid_files))), np.array(valid_files), fmt="%s", delimiter=",")
    train_set = csv_reader_dataset(train_files, batch_size=batch_size, n_sec_per_sample=n_sec_per_sample,
                           sr=sampling_rate)
    valid_set = csv_reader_dataset(valid_files, batch_size=batch_size, n_sec_per_sample=n_sec_per_sample,
                           sr=sampling_rate)
    
    
    model = AAE(input_size, h_dim, z_dim, run_logdir)
    model.plot_models()
    metrics = model.train(n_epochs, train_set, valid_set)
    with open(run_logdir+'/metrics.pickle', 'wb') as handle:
        pickle.dump(metrics, handle)
    plot_dict_loss(metrics, run_logdir)
    model.clear_model()
    
    
    
    
    
