
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
from os import listdir
from os.path import isfile, join
import random
import os

random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)


# list the files
def get_data_files_from_folder(path, train_valid_split=True, train_percentage=0.8):
    """
    Get files belong to one rat given the path
    :param path:
    :param train_valid_split:
    :param train_percentage:
    :return:
    """
    files = sorted([path+f for f in listdir(path) if isfile(join(path, f))])
    files = list(filter(lambda x: "new.csv" in x, files))
    if train_valid_split:
        train_files = files[:round(len(files)*train_percentage)]
        valid_files = files[round(len(files)*train_percentage):]
        return train_files, valid_files
    else:
        return files

def get_all_data_files(data_path, test_animal, train_valid_split=True, train_percentage=0.9):
    """
    Get LOO data from the rest of the animals
    :param data_path: str
    :param test_animal: str
    :param train_valid_split:
    :param train_percentage:
    :return:
    """
    animals = sorted([f for f in listdir(data_path)])
    animals.remove(test_animal)
    files = []
    for animal in animals:
        animal_path = data_path + animal
        files.extend(sorted([animal_path+'/BL/'+f for f in listdir(animal_path+'/BL/')]))
    if train_valid_split:
        random.shuffle(files)
        train_files = files[:round(len(files)*train_percentage)]
        valid_files = files[round(len(files)*train_percentage):]
        return train_files, valid_files
    else:
        return files


# list the files
def get_train_val_files(data_path, train_valid_split=True, train_percentage=0.8, num2use=None):
    """
    Get files from all animals
    :param data_path:
    :param train_valid_split:
    :param train_percentage:
    :param num2use:
    :return:
    """
    animals = sorted([f for f in os.listdir(data_path)])[:]  # all animal IDs
    files_list, train_file_list, valid_file_list = [], [], []
    
    for animal in animals:
        animal_path = os.path.join(data_path, animal, animal)
        BL_path = os.path.join(animal_path, 'BL')
        BL_files = sorted([BL_path + f for f in listdir(BL_path) if isfile(join(BL_path, f))])
        BL_files = list(filter(lambda x: "new.csv" in x, BL_files))
        np.random.shuffle(BL_files)

        num2use = len(BL_files) if not num2use else num2use  # when num2use is None, then take all files from this animal
        picked_files = BL_files[0:min(len(BL_files), num2use)]
        files_list.extend(picked_files)
        if train_valid_split:
            train_file_list.extend(picked_files[
                                      :round(len(picked_files) * train_percentage)])
            valid_file_list.extend(picked_files[
                          round(len(picked_files) * train_percentage):])
            
    if train_valid_split:
        np.random.shuffle(train_file_list)
        np.random.shuffle(valid_file_list)
        return train_file_list, valid_file_list
    else:
        np.random.shuffle(files_list)
        return files_list  # the valid list is empty
        
        
def get_data_files_LOO(data_path, train_valid_split=True,
                       train_percentage=0.9, num2use=None,
                       LOO_ID=None, if_LOO_ctrl=False, current_folder="PPS"):
    """
    Get both BL and EPG files
    :param data_path: str, data root dir
    :param train_file_list: list, with training file names
    :param valid_file_list: list, with validation file names
    :param train_valid_split: bool,
    :param train_percentage: percentage of training files
    :param if_LOO: bool, whether remove the LOO rat ID
    :param LOO_ID: str, test animal id
    :param with_control: bool, whether include control animals' data
    :param current_folder: str, the current group "PPS" or "Ctrl"
    :param num2use: int, None when take all files. number of files to randomly pick for training and validation
    :return:
    """
    PPS_animals = ["1227", "1237", "1270", "1275", "1276", "32140", "32141"]#sorted([f for f in listdir(data_path)])
    Ctrl_animals = ["3263", "3266", "3267"]#sorted([f for f in listdir(data_path)])
    assert LOO_ID is not None, "You have to put in the LOO animal ID" # if LOO_ID is not None
    if current_folder == "PPS" and not if_LOO_ctrl:  #leave out PPS
        PPS_animals.remove(LOO_ID)  # Leave out one animal
        animals = PPS_animals
    elif current_folder == "PPS" and if_LOO_ctrl:  # get all PPS data, only BL
        animals = PPS_animals
    elif current_folder == "Ctrl" and not if_LOO_ctrl:  #then get all data BL + EPG
        animals = Ctrl_animals
    elif current_folder == "Ctrl" and if_LOO_ctrl:   # then leave one animal, get BL + EPG
        Ctrl_animals.remove(LOO_ID)
        animals = Ctrl_animals
        
    files_list, train_file_list, valid_file_list = [], [], []
    for animal in animals:
        animal_path = os.path.join(data_path, animal, animal)
        
        if current_folder == "Ctrl":  # Get BL + EPG
            files_of_this_animal = sorted(
                [os.path.join(animal_path, 'BL', f) for f in
                 listdir(os.path.join(animal_path, 'BL'))])
            files_of_this_animal = list(filter(lambda x: "new.csv" in x,
                                               files_of_this_animal))  # get only .csv files

            
            files_of_this_animal2 = sorted(
                            [os.path.join(animal_path, 'EPG', f) for f in
                             listdir(os.path.join(animal_path, 'EPG'))])
            files_of_this_animal2 = list(filter(lambda x: "new.csv" in x,
                                               files_of_this_animal))  # get only .csv files

            files_of_this_animal.extend(files_of_this_animal2)
            
            np.random.shuffle(files_of_this_animal)
            num2use = len(
                files_of_this_animal) if not num2use else num2use  # when num2use is None, then take all files from this animal

            picked_files = files_of_this_animal[0:min(len(files_of_this_animal),
                                                      num2use)]

        else:
            files_of_this_animal = sorted([os.path.join(animal_path, 'BL', f) for f in listdir(os.path.join(animal_path, 'BL'))])
            files_of_this_animal = list(filter(lambda x: "new.csv" in x,
                                               files_of_this_animal))  # get only .csv files
            np.random.shuffle(files_of_this_animal)
            num2use = len(
                files_of_this_animal) if not num2use else num2use  # when num2use is None, then take all files from this animal
            picked_files = files_of_this_animal[0:min(len(files_of_this_animal),
                                           num2use)] # randomly pick a certai number of hours
        
        files_list.extend(picked_files)
    
        if train_valid_split:
            random.shuffle(picked_files)
            train_file_list.extend(picked_files[:round(len(picked_files)*train_percentage)])
            valid_file_list.extend(picked_files[round(len(picked_files)*train_percentage):])
        
    if train_valid_split:
        np.random.shuffle(train_file_list)
        np.random.shuffle(valid_file_list)
        return train_file_list, valid_file_list
    else:
        np.random.shuffle(files_list)
        return files_list  # the valid list is empty


def csv_reader_dataset(filepaths, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=tf.data.experimental.AUTOTUNE,
                       batch_size=32, shuffle=True, n_sec_per_sample=1,
                       sr=512):
    def read(line):
        n_inputs = 2561
        defs = [tf.constant([], dtype=tf.string)] + [0.] * n_inputs
        fields = tf.io.decode_csv(line, record_defaults=defs)[
                 2:]  # exclude filename and label
        x = tf.stack(fields)
        x = tf.expand_dims(x, 1)
        return x
    
    n_row_per_file = 720
    dataset = tf.data.Dataset.list_files(filepaths, shuffle=shuffle)
    shuffle_buffer_size = min(len(filepaths) * n_row_per_file, 10000)
    if shuffle:
        dataset = dataset.interleave(
            lambda filepath: tf.data.TextLineDataset(filepath).skip(1), # why skip 1?
            cycle_length=n_readers,
            num_parallel_calls=n_read_threads)
        dataset = dataset.shuffle(shuffle_buffer_size)
    else:
        dataset = tf.data.TextLineDataset(dataset)
    dataset = dataset.map(map_func=lambda x: read(x), num_parallel_calls=n_parse_threads)
    dataset = dataset.map(lambda x: (x-tf.reduce_mean(x)) / (tf.math.reduce_std(x) + np.finfo(np.float32).eps), num_parallel_calls=n_parse_threads)

    # reshape the sample to 1 second
    def reshape_to_k_sec(feature, n_sec=1, sr=512):
        return tf.reshape(feature[:(5//n_sec)*n_sec*sr], (5//n_sec, n_sec*sr, 1))  # flexible to the number of sec per sample

    dataset = dataset.map(map_func=lambda x: reshape_to_k_sec(x, n_sec=n_sec_per_sample, sr=sr), num_parallel_calls=n_parse_threads)
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(2)
