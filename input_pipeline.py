
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import random
import pdb
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
            np.random.shuffle(files_of_this_animal)
            picked_files = files_of_this_animal[0:min(len(files_of_this_animal),
                                                      15)]  # TODO: hard coded num2use for control BL
            
            files_of_this_animal = sorted(
                            [os.path.join(animal_path, 'EPG', f) for f in
                             listdir(os.path.join(animal_path, 'EPG'))])
            files_of_this_animal = list(filter(lambda x: "new.csv" in x,
                                               files_of_this_animal))  # get only .csv files
            np.random.shuffle(files_of_this_animal)
            num2use = len(
                files_of_this_animal) if not num2use else num2use  # when num2use is None, then take all files from this animal
            picked_files.extend(files_of_this_animal[0:min(len(files_of_this_animal),
                                                      num2use)]) # randomly pick a certai number of hours
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

def compute_data_parameters(files, dims=2560):

    sums = np.zeros((dims,))
    counter = 0
    for file in files:
        data = np.genfromtxt(file, delimiter=',')[:,2:]
        counter += data.shape[0]
        sums += data.sum(axis=0)
    mean = sums/counter

    sums = np.zeros((dims,))
    for file in files:
        data = np.genfromtxt(file, delimiter=',')[:,2:]
        sums += ((data-mean)**2).sum(axis=0)
    std=np.sqrt(sums/(counter-1))

    return mean, std


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
    # dataset = dataset.map(lambda x: (x-mean) / (std + np.finfo(np.float32).eps), num_parallel_calls=n_parse_threads)
    dataset = dataset.map(lambda x: (x-tf.reduce_mean(x)) / (tf.math.reduce_std(x) + np.finfo(np.float32).eps), num_parallel_calls=n_parse_threads)

    # reshape the sample to 1 second
    def reshape_to_k_sec(feature, n_sec=1, sr=512):
        return tf.reshape(feature[:(5//n_sec)*n_sec*sr], (5//n_sec, n_sec*sr, 1))  # flexible to the number of sec per sample

    dataset = dataset.map(map_func=lambda x: reshape_to_k_sec(x, n_sec=n_sec_per_sample, sr=sr), num_parallel_calls=n_parse_threads)
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    # dataset = dataset.map(lambda x: (x,x) , num_parallel_calls=n_parse_threads)
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(2)

################## Lu Data input pipeline ##########################
def get_train_test_files_split(root, fns, ratio,
                               rat_id="1227", label=0, num2use=100):
    """
    Get equal number of files for testing from each folder
    :param fns: list, all file names from the folder
    :param ratio: float, the test file ratio.
    :param label: int or list, the label need to be assigned to the file. In different experiments, the same file might be assigned with different labels.
    :param num2use: int, the number of files that want to use(randomize file selection)
    :return: lists, edited train and test file lists
    """
    train_list = []
    test_list = []
    rand_inds = np.arange(len(fns)).astype(np.int)
    np.random.shuffle(rand_inds)
    if isinstance(label, int):  # get the randomly selected files with their labels
        labels = np.repeat(label, len(fns))
        rand_fns = np.array(fns)[rand_inds]
    elif isinstance(label, list):
        labels = np.array(label)[rand_inds]
        rand_fns = np.array(fns)[rand_inds]
    
    num_files_need = min(len(rand_fns), num2use)
    
    num_test_files = np.ceil(ratio * num_files_need).astype(np.int)
    
    train_within_folder = []
    
    # get the filenames, the label, number of rows, and rat_id for future analysis
    for ind, f, lb in zip(np.arange(num_files_need), rand_fns[0:num_files_need],
                          labels):
        num_rows = os.path.basename(f).split('-')[-2]
        # num_rows = os.path.basename(f).split('-')[-1].split('.')[0]
        if ind < num_test_files:
            test_list.append((os.path.join(root, f), lb, num_rows, rat_id))
        else:
            train_list.append((os.path.join(root, f), lb, num_rows, rat_id))
            train_within_folder.append(
                (os.path.join(root, f), lb, num_rows, rat_id))
    
    return train_list, test_list


def create_dataset(filenames_w_lb, args, batch_size=32, if_shuffle=True,
                   if_repeat=True):
    """

    :param filenames_w_lb:
    :param batch_size:
    :param if_shuffle:
    :return:
    """

    def parse_function(filename, label, args):
        """
        parse the file. It does following things:
        1. init a TextLineDataset to read line in the files
        2. decode each line and group args.secs_per_samp*args.num_segs rows together as one sample
        3. repeat the label for each long chunk
        4. return the transformed dataset
        :param filename: str, file name
        :param label: int, label of the file
        :param num_rows: int, the number of rows in the file (since they are artifacts free)
        :param args: Param object, contains hyperparams
        :return: transformed dataset with the label of the file assigned to each batch of data from the file
        """
        
        # three functions used for parsing each line
        def decode_label_fn(features, label, filename, assign_label=0):
            """
            Modify the label for each sample given different data_mode. E.g., EPG by default is 1, but i EPG_id mode,
            """
            return features, assign_label, filename
        
        def decode_csv(line, args=None):
            # Map function to decode the .csv file in TextLineDataset
            # @param line object in TextLineDataset
            # @return: zipped Dataset, (features, labels)
            defaults = [['']] + [[0.0]] * (
                        args.sr * args.secs_per_row + 1)  # there are 5 sec in one row
            csv_row = tf.compat.v1.decode_csv(line, record_defaults=defaults)
        
            filename = tf.cast(csv_row[0], tf.string)
            label = tf.cast(csv_row[1], tf.int32)  # given the label
            features = tf.cast(tf.stack(csv_row[2:]), tf.float32)
        
            return features, label, filename
        
        def scale_to_zscore(data, label, filename):
            """
            zscore normalize the features
            :param data: 2d-array, batch_size, seq_len
            :param label: 1d-array, batch_size,
            :param filename: 1d-array, batch_size, seq_len
            :return: normalized data
            """
            # ret = tf.nn.moments(data, 0)
            mean = tf.reduce_mean(data)
            std = tf.compat.v1.math.reduce_std(data)
            zscore = (data - mean) / (std + 1e-13)
        
            return zscore, label, filename

        skip = 0
    
        decode_ds = tf.compat.v1.data.TextLineDataset(filename).skip(skip).map(
            lambda line: decode_csv(line, args=args))
        # decode_ds = decode_ds.map(lambda feature: decode_label_fn(feature, assign_label=label, assign_fn=filename))
        decode_ds = decode_ds.map(
            lambda feature, lb, fn: decode_label_fn(feature, lb, fn, assign_label=label))
    
        decode_ds = decode_ds.map(scale_to_zscore)  # zscore norm the data
    
        return decode_ds

    if if_shuffle:
        inds = np.arange(len(filenames_w_lb))
        np.random.shuffle(inds)
    else:
        inds = np.arange(len(filenames_w_lb))
    
    labels = np.array(filenames_w_lb)[:, 1][inds].astype(np.int32)
    filenames = np.array(filenames_w_lb)[:, 0][inds].astype(np.str)
    num_rows = np.array(filenames_w_lb)[:, 2][inds].astype(np.int32)
    file_ids = list(np.array(filenames_w_lb)[:, 3][inds].astype(np.str))
    
    # get dataset from list of filenames and their corresponding labels
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.flat_map(
        lambda fname, lbs: parse_function(fname, lbs, args=args))
    
    rat_ids = []
    for id, num in zip(file_ids, num_rows):
        rat_ids += [id] * np.int(num)
    ds_rat_ids = tf.compat.v1.data.Dataset.from_tensor_slices(
        (rat_ids))  # up to now, each row is one element in the dataset
    # add rat_ids also in the dataset to keep track
    comb_ds = tf.compat.v1.data.Dataset.zip((dataset, ds_rat_ids))
    if if_shuffle:
        comb_ds = comb_ds.shuffle(buffer_size=10000)  # fn_lb: filename and label
    if if_repeat:
        comb_ds = comb_ds.repeat()  # fn_lb: filename and label
    comb_ds = comb_ds.batch(batch_size, drop_remainder=True)
    
    return comb_ds.prefetch(2)


def creat_data_tensors(dataset, data_tensors, filenames_w_lb, args, batch_size=32,
                       prefix='test'):
    """
    Create the data tensors for test or train
    :param dataset:
    :param data_tensors:
    :param filenames_w_lb:
    :param args:
    :param batch_size:
    :param prefix:
    :return:
    """
    num_rows = np.array(filenames_w_lb)[:, 2].astype(np.int)

    iter = dataset.make_initializable_iterator()
    batch_ds = iter.get_next()  # test contains features and label
    data_tensors["{}_iter_init".format(prefix)] = iter.initializer
    
    data_tensors["{}_features".format(prefix)] = tf.reshape(batch_ds[0][0],
                                                                [-1,
                                                                 args.sr * args.secs_per_samp])
    
    data_tensors["{}_labels".format(prefix)] = tf.one_hot(
        tf.repeat(batch_ds[0][1],
                  repeats=args.secs_per_row // args.secs_per_samp, axis=0),
        args.num_classes,
        dtype=tf.int32)
    # data_tensors["{}_filenames".format(prefix)] = tf.cast(batch_ds[0][2], dtype=tf.string)
    # data_tensors["{}_ids".format(prefix)] = tf.cast(batch_ds[1], dtype=tf.string)
    data_tensors["{}_filenames".format(prefix)] = tf.cast(
        tf.repeat(batch_ds[0][2],
                  repeats=args.secs_per_row // args.secs_per_samp, axis=0),
        dtype=tf.string)
    data_tensors["{}_ids".format(prefix)] = tf.cast(tf.repeat(batch_ds[1],
                                                              repeats=args.secs_per_row // args.secs_per_samp,
                                                              axis=0),
                                                    dtype=tf.string)
    data_tensors["tot_{}_batches".format(prefix)] = np.int(
        (np.sum(num_rows) * args.secs_per_row // args.secs_per_samp) //
        data_tensors["{}_features".format(prefix)].get_shape().as_list()[
            0])  # when use non-5 sec as length, the batchsize changes
    
    return data_tensors


def get_data_tensors(args, if_shuffle_train=True, if_shuffle_test=True,
                     if_repeat_train=True, if_repeat_test=True):
    """
    :param args: contrain hyperparams
    :return: train_data: dict, contains 'features', 'labels'
    :return: test_data, dict, contains 'features', 'labels'
    :return: num_samples, dict, contains 'n_train', 'n_test'
    """
    data_tensors = {}
    
    train_f_with_l, test_f_with_l = find_files(args)
    files = sorted([path + f for f in listdir(path) if isfile(join(path, f))])
    
    test_ds = create_dataset(test_f_with_l, args,
                             batch_size=args.test_bs,
                             if_shuffle=if_shuffle_test,
                             if_repeat=if_repeat_test)
    
    data_tensors = creat_data_tensors(test_ds, data_tensors,
                                      test_f_with_l, args,
                                      batch_size=args.test_bs,
                                      prefix='test')
    
    if not args.test_only:
        train_ds = create_dataset(train_f_with_l, args,
                                  batch_size=args.batch_size,
                                  if_shuffle=if_shuffle_train,
                                  if_repeat=if_repeat_train)
        
        data_tensors = creat_data_tensors(train_ds, data_tensors,
                                          train_f_with_l, args,
                                          batch_size=args.batch_size,
                                          prefix='train')
    
    return data_tensors, args
