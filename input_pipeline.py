
import tensorflow as tf
if tf.__version__ != '2.0.0':
    tf.enable_eager_execution()
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from random import randrange

random_seed = 42
tf.random.set_random_seed(random_seed)
np.random.seed(random_seed)


# list the files

def get_data_files(path, train_percentage=0.8):
    files = [path+f for f in listdir(path) if isfile(join(path, f))]
    train_files = files[:round(len(files)*train_percentage)]
    valid_files = files[round(len(files)*train_percentage):]

    return train_files, valid_files


def compute_data_parameters(files, dims=2561):

    sums = np.zeros((dims,))
    counter = 0
    for file in files:
        data = np.genfromtxt(file, delimiter=',')[:,1:]
        counter += data.shape[0]
        sums += data.sum(axis=0)
    mean = sums/counter

    sums = np.zeros((dims,))
    for file in files:
        data = np.genfromtxt(file, delimiter=',')[:,1:]
        sums += ((data-mean)**2).sum(axis=0)
    std=np.sqrt(sums/(counter-1))

    return mean, std

  


def read(line):
    n_inputs = 2561
    defs = [tf.constant([], dtype=tf.string)]+ [0.]*n_inputs
    fields = tf.io.decode_csv(line, record_defaults=defs)[1:-1]
    x = tf.stack(fields)
    x = tf.expand_dims(x,1)
    return x

def csv_reader_dataset(filepaths, mean, std, n_readers=5, n_read_threads=None, shuffle_buffer_size=10000, 
                       n_parse_threads=tf.data.experimental.AUTOTUNE, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1), 
        cycle_length=n_readers, 
        num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(read, num_parallel_calls=n_parse_threads)
    dataset = dataset.map(lambda x: (x-mean) / (std + np.finfo(np.float32).eps), num_parallel_calls=n_parse_threads)
    
    # dataset = dataset.map(lambda x: (x,x) , num_parallel_calls=n_parse_threads)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(1)
