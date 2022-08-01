from input_pipeline import csv_reader_dataset, get_data_files_from_folder
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import matplotlib
matplotlib.use('Agg')
import os
import scipy
import seaborn as sns
sns.set(style="darkgrid")


random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

data_path_general = "/home/epilepsy-data/data/PPS-rats-from-Sebastian/"
root_logdir = '/home/farahat/Documents/my_logs/final_2/'
batch_size = 512
models = sorted([f for f in os.listdir(root_logdir)])
z_dim = 128

for model_name in models[:]:
    print('working on: '+model_name)

    animal = model_name[40:]
    if "326" in animal:
        data_path = data_path_general + 'Control-Rats/'
    else:
        data_path = data_path_general + 'PPS-Rats/'
    animal_path = os.path.join(data_path, animal, animal)

    run_logdir = root_logdir + model_name
    output_directory = run_logdir +  '/stats/'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # loading the data 
    # test animal data epileptogenesis period
    epg_files = get_data_files_from_folder(animal_path+'/EPG/', train_valid_split=False)
    # test animal data baseline period
    valid_files = get_data_files_from_folder(animal_path+'/BL/', train_valid_split=False)
    # train data pooled from all other animals
    trained_files_filename = [i for i in os.listdir(run_logdir) if 'picked_train' in i][0]
    with open(run_logdir + '/' + trained_files_filename) as f:
        train_files = f.readlines()
        train_files = [i.strip() for i in train_files]


    epg_set = csv_reader_dataset(epg_files, batch_size=batch_size, shuffle=False)
    valid_set = csv_reader_dataset(valid_files, batch_size=batch_size, shuffle=False)
    train_set = csv_reader_dataset(train_files, batch_size=batch_size, shuffle=False)

    # loading the model parts
    encoder = tf.keras.models.load_model(run_logdir+'/encoder.h5')
    decoder = tf.keras.models.load_model(run_logdir+'/decoder.h5')


    def compute_batch_distance(z):
        '''
        paramter z: a matrix of size (batch_size * z_dim)
        return : the distance of vectors z to the origin of the prior distribution
        '''
        distance = []
        for i in range(z.shape[0]):
            distance.append(scipy.spatial.distance.euclidean(z[i].numpy(),np.zeros(z_dim)))
        return np.array(distance)


    def compute_distros(dataset, directory):
        '''
        compute the performance metrics (reconstruction error and distance to the origin of the prior distribution) for all one-second segments in the dataset
        '''
        if not os.path.exists(directory):
            os.mkdir(directory)
        errors = np.array([])
        distances = np.array([])
        z_all = np.zeros(z_dim)

        for i, batch in enumerate(dataset):
            z = encoder(batch)
            z_all = np.vstack((z_all,z.numpy()))

            x_hat = decoder(z)

            loss = np.square(batch-x_hat)[:,:,0]
            error = np.mean(loss, axis=1).ravel()
            errors = np.concatenate((errors,error),axis=0)

            distance = compute_batch_distance(z)
            distances = np.concatenate((distances,distance),axis=0)

            if (i+1) % 10 == 0:
                print('finished: '+str(i)+' batches')
        np.save(directory+'/whole_segment_errors.npy', errors)
        np.save(directory+'/distances.npy', distances)
        np.save(directory+'/z.npy', z_all[1:,:])       
    
# run the funstion for all datasets (train, test animal baseline and test animal epileptogenesis)
compute_distros(train_set, output_directory+'train_data')
compute_distros(valid_set, output_directory+'valid_data')
compute_distros(epg_set, output_directory+'epg_data')

