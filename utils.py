import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
import numpy as np
# tf.enable_eager_execution()



def plot_dict_loss(d, run_logdir):
    fig = plt.figure(figsize=(30,20))
    for i, key in enumerate([x for x in list(d.keys()) if not x.startswith('v_')]):
        ax = fig.add_subplot(4, 3, i+1)
        ax.plot(d[key], label=key, linewidth=2, color='blue')
        ax.legend()
    plt.savefig(run_logdir+'/losses.png')



def get_run_logdir(root_logdir, animal):
    run_id = time.strftime("run_EPG_anomaly_%Y_%m_%d-%H_%M_LOO_%S")+'_'+animal
    path = os.path.join(root_logdir, run_id)
    os.mkdir(path)
    return path

def predict_validation_samples(model, valid_set, no_samples=6):
    
    random_dataset = tf.data.experimental.sample_from_datasets([valid_set])

    original_data = []
    reconstructions = []

    for item in random_dataset.take(no_samples):
        reconstruction = model.predict(tf.expand_dims(item[0], axis=0))
        original_data.append(item[0].numpy())
        reconstructions.append(reconstruction[0].numpy())
    
    return original_data, reconstructions

def sample_data(model, z_dim, run_logdir, norm_params, std, epoch, no_samples=10):

    weights = np.ones(len(norm_params), dtype=np.float64) / len(norm_params)
    mixture_idx = np.random.choice(len(weights), size=no_samples, replace=True, p=weights)
    z = tf.convert_to_tensor([np.random.normal(norm_params[idx], std, size=(z_dim,1)) for idx in mixture_idx], dtype=tf.float32)

    x = model(z).numpy()
    fig = plt.figure(figsize=(20,10))
    for i in range(1, no_samples+1):
        ax = fig.add_subplot(no_samples, 1, i)
        ax.plot(x[i-1], c='black', label='generated_data',  linewidth=2)
        ax.set_yticks([], [])
        if i < no_samples: 
            ax.set_xticks([], [])
    plt.legend(loc='upper right', shadow=True)
    plt.savefig(run_logdir+'/generated_data_'+str(epoch)+'.png')
    plt.close('all')


def plot_samples(original_data, reconstructions, run_logdir, epoch):
    fig = plt.figure(figsize=(20,10))
    for i in range(1, len(original_data)+1):
        ax = fig.add_subplot(len(original_data), 1, i)
        ax.plot(original_data[i-1], c='red', label='original',  linewidth=2)
        ax.plot(reconstructions[i-1], c='black', label='reconstructed',  linewidth=2)
        ax.set_yticks([], [])
        if i < len(original_data): 
            ax.set_xticks([], [])
    plt.legend(loc='upper right', shadow=True)
    plt.savefig(run_logdir+'/valid_samples_plot_'+str(epoch)+'.png')
    plt.close('all')






