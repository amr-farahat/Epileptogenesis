import matplotlib.pyplot as plt
import os
import pdb
import time
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
# tf.enable_eager_execution()

def plot_errors(errors, path, err='reconstruction'):
    fig = plt.figure(figsize=(15,10))
    plt.plot(errors)
        
    if err == 'pdf':
        # plt.ylim(0, 1.4e-20)
        plt.ylabel('PDF')
        # ticks = np.arange(0,errors.shape[0], 720/2)
        # plt.xticks(ticks, np.arange(0,len(ticks)/2, 0.5))        
    if err == 'reconstruction':
        plt.ylim(0, 1500)
        plt.ylabel('Error')
        ticks = np.arange(0,errors.shape[0], 1843200/2)
        plt.xticks(ticks, np.arange(0,len(ticks)/2, 0.5))
        
    
    plt.grid(True)
    plt.xlabel('Time in hours')
    plt.savefig(path)
    plt.close()

def plot_dict_loss(d, run_logdir):
    fig = plt.figure(figsize=(30,20))
    # fig.subplots_adjust(hspace=0.4, wspace=0.2)
    for i, key in enumerate([x for x in list(d.keys()) if not x.startswith('v_')]):
        ax = fig.add_subplot(4, 3, i+1)
        ax.plot(d[key], label=key, linewidth=2, color='blue')

        # ax.plot(d['v_'+key], label='v_'+key, linewidth=1,  linestyle='dashed', color='red')

        # if max(d[key] + d['v_'+key]) > 1:
        #     ax.set_ylim([0, 1])
        ax.legend()
    plt.savefig(run_logdir+'/losses.png')

def plot_loss(history, run_logdir):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.title('Loss')
    plt.plot(loss, label="train")
    plt.plot(val_loss, label="validation")
    plt.legend()
    plt.savefig(run_logdir+'/loss.png')
    plt.close('all')

def plot_latent_space(model, valid_set, run_logdir, epoch):

    codes = []
    for batch in valid_set:
        code = model.predict(batch)
        # code = model.predict(batch)[:,:,0]
        codes.append(code)

    codes = np.array(codes)
    codes_flattened = np.reshape(codes, (codes.shape[0]*codes.shape[1], codes.shape[2]))
    # codes_embedded = TSNE(n_components=2).fit_transform(codes_flattened)

    plt.scatter(codes_flattened[:, 0], codes_flattened[:, 1], s=2)
    plt.savefig(run_logdir+'/latent_space_'+str(epoch)+'.png')
    plt.close()


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
        # reconstruction = model.predict(tf.expand_dims(item[0][0], axis=0))[0,:,0]
        # original_data.append(tf.expand_dims(item[0][0], axis=0)[0,:,0].numpy())
        # pdb.set_trace()
        reconstruction = model.predict(tf.expand_dims(item[0], axis=0))
        original_data.append(item[0].numpy())
        reconstructions.append(reconstruction[0].numpy())
    
    return original_data, reconstructions

def sample_data(model, z_dim, run_logdir, norm_params, std, epoch, no_samples=10):
    # z = tf.random.normal([no_samples, z_dim, 1], mean=0.0, stddev=1.0)

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


def save_results(history, model, valid_set, note, run_logdir, no_samples=6):

    plot_loss(history, run_logdir)

    model.save(run_logdir+'/the_model.h5')

    original_data, reconstructions = predict_validation_samples(model, valid_set, no_samples=no_samples)

    plot_samples(original_data, reconstructions, run_logdir, 0)

    with open(run_logdir+'/notes.txt', 'a') as f:
        f.write(note)




def KnuthMorrisPratt(text, pattern):
    
    '''Yields all starting positions of copies of the pattern in the text.
Calling conventions are similar to string.find, but its arguments can be
lists or iterators, not just strings, it returns all matches, not just
the first one, and it does not need the whole text in memory at once.
Whenever it yields, it will have read the text exactly up to and including
the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos