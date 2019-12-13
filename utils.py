import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import os
import pdb
import time
import tensorflow as tf
if tf.__version__ != '2.0.0':
    tf.enable_eager_execution()


def plot_dict_loss(d, run_logdir):
    fig = plt.figure(figsize=(20,6))
    # fig.subplots_adjust(hspace=0.4, wspace=0.2)
    for i, key in enumerate([x for x in list(d.keys()) if not x.startswith('v_')]):
        ax = fig.add_subplot(2, 4, i+1)
        ax.plot(d[key], label=key, linewidth=2, color='blue')
        ax.plot(d['v_'+key], label='v_'+key, linewidth=2, color='red')
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
    plt.close()


def get_run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
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
    plt.close()


def save_results(history, model, valid_set, note, run_logdir, no_samples=6):

    plot_loss(history, run_logdir)

    model.save(run_logdir+'/the_model.h5')

    original_data, reconstructions = predict_validation_samples(model, valid_set, no_samples=no_samples)

    plot_samples(original_data, reconstructions, run_logdir, 0)

    with open(run_logdir+'/notes.txt', 'a') as f:
        f.write(note)



