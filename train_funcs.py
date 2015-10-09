import glob
import time
import os

import numpy as np

import hickle as hkl

from proc_load import crop_and_mirror 

def proc_configs(config):
    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])
        print "Create folder: " + config['weights_dir']

    return config


def unpack_configs(config, ext_data='.hkl', ext_label='.npy'):
    flag_para_load = config['para_load']

    # Load Training/Validation Filenames and Labels
    train_folder = config['train_folder']
    val_folder = config['val_folder']
    label_folder = config['label_folder']

    trainA_filenames = sorted(glob.glob(train_folder + '/*A' + ext_data))
    valA_filenames = sorted(glob.glob(val_folder + '/*A' + ext_data))
    trainB_filenames = sorted(glob.glob(train_folder + '/*B' + ext_data))
    valB_filenames = sorted(glob.glob(val_folder + '/*B' + ext_data))

    #train_labels = np.load(label_folder + 'train_labels' + ext_label)
    #val_labels = np.load(label_folder + 'val_labels' + ext_label)
    img_mean = np.load(config['mean_file'])
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32')
    return (flag_para_load, 
            trainA_filenames, valA_filenames, trainB_filenames, valB_filenames,
            img_mean)


def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt Learning Rate
    if config['lr_policy'] == 'step':
        if epoch == config['lr_step'][step_idx]:
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            step_idx += 1
            if step_idx >= len(config['lr_step']):
                step_idx = 0  # prevent index out of range error
            print 'Learning rate changed to:', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3] - val_record[-1] <
                            config['lr_adapt_threshold']):
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            print 'Learning rate changed to::', learning_rate.get_value()

    return step_idx


def get_val_error_loss(rand_arrs, shared_xquery, shared_xp, shared_xns,
                       filenamesA, filenamesB,
                       flag_para_load, img_mean,
                       batch_size, validate_model,
                       send_queue=None, recv_queue=None,
                       flag_top_5=False):

    validation_losses = []
    validation_errors = []
    if flag_top_5:
        validation_errors_top_5 = []

    n_val_batches = len(filenamesA)

    if flag_para_load:
        # send the initial message to load data, before each epoch
        send_queue.put(str(filenamesA[0]))
        send_queue.put(str(filenamesB[0]))
        send_queue.put(np.float32([0.5, 0.5, 0]))
        send_queue.put('calc_finished')

    for val_index in xrange(n_val_batches):

        if flag_para_load:
            # load by self or the other process

            # wait for the copying to finish
            msg = recv_queue.get()
            assert msg == 'copy_finished'

            if val_index + 1 < n_val_batches:
                name_to_read1 = str(filenamesA[val_index + 1])
                send_queue.put(name_to_read1)
                send_queue.put(np.float32([0.5, 0.5, 0]))

                name_to_read2 = str(filenamesB[val_index + 1])
                send_queue.put(name_to_read2)
                send_queue.put(np.float32([0.5, 0.5, 0]))
        else:
            img1 = hkl.load(str(filenamesA[val_index])) - img_mean            
            img2 = hkl.load(str(filenamesB[val_index])) - img_mean            
            param_rand = [0.5,0.5,0]              
            img1 = crop_and_mirror(img1, param_rand, flag_batch=True)
            img2 = crop_and_mirror(img2, param_rand, flag_batch=True)
            shared_xquery.set_value(img1)
            shared_xp.set_value(img2)
            # set false colour variant matches
            for i in xrange(len(shared_xns)):
                shared_xn = shared_xns[i]
                N = img2.shape[3]
                n_indexes = np.arange(N) + np.random.randint(1,N-5)
                n_indexes %= N
                imgN = img2[:,:,:,n_indexes]
                shared_xn.set_value(imgN)

        if flag_top_5:
            loss, error, error_top_5 = validate_model()
        else:
            loss, error = validate_model()


        if flag_para_load and (val_index + 1 < n_val_batches):
            send_queue.put('calc_finished')

        validation_losses.append(loss)
        validation_errors.append(error)

        if flag_top_5:
            validation_errors_top_5.append(error_top_5)

    this_validation_loss = np.mean(validation_losses)
    this_validation_error = np.mean(validation_errors)

    #if flag_top_5:
    #    this_validation_error_top_5 = np.mean(validation_errors_top_5)
    #    return this_validation_error, this_validation_error_top_5, this_validation_loss
    #else:
    return this_validation_error, this_validation_loss


def get_rand3d():
    tmp_rand = np.float32(np.random.rand(3))
    tmp_rand[2] = round(tmp_rand[2])
    return tmp_rand


def train_model_wrap(train_model, shared_xquery, shared_xp, shared_xns, rand_arrs, img_mean,
                     count, minibatch_index, minibatch_range, batch_size,
                     filenamesA, filenamesB,
                     flag_para_load, flag_batch,
                     send_queue=None, recv_queue=None):

    if flag_para_load:
        # load by self or the other process

        # wait for the copying to finish
        msg = recv_queue.get()
        assert msg == 'copy_finished'

        if count < len(minibatch_range):
            ind_to_read = minibatch_range[count]
            name_to_read1 = str(filenamesA[ind_to_read])
            send_queue.put(name_to_read1)
            send_queue.put(get_rand3d())

            name_to_read2 = str(filenamesB[ind_to_read])
            send_queue.put(name_to_read2)
            send_queue.put(get_rand3d())

    else:
        img1 = hkl.load(str(filenamesA[minibatch_index])) - img_mean
        img2 = hkl.load(str(filenamesB[minibatch_index])) - img_mean
        param_rand1 = get_rand3d()
        param_rand2 = get_rand3d()
        img1 = crop_and_mirror(img1, param_rand1, flag_batch=flag_batch)         
        img2 = crop_and_mirror(img2, param_rand2, flag_batch=flag_batch)         
        shared_xquery.set_value(img1)
        shared_xp.set_value(img2)
        # set false colour variant matches
        for i in xrange(len(shared_xns)):
            shared_xn = shared_xns[i]
            N = img2.shape[3]
            n_indexes = np.arange(N) + np.random.randint(1,N-5)
            n_indexes %= N
            imgN = img2[:,:,:,n_indexes]
            shared_xn.set_value(imgN)

    cost_ij = train_model()

    return cost_ij
