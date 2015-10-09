import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
import theano.tensor as T

import numpy as np

from layers import DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer


class AlexNet(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        batch_size = config['batch_size']
        flag_datalayer = config['use_data_layer']
        lib_conv = config['lib_conv']

        layers = []
        params = []
        weight_types = []

        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        x1 = T.ftensor4('x1')
        x2 = T.ftensor4('x2')
        y = T.lvector('y') # The ground truth to be compared with will go here
        rand1 = T.fvector('rand1')
        rand2 = T.fvector('rand2')

        print '... building the model'

        if flag_datalayer:
            data_layerA = DataLayer(input=x1, image_shape=(3, 256, 256,
                                                         batch_size),
                                   cropsize=227, rand=rand, mirror=True,
                                   flag_rand=config['rand_crop'])

            layer1A_input = data_layerA.output
        else:
            layer1A_input = x1

        if flag_datalayer:
            data_layerB = DataLayer(input=x2, image_shape=(3, 256, 256,
                                                         batch_size),
                                   cropsize=227, rand=rand, mirror=True,
                                   flag_rand=config['rand_crop'])

            layer1B_input = data_layerB.output
        else:
            layer1B_input = x2

        fc_layer2_input = T.concatenate(
            (T.flatten(layer1A_input.dimshuffle(3, 0, 1, 2), 2),
            T.flatten(layer1B_input.dimshuffle(3, 0, 1, 2), 2)), axis=1 )
        fc_layer2 = FCLayer(input=fc_layer2_input, n_in=154587*2, n_out=4096)
        layers.append(fc_layer2)
        params += fc_layer2.params
        weight_types += fc_layer2.weight_type

        dropout_layer2 = DropoutLayer(fc_layer2.output, n_in=4096, n_out=4096)

        fc_layer3 = FCLayer(input=dropout_layer2.output, n_in=4096, n_out=4096)
        layers.append(fc_layer3)
        params += fc_layer3.params
        weight_types += fc_layer3.weight_type

        dropout_layer3 = DropoutLayer(fc_layer3.output, n_in=4096, n_out=4096)

        # Final softmax layer
        softmax_layer3 = SoftmaxLayer(
            input=dropout_layer3.output, n_in=4096, n_out=2) # Only a single binary output is required!
        layers.append(softmax_layer3)
        params += softmax_layer3.params
        weight_types += softmax_layer3.weight_type

        # #################### NETWORK BUILT #######################

        self.cost = softmax_layer3.negative_log_likelihood(y)
        self.errors = softmax_layer3.errors(y)
        self.errors_top_5 = softmax_layer3.errors_top_x(y, 5)
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.rand1 = rand1
        self.rand2 = rand2
        self.layers = layers
        self.params = params
        self.weight_types = weight_types
        self.batch_size = batch_size

def compile_models(model, config, flag_top_5=False):

    x1 = model.x1
    x2 = model.x2
    y = model.y
    rand1 = model.rand1
    rand2 = model.rand2
    weight_types = model.weight_types

    cost = model.cost
    params = model.params
    errors = model.errors
    errors_top_5 = model.errors_top_5
    batch_size = model.batch_size

    mu = config['momentum']
    eta = config['weight_decay']

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []

    learning_rate = theano.shared(np.float32(config['learning_rate']))
    lr = T.scalar('lr')  # symbolic learning rate

    if config['use_data_layer']:
        raw_size = 256
    else:
        raw_size = 227

    shared_x1 = theano.shared(np.zeros((3, raw_size, raw_size,
                                       batch_size),
                                      dtype=theano.config.floatX),
                             borrow=True)
    shared_x2 = theano.shared(np.zeros((3, raw_size, raw_size,
                                       batch_size),
                                      dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.zeros((batch_size,), dtype=int),
                             borrow=True)

    rand1_arr = theano.shared(np.zeros(3, dtype=theano.config.floatX),
                             borrow=True)
    rand2_arr = theano.shared(np.zeros(3, dtype=theano.config.floatX),
                             borrow=True)

    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]

    if config['use_momentum']:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in \
                zip(params, grads, vels, weight_types):

            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")

            if config['use_nesterov_momentum']:
                vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
            else:
                vel_i_next = mu * vel_i - real_lr * real_grad

            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i + vel_i_next))

    else:
        for param_i, grad_i, weight_type in zip(params, grads, weight_types):
            if weight_type == 'W':
                updates.append((param_i,
                                param_i - lr * grad_i - eta * lr * param_i))
            elif weight_type == 'b':
                updates.append((param_i, param_i - 2 * lr * grad_i))
            else:
                raise TypeError("Weight Type Error")

    # Define Theano Functions

    train_model = theano.function([], cost, updates=updates,
                                  givens=[(x1, shared_x1), (x2, shared_x2),
                                          (y, shared_y),
                                          (lr, learning_rate),
                                          (rand1, rand1_arr), (rand2, rand2_arr)])

    validate_outputs = [cost, errors]
    if flag_top_5:
        validate_outputs.append(errors_top_5)

    validate_model = theano.function([], validate_outputs,
                                     givens=[(x1, shared_x1), (x2, shared_x2), (y, shared_y),
                                             (rand1, rand1_arr), (rand2, rand2_arr)])

    train_error = theano.function(
        [], errors, givens=[(x1, shared_x1), (x2, shared_x2), (y, shared_y), (rand1, rand1_arr), (rand2, rand2_arr)])

    return (train_model, validate_model, train_error,
            learning_rate, shared_x1, shared_x2, shared_y, rand1_arr, rand2_arr, vels)
