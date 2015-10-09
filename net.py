# -*- coding: utf-8 -*-

import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
import theano.tensor as T

import numpy as np

from layers import Weight, DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, MaxoutLayer


def cosine(x, y, epsilon=np.array(1e-6).astype(np.float32)):
    norm_x = T.sqrt(T.sum(x ** 2, 1)) + epsilon
    norm_y = T.sqrt(T.sum(y ** 2, 1)) + epsilon
    return T.sum(x * y, 1) / (norm_x * norm_y)


class AlexNet(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        n_images = config['n_images']

        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        xquery = T.ftensor4('xquery') # Trying to find colour variant of this image
        xp = T.ftensor4('xp') # Correct colour variant image
        xns = [] # Non-variant images
        for i in xrange(n_images-1):
            xns.append( T.ftensor4('xn'+str(i+1)) )
        rands = []
        for i in xrange(n_images+1):
                rands.append( T.fvector('rand'+str(i)) )
        layers, params, weight_types = [], [], []

        print '... building the model'

        # Get the representations of all input images
        query_repr, query_layers, query_params, query_weight_types = \
                self.image_repr(xquery, rands[0], config)
        layers += query_layers
        params += query_params
        weight_types += query_weight_types

        p_repr, p_layers, p_params, p_weight_types = \
                self.image_repr(xp, rands[1], config)
        layers += p_layers
        params += p_params
        weight_types += p_weight_types

        n_reprs = []
        for i in xrange(n_images-1):
            n_repr, n_layers, n_params, n_weight_types = \
                    self.image_repr(xns[i], rands[i+2], config)
            n_reprs.append( n_repr )
            layers += n_layers
            params += n_params
            weight_types += n_weight_types

        # Compute cosine distance from query image to target images
        sims_ = []
        sims_.append( cosine(query_repr.output,
                                 p_repr.output ).dimshuffle(0,'x') )
        for i in xrange(n_images-1):
            sims_.append( cosine(query_repr.output,
                                 n_reprs[i].output ).dimshuffle(0,'x') )
        sims = T.concatenate(sims_, axis=1)
        #sims = T.concatenate([ sims[:,1].dimshuffle(0,'x'), sims[:,0].dimshuffle(0,'x') ], axis=1)

        # Temp: Permute location of correct colour variant, to check that improvements are real
        #rng = T.shared_randomstreams.RandomStreams(12345)
        #perm = rng.permutation(size=(sims.shape[0],), n=2)
        #sims2 = T.concatenate([ sims[T.arange(sims.shape[0]),perm[:,0]].dimshuffle(0,'x'),
        #                        sims[T.arange(sims.shape[0]),perm[:,1]].dimshuffle(0,'x') ], axis=1)
        #index_of_variant = T.argmin(perm, axis=1)

        # Compute probabilities
        p_y_given_x = T.nnet.softmax(sims)
        cost = -T.mean(T.log(p_y_given_x[0, :]))

        y_pred = T.argmax(p_y_given_x, axis=1)
        errors = T.neq(y_pred, 0) # index_of_variant) # 0)

        # #################### NETWORK BUILT #######################

        self.testfunc = query_repr.output.shape # sims # errors # T.extra_ops.bincount(y_pred)

        self.cost = cost
        self.errors = T.mean(errors)
        self.errors_top_5 = None
        self.xquery = xquery
        self.xp = xp
        self.xns = xns
        self.rands = rands
        self.layers = layers
        self.params = params
        self.weight_types = weight_types
        self.batch_size = batch_size
        self.n_images = n_images

    def image_repr(self, x, rand, config):
        batch_size = config['batch_size']
        flag_datalayer = config['use_data_layer']
        lib_conv = config['lib_conv']

        layers = []
        params = []
        weight_types = []
        
        if flag_datalayer:
            data_layer = DataLayer(input=x, image_shape=(3, 256, 256,
                                                         batch_size),
                                   cropsize=227, rand=rand, mirror=True,
                                   flag_rand=config['rand_crop'])

            layer1_input = data_layer.output
        else:
            layer1_input = x

        convpool_layer1 = ConvPoolLayer(input=layer1_input,
                                        image_shape=(3, 227, 227, batch_size), 
                                        filter_shape=(3, 11, 11, 96), 
                                        convstride=4, padsize=0, group=1, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.0, lrn=True,
                                        lib_conv=lib_conv,
                                        )
        layers.append(convpool_layer1)
        params += convpool_layer1.params
        weight_types += convpool_layer1.weight_type

        convpool_layer2 = ConvPoolLayer(input=convpool_layer1.output,
                                        image_shape=(96, 27, 27, batch_size),
                                        filter_shape=(96, 5, 5, 256), 
                                        convstride=1, padsize=2, group=2, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.1, lrn=True,
                                        lib_conv=lib_conv,
                                        )
        layers.append(convpool_layer2)
        params += convpool_layer2.params
        weight_types += convpool_layer2.weight_type

        convpool_layer3 = ConvPoolLayer(input=convpool_layer2.output,
                                        image_shape=(256, 13, 13, batch_size),
                                        filter_shape=(256, 3, 3, 384), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=0, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        layers.append(convpool_layer3)
        params += convpool_layer3.params
        weight_types += convpool_layer3.weight_type

        convpool_layer4 = ConvPoolLayer(input=convpool_layer3.output,
                                        image_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 384), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=1, poolstride=0, 
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        layers.append(convpool_layer4)
        params += convpool_layer4.params
        weight_types += convpool_layer4.weight_type
        
        convpool_layer5 = ConvPoolLayer(input=convpool_layer4.output,
                                        image_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 256), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        layers.append(convpool_layer5)
        params += convpool_layer5.params
        weight_types += convpool_layer5.weight_type

        fc_layer6_input = T.flatten(
            convpool_layer5.output.dimshuffle(3, 0, 1, 2), 2)
        fc_layer6 = MaxoutLayer(input=fc_layer6_input, n_in=9216, n_out=4096)
        layers.append(fc_layer6)
        params += fc_layer6.params
        weight_types += fc_layer6.weight_type

        dropout_layer6 = DropoutLayer(fc_layer6.output, n_in=4096, n_out=4096)

        fc_layer7 = MaxoutLayer(input=dropout_layer6.output, n_in=4096, n_out=4096)
        layers.append(fc_layer7)
        params += fc_layer7.params
        weight_types += fc_layer7.weight_type

        #dropout_layer7 = DropoutLayer(fc_layer7.output, n_in=4096, n_out=4096)
    
        # Rename weight types so that weights can be shared
        new_weight_types = []
        counter_W = 0
        counter_b = 0
        for w in weight_types:
            if w == 'W':
                new_weight_types.append('W'+str(counter_W))
                counter_W += 1
            elif w == 'b':
                new_weight_types.append('b'+str(counter_b))
                counter_b += 1
        weight_types = new_weight_types

        return fc_layer7, layers, params, weight_types


def compile_models(model, config, flag_top_5=False):

    xquery = model.xquery
    xp = model.xp
    xns = model.xns
    rands = model.rands
    weight_types = model.weight_types

    cost = model.cost
    params = model.params
    errors = model.errors
    #errors_top_5 = model.errors_top_5
    batch_size = model.batch_size
    n_images = model.n_images

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

    shared_xquery = theano.shared(np.zeros((3, raw_size, raw_size,
                                       batch_size),
                                      dtype=theano.config.floatX),
                             borrow=True)
    shared_xp = theano.shared(np.zeros((3, raw_size, raw_size,
                                       batch_size),
                                      dtype=theano.config.floatX),
                             borrow=True)
    shared_xns = []
    for i in xrange(len(xns)):
        shared_xn = theano.shared(np.zeros((3, raw_size, raw_size,
                                           batch_size),
                                          dtype=theano.config.floatX),
                                 borrow=True)
        shared_xns.append( shared_xn )
    
    rand_arrs = []
    for i in xrange(n_images+1):
        rand_arr = theano.shared(np.zeros(3, dtype=theano.config.floatX),
                                 borrow=True)
        rand_arrs.append( rand_arr )

    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]

    assert len(weight_types) == len(params)

    # Shared weights between all image networks
    iter_indexes = []
    for i in xrange(20):
        W_indexes = []
        b_indexes = []
        for j in xrange(len(weight_types)):
            weight_type = weight_types[j]
            if weight_type == 'W'+str(i):
                W_indexes.append(j)
            elif weight_type == 'b'+str(i):
                b_indexes.append(j)
        
        if len(W_indexes)>0:
            iter_indexes.append(W_indexes)
        if len(b_indexes)>0:
            iter_indexes.append(b_indexes)
        if len(W_indexes)==0 and len(b_indexes)==0:
            break
    
    for indexes in iter_indexes:
        index_i = indexes[0]

        weight_type = weight_types[index_i][0]
        param_i = params[index_i]
        grad_i = grads[index_i]
        vel_i = vels[index_i]

        change_i = 0
        if config['use_momentum']:
            if weight_type ==  'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")
            
            if config['use_nesterov_momentum']:
                change_i = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
            else:
                change_i = mu * vel_i - real_lr * real_grad

        else:
            if weight_type ==  'W':
                change_i = - lr * grad_i - eta * lr * param_i
            elif weight_type == 'b':
                change_i = - 2 * lr * grad_i
            else:
                raise TypeError("Weight Type Error")

        newval = param_i + change_i

        for index in indexes:
            param = params[index]
            updates.append((param, newval))
            if config['use_momentum']:
                vel = vels[index]
                updates.append((vel, change_i))

    #if config['use_momentum']:
    #    for param_i, grad_i, vel_i, weight_type in \
    #            zip(params, grads, vels, weight_types):

    #        if weight_type == 'W':
    #            real_grad = grad_i + eta * param_i
    #            real_lr = lr
    #        elif weight_type == 'b':
    #            real_grad = grad_i
    #            real_lr = 2. * lr
    #        else:
    #            raise TypeError("Weight Type Error")

    #        if config['use_nesterov_momentum']:
    #            vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
    #        else:
    #            vel_i_next = mu * vel_i - real_lr * real_grad

    #        updates.append((vel_i, vel_i_next))
    #        updates.append((param_i, param_i + vel_i_next))
    #else:
    #    for param_i, grad_i, weight_type in zip(params, grads, weight_types):
    #        #weight_type = weight_type[0]
    #        if weight_type == 'W':
    #            updates.append((param_i,
    #                            param_i - lr * grad_i - eta * lr * param_i))
    #        elif weight_type == 'b':
    #            updates.append((param_i, param_i - 2 * lr * grad_i))
    #        else:
    #            continue
    #            #raise TypeError("Weight Type Error")
        
    # Define Theano Functions
    givens = []
    givens.append((lr, learning_rate))
    givens.append((xquery, shared_xquery))
    givens.append((xp, shared_xp))
    for i in xrange(len(xns)):
        givens.append((xns[i], shared_xns[i]))
    for i in xrange(len(rands)):
        givens.append((rands[i], rand_arrs[i]))

    train_model = theano.function([], cost, updates=updates,
                                  givens=givens)

    validate_outputs = [cost, errors]
    #if flag_top_5:
    #    validate_outputs.append(errors_top_5)

    validate_model = theano.function([], validate_outputs, givens=givens)

    train_error = theano.function([], errors, givens=givens[1:])

    if model.testfunc is not None:
        testfunc = theano.function([], model.testfunc, givens=givens)
    else:
        testfunc = None

    #
    # Metrics that can be logged to understand cnn better:
    #
    # Variance & mean of weight matrices at each layer
    # Norm of weight matrices along each of their dimensions
    # Mean & variance of intermediate representations after each layer
    #  - Also, mean & variance per class label
    # Mean, variance, norm of gradient
    #  - norm of gradient should not exceed 5 or 15
    # Ratio between the update norm and weight norm -> should be around 0.001
    #
    #

    return (train_model, validate_model, train_error, learning_rate,
        shared_xquery, shared_xp, shared_xns, rand_arrs, vels, testfunc)
