import sys

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from pylearn2.expr.normalize import CrossChannelNormalization

import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState(23455)
# set a fixed number for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights


class Weight(object):

    def __init__(self, w_shape, mean=0, std=0.01, glorot=False):
        super(Weight, self).__init__()

        if glorot:
            if std==0 or len(w_shape) not in [2, 3, 4]:
                raise TypeError("Unexpected weight dimensiosns")
            # Glorot Weight Initialization
            if len(w_shape) == 2: # fc layer weights / maxout layer bias
                n1, n2 = w_shape
                receptive_field_size = 1
            elif len(w_shape) == 3: # maxout layer weights
                n1, n2 = w_shape[1], w_shape[2]
                receptive_field_size = w_shape[0]
            elif len(w_shape) == 4: # conv layer
                n1, n2 = w_shape[0], w_shape[3]
                receptive_field_size = w_shape[1] * w_shape[2]
            gain = np.sqrt(2)
            std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size) )

        if std != 0:
            self.np_values = np.asarray(
                rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
        else:
            self.np_values = np.cast[theano.config.floatX](
                mean * np.ones(w_shape, dtype=theano.config.floatX))

        self.val = theano.shared(value=self.np_values)

    def save_weight(self, dir, name):
        #print 'weight saved: ' + name
        np.save(dir + name + '.npy', self.val.get_value())

    def load_weight(self, dir, name):
        print 'weight loaded: ' + name
        self.np_values = np.load(dir + name + '.npy')
        self.val.set_value(self.np_values)


class DataLayer(object):

    def __init__(self, input, image_shape, cropsize, rand, mirror, flag_rand):
        '''
        The random mirroring and cropping in this function is done for the
        whole batch.
        '''

        # trick for random mirroring
        mirror = input[:, :, ::-1, :]
        input = T.concatenate([input, mirror], axis=0)

        # crop images
        center_margin = (image_shape[2] - cropsize) / 2

        if flag_rand:
            mirror_rand = T.cast(rand[2], 'int32')
            crop_xs = T.cast(rand[0] * center_margin * 2, 'int32')
            crop_ys = T.cast(rand[1] * center_margin * 2, 'int32')
        else:
            mirror_rand = 0
            crop_xs = center_margin
            crop_ys = center_margin

        self.output = input[mirror_rand * 3:(mirror_rand + 1) * 3, :, :, :]
        self.output = self.output[
            :, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :]

        print "data layer with shape_in: " + str(image_shape)


class ConvPoolLayer(object):

    def __init__(self, input, image_shape, filter_shape, convstride, padsize,
                 group, poolsize, poolstride, bias_init, lrn=False,
                 lib_conv='cudnn',
                 ):
        '''
        lib_conv can be cudnn (recommended)or cudaconvnet
        '''

        self.filter_size = filter_shape
        self.convstride = convstride
        self.padsize = padsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.channel = image_shape[0]
        self.lrn = lrn
        self.lib_conv = lib_conv
        assert group in [1, 2]

        self.filter_shape = np.asarray(filter_shape)
        self.image_shape = np.asarray(image_shape)

        if self.lrn:
            self.lrn_func = CrossChannelNormalization()

        if group == 1:
            self.W = Weight(self.filter_shape, glorot=True)
            self.b = Weight(self.filter_shape[3], bias_init, std=0)
        else:
            self.filter_shape[0] = self.filter_shape[0] / 2
            self.filter_shape[3] = self.filter_shape[3] / 2
            self.image_shape[0] = self.image_shape[0] / 2
            self.image_shape[3] = self.image_shape[3] / 2
            self.W0 = Weight(self.filter_shape, glorot=True)
            self.W1 = Weight(self.filter_shape, glorot=True)
            self.b0 = Weight(self.filter_shape[3], bias_init, std=0)
            self.b1 = Weight(self.filter_shape[3], bias_init, std=0)

        if lib_conv == 'cudaconvnet':
            self.conv_op = FilterActs(pad=self.padsize, stride=self.convstride,
                                      partial_sum=1)

            # Conv
            if group == 1:
                contiguous_input = gpu_contiguous(input)
                contiguous_filters = gpu_contiguous(self.W.val)
                conv_out = self.conv_op(contiguous_input, contiguous_filters)
                conv_out = conv_out + self.b.val.dimshuffle(0, 'x', 'x', 'x')
            else:
                contiguous_input0 = gpu_contiguous(
                    input[:self.channel / 2, :, :, :])
                contiguous_filters0 = gpu_contiguous(self.W0.val)
                conv_out0 = self.conv_op(
                    contiguous_input0, contiguous_filters0)
                conv_out0 = conv_out0 + \
                    self.b0.val.dimshuffle(0, 'x', 'x', 'x')

                contiguous_input1 = gpu_contiguous(
                    input[self.channel / 2:, :, :, :])
                contiguous_filters1 = gpu_contiguous(self.W1.val)
                conv_out1 = self.conv_op(
                    contiguous_input1, contiguous_filters1)
                conv_out1 = conv_out1 + \
                    self.b1.val.dimshuffle(0, 'x', 'x', 'x')
                conv_out = T.concatenate([conv_out0, conv_out1], axis=0)

            # ReLu
            self.output = T.maximum(conv_out, 0)
            conv_out = gpu_contiguous(conv_out)

            # Pooling
            if self.poolsize != 1:
                self.pool_op = MaxPool(ds=poolsize, stride=poolstride)
                self.output = self.pool_op(self.output)

        elif lib_conv == 'cudnn':

            input_shuffled = input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
            # in01out to outin01
            # print image_shape_shuffled
            # print filter_shape_shuffled
            if group == 1:
                W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out = dnn.dnn_conv(img=input_shuffled,
                                        kerns=W_shuffled,
                                        subsample=(convstride, convstride),
                                        border_mode=padsize,
                                        )
                conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')
            else:
                W0_shuffled = \
                    self.W0.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out0 = \
                    dnn.dnn_conv(img=input_shuffled[:, :self.channel / 2,
                                                    :, :],
                                 kerns=W0_shuffled,
                                 subsample=(convstride, convstride),
                                 border_mode=padsize,
                                 )
                conv_out0 = conv_out0 + \
                    self.b0.val.dimshuffle('x', 0, 'x', 'x')
                W1_shuffled = \
                    self.W1.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out1 = \
                    dnn.dnn_conv(img=input_shuffled[:, self.channel / 2:,
                                                    :, :],
                                 kerns=W1_shuffled,
                                 subsample=(convstride, convstride),
                                 border_mode=padsize,
                                 )
                conv_out1 = conv_out1 + \
                    self.b1.val.dimshuffle('x', 0, 'x', 'x')
                conv_out = T.concatenate([conv_out0, conv_out1], axis=1)

            # ReLu
            self.output = T.maximum(conv_out, 0)

            # Pooling
            if self.poolsize != 1:
                self.output = dnn.dnn_pool(self.output,
                                           ws=(poolsize, poolsize),
                                           stride=(poolstride, poolstride))

            self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b

        else:
            NotImplementedError("lib_conv can only be cudaconvnet or cudnn")

        # LRN
        if self.lrn:
            # lrn_input = gpu_contiguous(self.output)
            self.output = self.lrn_func(self.output)

        if group == 1:
            self.params = [self.W.val, self.b.val]
            self.weight_type = ['W', 'b']
        else:
            self.params = [self.W0.val, self.b0.val, self.W1.val, self.b1.val]
            self.weight_type = ['W', 'b', 'W', 'b']

        print "conv ({}) layer with shape_in: {}".format(lib_conv,
                                                         str(image_shape))


class FCLayer(object):

    def __init__(self, input, n_in, n_out):

        self.W = Weight((n_in, n_out), std=0.005, glorot=True)
        self.b = Weight(n_out, mean=0.1, std=0)
        self.input = input
        lin_output = T.dot(self.input, self.W.val) + self.b.val
        self.output = T.maximum(lin_output, 0)
        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']
        print 'fc layer with num_in: ' + str(n_in) + ' num_out: ' + str(n_out)

class MaxoutLayer(object):

    def __init__(self, input, n_in, n_out):
        num_pieces = 2 # The number of linear pieces to use in each maxout unit

        self.Wpos = Weight((num_pieces, n_in, n_out), std=0.005, glorot=True)
        self.Wneg = Weight((num_pieces, n_in, n_out), std=0.005, glorot=True)
        self.bpos = Weight((num_pieces, n_out), mean=0.1, std=0.005)
        self.bneg = Weight((num_pieces, n_out), mean=0.1, std=0.005)
        self.input = input
        
        pos_lin_output = T.dot(self.input, self.Wpos.val) + self.bpos.val
        neg_lin_output = T.dot(self.input, self.Wneg.val) + self.bneg.val

        pos_out = T.max(pos_lin_output, axis=0)
        neg_out = T.max(neg_lin_output, axis=0)

        self.output = pos_out - neg_out

        self.params = [self.Wpos.val, self.Wneg.val, self.bpos.val, self.bneg.val]
        self.weight_type = ['W', 'W', 'b', 'b']

        print 'maxout layer with num_in: ' + str(n_in) + ' num_out: ' + str(n_out) + ' num_pieces' + str(num_pieces)

class DropoutLayer(object):
    seed_common = np.random.RandomState(0)  # for deterministic results
    # seed_common = np.random.RandomState()
    layers = []

    def __init__(self, input, n_in, n_out, prob_drop=0.5):

        self.prob_drop = prob_drop
        self.prob_keep = 1.0 - prob_drop
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on

        seed_this = DropoutLayer.seed_common.randint(0, 2**31-1)
        mask_rng = theano.tensor.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=input.shape)

        self.output = \
            self.flag_on * T.cast(self.mask, theano.config.floatX) * input + \
            self.flag_off * self.prob_keep * input

        DropoutLayer.layers.append(self)

        print 'dropout layer with P_drop: ' + str(self.prob_drop)

    @staticmethod
    def SetDropoutOn():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(1.0)

    @staticmethod
    def SetDropoutOff():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(0.0)


class SoftmaxLayer(object):

    def __init__(self, input, n_in, n_out):

        self.W = Weight((n_in, n_out))
        self.b = Weight((n_out,), std=0)

        self.p_y_given_x = T.nnet.softmax(
            T.dot(input, self.W.val) + self.b.val)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        print 'softmax layer with num_in: ' + str(n_in) + \
            ' num_out: ' + str(n_out)

    def negative_log_likelihood(self, y):
        #return -T.mean(T.log(self.p_y_given_x+1e-10)[T.arange(y.shape[0]), y])
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def errors_top_x(self, y, num_top=5):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_pred_top_x = T.argsort(self.p_y_given_x, axis=1)[:, -num_top:]
            y_top_x = y.reshape((y.shape[0], 1)).repeat(num_top, axis=1)
            return T.mean(T.min(T.neq(y_pred_top_x, y_top_x), axis=1))
        else:
            raise NotImplementedError()
