import datetime
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import tensorflow as tf
from deep_utils import ECA 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D, Conv2D 
from tensorflow.keras.layers import UpSampling2D, Cropping2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation


def Model_multi(input_shape, out_seq_length, era_var):
    '''
    - Using multi input (RDR+ERA5)
    - Channel numbers of RDR and ERA are controlled manually.
    '''
    ksize1 = 5   # kernel size for outermost layer
    ksize2 = 3   # kernel size for mid- to innermost layer
    strides = 2
    activation1 = 'relu'     # activation for output layer
    activation2 = 'relu'     # activation for ConvLSTM2D

# Set Subfilter for the first conv2D layer
# The number of Channels:  48 for RDR
#                          12 for each ERA variables
    nf0_rdr = 48    # output features from first CONV2D layer for RDR 
    nf0_era = 12    # output features from first CONV2D layer for each ERA5 variable
    
    nf0_all = nf0_rdr+nf0_era*len(era_var)
    nfilter = [ nf0_all, nf0_all*2, nf0_all*4, 16 ]  # the number of filters for each layer
#   if nf0_rdr = 48, nf0_era = 12,  nfilter = [ 60, 120, 240, 16 ]  

# DIV & VOR have (+), (-) values.
# In this case, we devide values depending on (+),(-) signs and treat as separate channels.
    nch = 1+len(era_var)
    if "DIV925" in era_var:
        nch = nch+1
    if "VOR925" in era_var:
        nch = nch+1

    subfilter = np.zeros(nch, dtype=int)
    subfilter[0] = nf0_rdr
    subfilter[1:] = (nfilter[0]-nf0_rdr)/(nch-1)


    print('== Encoding part ==')
    print(f'++ input_shape: {input_shape}')
    inputs_sub = [ f'sub_input{i}' for i in range(nch) ]
    print(f'++ inputs_sub = {inputs_sub}')
    net_sub = [ f'sub_net{i}' for i in range(nch) ]
    sub_model = [ f'sub_model{i}' for i in range(nch) ]
    for i in range(nch):
        inputs_sub[i] = tf.keras.Input(shape=([input_shape[0],input_shape[1],input_shape[2],1]))
        print(f'++ inputs_sub[i]: {inputs_sub[i]}')
        net_sub[i] = TimeDistributed(Conv2D(filters=subfilter[i], kernel_size=ksize1, activation=activation2,
                                            strides=strides, padding='same'))(inputs_sub[i])
        sub_model[i] = Model(inputs=inputs_sub[i], outputs=net_sub[i])
        print(f' *Enc_Conv2D_1_sub{i}: {sub_model[i].output.shape}')

    net = Concatenate(axis=4)([sub_model[i].output for i in range(nch)])
    print(f'++ After concatenate: {net}')
    print(f' * Enc_Conv2D_1: {net.shape}')

    net = TimeDistributed(ECA())(net)
    print(f' * Enc_ECA_1: {net.shape}')
    net, e_h_1, e_c_1 = ConvLSTM2D(filters=nfilter[0], kernel_size=(ksize1,ksize1), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net)
    e_s_1 = [e_h_1, e_c_1]
    print(f' * Enc_ConvLSTM2D_1: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[1], kernel_size=ksize2, activation=activation2,
                                 strides=strides, padding='same'))(net)
    print(f' * Enc_Conv2D_2: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Enc_ECA_2: {net.shape}')
    net, e_h_2, e_c_2 = ConvLSTM2D(filters=nfilter[1], kernel_size=(ksize2,ksize2), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net)
    e_s_2 = [e_h_2, e_c_2]
    print(f' * Enc_ConvLSTM2D_2: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[2], kernel_size=ksize2, activation=activation2,
                                 strides=strides, padding='same'))(net)
    print(f' * Enc_Conv2D_3: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Enc_ECA_3: {net.shape}')
    net, e_h_3, e_c_3 = ConvLSTM2D(filters=nfilter[2], kernel_size=(ksize2,ksize2), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net)
    e_s_3 = [e_h_3, e_c_3]
    print(f' * Enc_ConvLSTM2D_3: {net.shape}')


    tensor_shape = net.get_shape().as_list()
    print(f' - tensor_shape: {tensor_shape}')
    zero_input = tf.zeros([tf.shape(net)[0], out_seq_length, tensor_shape[2], tensor_shape[3], tensor_shape[4]], tf.float32)
    print(f' - zero_input shape: {zero_input.shape}')

    print('')
    print('== Decoding part ==')
    net, d_h_1, d_c_1 = ConvLSTM2D(filters=nfilter[2], kernel_size=(ksize2,ksize2), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(zero_input, initial_state=e_s_3)
    print(f' * Dec_ConvLSTM2D_1: {net.shape}')
    net = TimeDistributed(UpSampling2D(size=2))(net)
    print(f' * Dec_UpSampling2D_1: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Dec_ECA_1: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[1], kernel_size=ksize2, activation=activation2, padding='same'))(net)
    print(f' * Dec_Conv2D_1: {net.shape}')
    net, d_h_2, d_c_2 = ConvLSTM2D(filters=nfilter[1], kernel_size=(ksize2,ksize2), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net, initial_state=e_s_2)
    print(f' * Dec_ConvLSTM2D_2: {net.shape}')
    net = TimeDistributed(UpSampling2D(size=2))(net)
    print(f' * Dec_UpSampling2D_2: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Dec_ECA_2: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[0], kernel_size=ksize2, activation=activation2, padding='same'))(net)
    print(f' * Dec_Conv2D_2: {net.shape}')
    net, d_h_3, d_c_3 = ConvLSTM2D(filters=nfilter[0], kernel_size=(ksize1,ksize1), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net, initial_state=e_s_1)
    print(f' * Dec_ConvLSTM2D_3: {net.shape}')
    net = TimeDistributed(UpSampling2D(size=2))(net)
    print(f' * Dec_UpSampling2D_3: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Dec_ECA_3: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[3], kernel_size=ksize1, activation=activation2, padding='same'))(net)
    print(f' * Dec_Conv2D_3: {net.shape}')

    outputs = TimeDistributed(Conv2D(filters=1, kernel_size=1, activation=activation1, padding='same'))(net) 
    print(f' == output: {outputs.shape}')

    model = Model([sub_model[i].input for i in range(nch)], outputs)
    return model






def Model_def(inputs, out_seq_length):
    ksize1 = 5   # kernel size for outermost layer
    ksize2 = 3   # kernel size for mid- to innermost layer
    strides = 2
    activation1 = 'relu'     # activation for output layer
    activation2 = 'relu'     # activation for ConvLSTM2D

    nfilter = [ 48, 96, 192, 16 ]    # for [def01_ECA]
 #   nfilter = [ 60, 60*2, 60*4, 16 ]  # for sensitivity exp 
                                       #    model capacity is same as multi-input 
                                       #    but, use RDR only.

    print('== Encoding part ==')
    net = TimeDistributed(Conv2D(filters=nfilter[0], kernel_size=ksize1, activation=activation2,
                                 strides=strides, padding='same'))(inputs)
    print(f' * Enc_Conv2D_1: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Enc_ECA_1: {net.shape}')
    net, e_h_1, e_c_1 = ConvLSTM2D(filters=nfilter[0], kernel_size=(ksize1,ksize1), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net)
    e_s_1 = [e_h_1, e_c_1]
    print(f' * Enc_ConvLSTM2D_1: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[1], kernel_size=ksize2, activation=activation2,
                                 strides=strides, padding='same'))(net)
    print(f' * Enc_Conv2D_2: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Enc_ECA_2: {net.shape}')
    net, e_h_2, e_c_2 = ConvLSTM2D(filters=nfilter[1], kernel_size=(ksize2,ksize2), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net)
    e_s_2 = [e_h_2, e_c_2]
    print(f' * Enc_ConvLSTM2D_2: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[2], kernel_size=ksize2, activation=activation2,
                                 strides=strides, padding='same'))(net)
    print(f' * Enc_Conv2D_3: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Enc_ECA_3: {net.shape}')
    net, e_h_3, e_c_3 = ConvLSTM2D(filters=nfilter[2], kernel_size=(ksize2,ksize2), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net)
    e_s_3 = [e_h_3, e_c_3]
    print(f' * Enc_ConvLSTM2D_3: {net.shape}')


    tensor_shape = net.get_shape().as_list()
    print(f' - tensor_shape: {tensor_shape}')
    zero_input = tf.zeros([tf.shape(net)[0], out_seq_length, tensor_shape[2], tensor_shape[3], tensor_shape[4]], tf.float32)
    print(f' - zero_input shape: {zero_input.shape}')

    print('')
    print('== Decoding part ==')
    net, d_h_1, d_c_1 = ConvLSTM2D(filters=nfilter[2], kernel_size=(ksize2,ksize2), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(zero_input, initial_state=e_s_3)
    print(f' * Dec_ConvLSTM2D_1: {net.shape}')
    net = TimeDistributed(UpSampling2D(size=2))(net)
    print(f' * Dec_UpSampling2D_1: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Dec_ECA_1: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[1], kernel_size=ksize2, activation=activation2, padding='same'))(net)
    print(f' * Dec_Conv2D_1: {net.shape}')
    net, d_h_2, d_c_2 = ConvLSTM2D(filters=nfilter[1], kernel_size=(ksize2,ksize2), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net, initial_state=e_s_2)
    print(f' * Dec_ConvLSTM2D_2: {net.shape}')
    net = TimeDistributed(UpSampling2D(size=2))(net)
    print(f' * Dec_UpSampling2D_2: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Dec_ECA_2: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[0], kernel_size=ksize2, activation=activation2, padding='same'))(net)
    print(f' * Dec_Conv2D_2: {net.shape}')
    net, d_h_3, d_c_3 = ConvLSTM2D(filters=nfilter[0], kernel_size=(ksize1,ksize1), activation=activation2,
                                   padding='same', return_sequences=True, data_format="channels_last",
                                   return_state=True)(net, initial_state=e_s_1)
    print(f' * Dec_ConvLSTM2D_3: {net.shape}')
    net = TimeDistributed(UpSampling2D(size=2))(net)
    print(f' * Dec_UpSampling2D_3: {net.shape}')
    net = TimeDistributed(ECA())(net)
    print(f' * Dec_ECA_3: {net.shape}')
    net = TimeDistributed(Conv2D(filters=nfilter[3], kernel_size=ksize1, activation=activation2, padding='same'))(net)
    print(f' * Dec_Conv2D_3: {net.shape}')

    outputs = TimeDistributed(Conv2D(filters=1, kernel_size=1, activation=activation1, padding='same'))(net)
    print(f' == output: {outputs.shape}')

    model = Model(inputs, outputs)
    return model












