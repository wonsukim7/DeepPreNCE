import datetime
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D, Conv1D, Conv2D
from tensorflow.keras.layers import UpSampling2D, Cropping2D


class get_Data:
    def __init__(self,path,sample_list,syear,eyear,ny,nx):
        self.path = path
        self.syear = syear    
        self.eyear = eyear
        self.ny = ny        # Y-direction : Dimension size of input data
        self.nx = nx        # X-direction : Dimension size of input data
        self.sample_list = sample_list
        self.df = pd.DataFrame()
    
    def getsample(self):
        sample_list = []
        period = list(np.arange(self.syear,self.eyear+1).astype(str))
        df = pd.read_csv(self.sample_list, header=None)
        df = df.astype(str)
        self.df = df[df.iloc[:,0].str[0:4].isin(period)] 
        return self.df

    def exclude_sample(self,sym_exc,eym_exc):
        nrow, ncol = self.df.shape
        print(f'* nrow & ncol: {nrow}, {ncol}')
        index_exc = []
        for case in range(nrow):
            stime = self.df.iloc[case,0][0:6]
            etime = self.df.iloc[case,-1][0:6]
            if int(stime) >= int(sym_exc) and int(etime) <= int(eym_exc):
                index_exc.append(case)
        self.df = self.df.drop([self.df.index[index_exc[i]] for i in range(len(index_exc))])
        self.df = self.df.reset_index(drop=True)
        return self.df


    def load_radar(self):
        nsample, nstep = self.df.shape

        data_rdr = np.ndarray(shape=(nsample,nstep,self.ny,self.nx),dtype=np.float32)
        for j in range(0, nsample):
            for i in range(0, nstep):
                cyy  = self.df.iloc[j,i][0:4]
                cmm  = self.df.iloc[j,i][4:6]
                cdd  = self.df.iloc[j,i][6:8]
                cHH  = self.df.iloc[j,i][8:10]
                cMM  = self.df.iloc[j,i][10:12]
                if int(cyy) in np.arange(2012,2016):
                    path_rdr = self.path+"RDR_downscale_1h_avg_CAPPI/"+cyy+"/"+cyy+cmm+cdd+"/"
                elif int(cyy) in np.arange(2016,2020):
                    path_rdr = self.path+"RDR_downscale_1h_avg_HSR/"+cyy+"/"+cyy+cmm+cdd+"/"

                file_rdr = "RDR_avg1h_128_4km_"+cyy+"-"+cmm+"-"+cdd+"_"+cHH+":"+cMM+":00.nc"
                nc_f = Dataset(path_rdr+file_rdr, "r")
                CAPPI_15 = nc_f.variables['rain1h']
                data_rdr[j][i][:][:] = CAPPI_15[:][:]
                nc_f.close()
        print(f'* done: load radar')
        return data_rdr

    def load_era(self,era_var):
        nsample, nstep = self.df.shape
        nvar = len(era_var)
        
        data_era = np.ndarray(shape=(nsample,nstep,self.ny,self.nx,nvar),dtype=np.float32)
        data_era_temp = np.ndarray(shape=(nsample,nstep,nvar,self.ny,self.nx))
        for k in range(0, nsample):
            for j in range(0, nstep):
                cyy = self.df.iloc[k,j][0:4]
                cmm = self.df.iloc[k,j][4:6]
                cdd = self.df.iloc[k,j][6:8]
                cHH = self.df.iloc[k,j][8:10]
                # ERA5 is written in according to UTC timeframe
                # Date should be shifted as 9 hours
                date_utc=datetime.datetime(int(cyy),int(cmm),int(cdd),int(cHH))-datetime.timedelta(hours=9)
                cyy_utc = date_utc.strftime('%Y')
                cymdh_utc = date_utc.strftime('%Y%m%d%H')

                path_era = self.path+"ERA5_intp_hourly/"+cyy_utc+"/"
                for i in range(0, nvar):
                    if era_var[i] in ['DIV925', 'VOR925']:
                        file_era = "era5_intp_128_4km_DIV_VOR_"+cymdh_utc+".nc"
                    if era_var[i] in ['CAPE_SFC', 'TCWV_SFC']:
                        file_era = "era5_intp_128_4km_CAPE_TPW_"+cymdh_utc+".nc"
                    nc_f = Dataset(path_era+file_era, "r")
                    era_var_temp = nc_f.variables[era_var[i]]
                    data_era_temp[k,j,i,:,:] = era_var_temp[:,:]
                    nc_f.close()
        data_era = np.moveaxis(data_era_temp,2,-1)
        print(f'* done: load era')
        return data_era



class processing:
    def __init__(self):
        self.minval = 0
        self.maxval = 0
        self.dval = 0

    # Standardization for the range of 0~1, minval and maxval are dependent to data
    def norm_minmax(self, data):
        self.minval = np.amin(data)
        self.maxval = np.amax(data)
        self.dval = self.maxval - self.minval
        data_norm  = (data-self.minval)/self.dval
        return data_norm

    def restore_minmax(self, data_norm):
        print(f'== Now restoring data with minval:{self.minval} and maxval:{self.maxval}')
        data = data_norm*self.dval + self.minval
        return data

    # Standardization for the range of 0~1, minval and maxval should be given.
    def norm_minmax_fix(self,data,minval=0,maxval=120):
        self.minval = minval
        self.maxval = maxval
        self.dval = self.maxval - self.minval
        data_norm = (data-self.minval)/self.dval
        return data_norm

    def restore_minmax_fix(self,data_norm):
        print(f'== Now restoring data with minval:{self.minval} and maxval:{self.maxval}')
        data = data_norm*self.dval + self.minval
        return data


#=============================================
# To avoid memory problem,
# use [fit_generator] instead of [fit].
#---------------------------------------------
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_set, batch_size, input_step, output_step, ny, nx, nch, shuffle=True):
        self.list_IDs = list_IDs
        self.data_set = data_set
        self.batch_size = batch_size
        self.input_step = input_step
        self.output_step = output_step
        self.ny = ny
        self.nx = nx
        self.nch = nch
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)/self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print(list_IDs_temp)
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X:[list within n_variables shape of (n_samples,n_steps,ny,nx,1)]
        # Initialization
        X = np.empty((self.batch_size,self.input_step,self.ny,self.nx,self.nch))
        y = np.empty((self.batch_size,self.output_step,self.ny,self.nx,1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,:,:,:,:] = self.data_set[ID,0:self.input_step,:,:,:]
            y[i,:,:,:,:] = np.expand_dims(self.data_set[ID,-self.output_step:,:,:,0], axis=-1)

        return X, y




#=============================================
# Trim and normalization for ERA variables 
#---------------------------------------------
def divide_pos_neg(data):
    data_pos = np.where(data<0,0,data)
    data_neg = np.where(data>0,0,data)
    return data_pos, data_neg

def trim_norm_era(data_era,minval,maxval):
    data_era = np.where(data_era>maxval, maxval, data_era)
    data_era = np.where(data_era<minval, minval, data_era)
    std_minmax_era = processing()
    normalized_era = std_minmax_era.norm_minmax_fix(data_era,minval,maxval)
    return normalized_era

#=============================================
# Data augmentation
#---------------------------------------------
def data_aug(data):
    nsample = data.shape[0]
    naug = int(nsample/2)
    np.random.shuffle(data)
    data_aug = data[:naug,::-1,::-1,::-1]
    return data_aug

def data_aug_multi(data):
    nsample = data.shape[0]
    naug = int(nsample/2)
    np.random.shuffle(data)
    data_aug = data[:naug,::-1,::-1,::-1,:]
    return data_aug



#========================================================
# Customized Loss functions
#--------------------------------------------------------
def mse_mae(y_true, y_pred):
    mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    mae = tf.multiply(mae,0.01)    # scaling mae by 0.01
    mse = tf.keras.losses.mean_squared_error(y_true,y_pred)
    custom_loss = tf.add(tf.multiply(mae,0.5),tf.multiply(mse,0.5))
    return custom_loss
        
def ssim(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1)
    return 1-ssim

def mse_mae_ssim(y_true, y_pred):
    mse_tensor = tf.keras.losses.mean_squared_error(y_true,y_pred)
    mse = tf.reduce_mean(mse_tensor)
    mae_tensor = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    mae = tf.reduce_mean(mae_tensor)
    ssim = tf.image.ssim(y_true, y_pred, max_val=1)
    return (mse+0.125*mae+0.011*(1-ssim))/3.   # set by trial & error

#--
def bmse_minmax(y_true, y_pred):
    node = [ 0, 1., 5., 10., 20., 50. ]
    weight = [ 1., 2., 5., 10., 20., 50.]
    norm = processing()   # create instance for [norm_minmax_fix] normalization
    norm_node = norm.norm_minmax_fix(np.asarray(node, dtype=np.float32))
    nnode = len(norm_node)
    diff_square = tf.square(tf.subtract(y_true, y_pred))
    for i in range(nnode):
        if i != nnode-1:
            diff_square = tf.where((y_true>=norm_node[i]) & (y_true<norm_node[i+1]),
                                   tf.multiply(diff_square,weight[i]), diff_square)
        elif i == nnode-1:
            diff_square = tf.where(y_true>=norm_node[i], 
                                   tf.multiply(diff_square,weight[i]), diff_square)
    bmse = tf.reduce_mean(diff_square)
    return bmse

def bmse_new(y_true, y_pred):
    node = [ 0, 1., 5., 10., 20., 50. ]
    weight = [ 1., 2., 5., 10., 20., 50.]
    norm = processing()   # create instance for [norm_minmax_fix] normalization
    norm_node = norm.norm_new(np.asarray(node, dtype=np.float32))
    nnode = len(norm_node)
    diff_square = tf.square(tf.subtract(y_true, y_pred))
    for i in range(nnode):
        if i != nnode-1:
            diff_square = tf.where((y_true>=norm_node[i]) & (y_true<norm_node[i+1]),
                                   tf.multiply(diff_square,weight[i]), diff_square)
        elif i == nnode-1:
            diff_square = tf.where(y_true>=norm_node[i], 
                                   tf.multiply(diff_square,weight[i]), diff_square)
    bmse = tf.reduce_mean(diff_square)
    return bmse

#--
def bmae_minmax(y_true, y_pred):
    node = [ 0, 1., 5., 10., 20., 50. ]
    weight = [ 1., 2., 5., 10., 20., 50.]
    norm = processing()   # create instance for [norm_minmax_fix] normalization
    norm_node = norm.norm_minmax_fix(np.asarray(node, dtype=np.float32))
    nnode = len(norm_node)
    diff_abs = tf.abs(tf.subtract(y_true, y_pred))
    for i in range(nnode):
        if i != nnode-1:
            diff_abs = tf.where((y_true>=norm_node[i]) & (y_true<norm_node[i+1]), 
                                tf.multiply(diff_abs,weight[i]), diff_abs)
        elif i == nnode-1:
            diff_abs = tf.where(y_true>=norm_node[i], 
                                tf.multiply(diff_abs,weight[i]), diff_abs)
    bmae = tf.reduce_mean(diff_abs)
    return bmae

def bmae_new(y_true, y_pred):
    node = [ 0, 1., 5., 10., 20., 50. ]
    weight = [ 1., 2., 5., 10., 20., 50.]
    norm = processing()   # create instance for [norm_minmax_fix] normalization
    norm_node = norm.norm_new(np.asarray(node, dtype=np.float32))
    nnode = len(norm_node)
    diff_abs = tf.abs(tf.subtract(y_true, y_pred))
    for i in range(nnode):
        if i != nnode-1:
            diff_abs = tf.where((y_true>=norm_node[i]) & (y_true<norm_node[i+1]), 
                                tf.multiply(diff_abs,weight[i]), diff_abs)
        elif i == nnode-1:
            diff_abs = tf.where(y_true>=norm_node[i], 
                                tf.multiply(diff_abs,weight[i]), diff_abs)
    bmae = tf.reduce_mean(diff_abs)
    return bmae

#--
def bmse_bmae_minmax(y_true, y_pred):
    bmse = bmse_minmax(y_true, y_pred)
    bmae = bmae_minmax(y_true, y_pred)
    tf.print("== bmse_bmae: ", (bmse+0.1*bmae)/2.)
    return (bmse+0.1*bmae)/2.         # based on [def01_M00_b20_K533_Nminmax_AGMT] exp set.

def bmse_bmae_new(y_true, y_pred):
    bmse = bmse_new(y_true, y_pred)
    bmae = bmae_new(y_true, y_pred)
    return (bmse+0.2*bmae)/2.         # based on [def01_M00_b20_K533_Nnew_AGMT] exp set.

#--
def bmae_ssim_new(y_true, y_pred):
    bmae = bmae_new(y_true, y_pred)
    ssim = tf.image.ssim(y_true, y_pred, max_val=1)
    return (bmae+0.225*(1-ssim))/2.   # based on [def01_M00_b20_K533_Nnew_AGMT] exp set.

#--
def bmse_bmae_ssim_minmax(y_true, y_pred):
    bmse = bmse_minmax(y_true, y_pred)
    bmae = bmae_minmax(y_true, y_pred)
    ssim = tf.image.ssim(y_true, y_pred, max_val=1)
    return (bmse+0.1*bmae+0.015*(1-ssim))/3.   # based on [def01_M00_b20_K533_Nminmax_AGMT] exp set.

def bmse_bmae_ssim_new(y_true, y_pred):
    bmse = bmse_new(y_true, y_pred)
    bmae = bmae_new(y_true, y_pred)
    ssim = tf.image.ssim(y_true, y_pred, max_val=1)
    return (bmse+0.2*bmae+0.04*(1-ssim))/3.   # based on [def01_M00_b20_K533_Nnew_AGMT] exp set.


#-------------------------------------
# Customized Loss function based on verification indices
#------------------------------------- 
class score:
    def __init__(self, thval, y_true, y_pred):
        self.thval = thval
        self.FO = 0.
        self.FX = 0.
        self.XO = 0.
        self.XX = 0.
        self.y_true = y_true
        self.y_pred = y_pred
        FO_0 = tf.reduce_sum(tf.where((self.y_pred>=self.thval) & (self.y_true>=self.thval),1.,0.))
        self.FO = tf.add(FO_0,1e-10)
        self.FX = tf.reduce_sum(tf.where((self.y_pred>=self.thval) & (self.y_true<self.thval),1.,0.))
        self.XO = tf.reduce_sum(tf.where((self.y_pred<self.thval) & (self.y_true>=self.thval),1.,0.))
        self.XX = tf.reduce_sum(tf.where((self.y_pred<self.thval) & (self.y_true<self.thval),1.,0.))

    def csi(self, weight):
        return weight*(1-self.FO/(self.FO+self.FX+self.XO))

    def far(self, weight):
        return weight*(self.FX/(self.FO+self.FX))


def MCSLoss(y_true, y_pred):
    thval  = [ 0.1,  1, 2, 4, 6, 8, 10 ]   # set by trial & error
    weight = [  20, 10, 5, 4, 3, 2,  1 ]   # set by trial & error
    norm = processing()  # create instance for [norm_minmax_fix] normalization
    norm_thval = norm.norm_minmax_fix(np.asarray(thval, dtype=np.float32))
    nthval = len(norm_thval)
    csi = 0
    far = 0
    for i in range(nthval):
        csi += score(norm_thval[i],y_true,y_pred).csi(weight[i])
        far += score(norm_thval[i],y_true,y_pred).far(weight[i])

    alpha = 0.00005
    bmse_bmae = bmse_bmae_minmax(y_true,y_pred)
    tf.print("== csi: ", alpha*csi)
    tf.print("== far: ", alpha*far)
    tf.print("== bmse_bmae: ", bmse_bmae)
    return (bmse_bmae + alpha*(csi+far))/3.   



#========================================================
# Customized Layers 
#--------------------------------------------------------
class ECA(Layer):
    def __init__(self, gamma=2, b=1):
        super(ECA, self).__init__()
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        nch = input_shape[-1]
        t = int(abs((np.log(nch)/np.log(2)+self.b)/self.gamma))
        self.k = t if t % 2 else t + 1
        self.avg_pool = GlobalAveragePooling2D(data_format='channels_last')
        self.conv1d = Conv1D(filters=1, kernel_size=self.k, strides=1, padding='same',
                             data_format='channels_last', activation='sigmoid')

    def call(self, inputs):
        @tf.function
        def multiply(inputs,x):
            return tf.math.multiply(inputs,x)
        x = self.avg_pool(inputs)                  # [batch, channels]
        x = self.conv1d(tf.expand_dims(x,axis=-1)) # [batch, channels,1]
        x = tf.squeeze(x,-1)                       # [batch, channels]
        x = tf.expand_dims(x, axis=1)              # [batch, 1, channels]
        x = tf.expand_dims(x, axis=1)              # [batch, 1, 1, channels]
        x = tf.broadcast_to(x, tf.shape(inputs))   # [batch, ny, nx, channels]
        return multiply(inputs, x)

    def compute_output_shape(self, input_shape):
        ## define output_shape explicitly
        #input_shape = tensor_shape.TensorShape(input_shape).as_list()
        #return tensor_shape.TensorShape([input_shape[0],input_shape[1],input_shape[2],input_shape[3]]) 
        return tensor_shape.TensorShape(input_shape) 

    def get_config(self):
        config = super(ECA, self).get_config()
        config.update({"gamma":self.gamma, "b":self.b})
        return config




