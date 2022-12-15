import deep_model
import deep_utils
import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta
from netCDF4 import Dataset
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import multi_gpu_model

mirrored_strategy = tf.distribute.MirroredStrategy(devices=[ GPU_CHOICE ])   ## GPU choice


exp_name="EXP_NAME"

syear=SYEAR_FIT     # Starting year to fit the model
eyear=EYEAR_FIT     # Last year to fit the model
sym_exc = SYM_EXC_FIT    # [yyyymm], starting month to be excluded for test or prediction
eym_exc = EYM_EXC_FIT    # [yyyymm], last month to be excluded for test or prediction

#- Set additional input from ERA5
era_var = [ ERA_VARS ]   

path_data = "/data/Data/"   # Path for Data dir
sample_list = "./sample_list_2012-2019_JJAS_RDR_avg1h_1hrs.csv"  # path for sample list file

pd.set_option('display.max_rows', 8000)

ny = 128   # Y-direction : Dimension size of input data
nx = 128   # X-direction : Dimension size of input data
nch = 1+len(era_var)    # Dimension size of channels
nera = len(era_var)


#-- set the number of channels
if "DIV925" in era_var:
    nch = nch+1
if "VOR925" in era_var:
    nch = nch+1


input_step = 6    # step number for input sequences
output_step = 6   # step number for output sequences


ckpt_dir='CKPT_DIR'
log_dir='LOG_DIR'




'== Get data =='
data = deep_utils.get_Data(path_data,sample_list,syear,eyear,ny,nx)
data.getsample()
data.exclude_sample(sym_exc,eym_exc)

'== load preprocessed radar data =='
data_rdr = data.load_radar()   # data_rdr = data_rdr[nsample,nstep,ny,nx]
print(f' Shape of loaded radar data: {data_rdr.shape}')

'== Preprocessing for missing values =='
print(f'**check Min Max data_rdr: min: {np.amin(data_rdr)} and max: {np.amax(data_rdr)}')
data_rdr = np.where(data_rdr==-30000, 0, data_rdr)
print(f'**check Min Max data_rdr: min: {np.amin(data_rdr)} and max: {np.amax(data_rdr)}')
data_rdr = np.where(data_rdr>=120, 120, data_rdr)
print(f'**check Min Max data_rdr: min: {np.amin(data_rdr)} and max: {np.amax(data_rdr)}')

'== load preprocessed ERA5 data =='
data_era = data.load_era(era_var)   # load ERA5 data 
print("==== Check data_era =====")
print(f' Shape of loaded era data: {data_era.shape}')
print('')



'== Standardization of Input data =='
std_minmax_rdr = deep_utils.processing()
#train_rdr = std_minmax_rdr.norm_minmax(data_rdr)
train_rdr = std_minmax_rdr.norm_minmax_fix(data_rdr)
print(f' Shape of [train_rdr] sequence for training: {train_rdr.shape}')
print(f'**check Min Max [train_rdr]: min: {np.amin(train_rdr)} and max: {np.amax(train_rdr)}')
print('')


'== Prepare [train_era] =='
# if DIV or VOR is used, it will be devided into 2 channels depending on sign (+ or -)'
train_era = np.ndarray(shape=(data_era.shape[0],data_era.shape[1],ny,nx,nch-1),dtype=np.float32)
aux_ch = 0
for i in range(len(era_var)):
    if era_var[i] in ["DIV925","VOR925"]:   # 
        if era_var[i] == "DIV925":
            minval = MIN_DIV
            maxval = MAX_DIV
        if era_var[i] == "VOR925":
            minval = MIN_VOR
            maxval = MAX_VOR
        dvvr_pos, dvvr_neg = deep_utils.divide_pos_neg(data_era[:,:,:,:,i])
        print(f'**check Min Max data_era_{i} {era_var[i]}_pos: min: {np.amin(dvvr_pos)} and max: {np.amax(dvvr_pos)}')
        print(f'**check Min Max data_era_{i} {era_var[i]}_neg: min: {np.amin(dvvr_neg)} and max: {np.amax(dvvr_neg)}')
        train_era[:,:,:,:,i] = deep_utils.trim_norm_era(dvvr_pos,0,maxval)
        train_era[:,:,:,:,nera+aux_ch] = deep_utils.trim_norm_era(-dvvr_neg,0,-minval)
        print(f'**check Min Max train_era_{i} {era_var[i]}: min: {np.amin(train_era[:,:,:,:,i])} and max: {np.amax(train_era[:,:,:,:,i])}')
        print(f'**check Min Max train_era_{nera+aux_ch} {era_var[i]}: min: {np.amin(train_era[:,:,:,:,nera+aux_ch])} and max: {np.amax(train_era[:,:,:,:,nera+aux_ch])}')
        aux_ch = aux_ch+1
    if era_var[i] == "CAPE_SFC":
        minval = MIN_CAPE     # 1st try
        maxval = MAX_CAPE
        train_era[:,:,:,:,i] = deep_utils.trim_norm_era(data_era[:,:,:,:,i],minval,maxval)
        print(f'**check Min Max data_era_{i} {era_var[i]}: min: {np.amin(data_era[:,:,:,:,i])} and max: {np.amax(data_era[:,:,:,:,i])}')
        print(f'**check Min Max train_era_{i} {era_var[i]}: min: {np.amin(train_era[:,:,:,:,i])} and max: {np.amax(train_era[:,:,:,:,i])}')
    if era_var[i] == "TCWV_SFC":
        minval = MIN_TCWV
        maxval = MAX_TCWV
        train_era[:,:,:,:,i] = deep_utils.trim_norm_era(data_era[:,:,:,:,i],minval,maxval)
        print(f'**check Min Max data_era_{i} {era_var[i]}: min: {np.amin(data_era[:,:,:,:,i])} and max: {np.amax(data_era[:,:,:,:,i])}')
        print(f'**check Min Max train_era_{i} {era_var[i]}: min: {np.amin(train_era[:,:,:,:,i])} and max: {np.amax(train_era[:,:,:,:,i])}')
        print('')
print(f' Shape of [train_era]: {train_era.shape}')
print('')






'== To enable data augmentation, construct [ndarray] containing all training data set =='
train_all = np.ndarray(shape=[data_rdr.shape[0],data_rdr.shape[1],ny,nx,nch],dtype=np.float32)
for i in range(0, nch):
    if i == 0:
        train_all[:,:,:,:,i] = train_rdr[:,:,:,:]
    else:
        train_all[:,:,:,:,i] = train_era[:,:,:,:,i-1]
print(f' Shape of [train_all]: {train_all.shape}')
print('')

'== Data Augmentation =='
TF_augment = TF_AGMT
if TF_augment is True:
    train_all_aug = deep_utils.data_aug_multi(train_all)
    print(f' Shape of [original training data]: {train_all.shape}')
    print(f' Shape of [augmented data]: {train_all_aug.shape}')
# In the fitting process, validation set will be chosen from the last.
# To prevent selecting of augmented data as validation set,
# augmented data will be appended to the reversed original data set.
    train_all = train_all[::-1,:,:,:,:]
    train_all = np.append(train_all, train_all_aug, axis=0)
    train_all = train_all[::-1,:,:,:,:]
# Dimensions for [train_all]: train_all[n_sample, n_step, ny, nx, nch]
    print(f' Shape of [resulted training data]: {train_all.shape}')


'== Construct Data generator for fitting =='
np.random.shuffle(train_all)
nsample = train_all.shape[0]
ntrain = int(nsample*0.8)
train_set = train_all[:ntrain,:,:,:,:]
valid_set = train_all[ntrain:,:,:,:,:]
nvalid = valid_set.shape[0]
print(' ')
print(f' ** [nsample], [ntrain], [nvalid]: {nsample}, {ntrain}, {nvalid}')

# Set partition for training and validation
#partition={'train':[i for i in range(ntrain)], 'valid':[i for i in range(nvalid)]}

# Generators
#training_generator = deep_utils.DataGenerator(partition['train'],train_set,BATCH_SIZE,
#                                  input_step,output_step,ny,nx,nch,shuffle=True)
#validation_generator = deep_utils.DataGenerator(partition['valid'],valid_set,BATCH_SIZE,
#                                  input_step,output_step,ny,nx,nch,shuffle=True)



'== Prepare input and output sequencese =='
train_in =[]
for i in range(nch):
    train_in.append(np.expand_dims(train_all[:,0:input_step,:,:,i],axis=-1))
    print(f' Shape of input sequence{i} for training: {np.expand_dims(train_all[:,0:input_step,:,:,i],axis=-1).shape}')
print(f' List length of input sequence for training: {len(train_in)}')

train_out = np.expand_dims(train_all[:,-output_step:,:,:,0],axis=-1)
print(f' Shape of output sequence for training: {train_out.shape}')




'== Set callbacks =='
# Early Stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# Model Checkpoint
ckpt_file=ckpt_dir+"/cp-{epoch:04d}-{val_loss:.5f}"
ckpt = ModelCheckpoint(ckpt_file, verbose=1, save_weights_only=True, #save_best_only=True,
                       monitor='val_loss', mode='min', period=5) #save_freq=100)

# Tensorboard
tsboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)

#callbacks = [es, ckpt, tsboard]
callbacks = [ckpt, tsboard]



'== Model Fit =='
input_shape = [input_step, ny, nx, nch]

with mirrored_strategy.scope():
    model = deep_model.Model_multi(input_shape,output_step,era_var)
    print(model.summary())

    model.compile(optimizer=OPTIMIZER, loss=LOSS)
    history = model.fit(train_in, train_out, batch_size=BATCH_SIZE, epochs=5, validation_split=0.2,
                    callbacks=callbacks)
    #history = model.fit_generator(generator=training_generator,
    #                              validation_data=validation_generator,
    #                              epochs=5,
    #                              #use_multiprocessing=True,
    #                              #workers=6,
    #                              callbacks=callbacks)


# save history
hist_df = pd.DataFrame(history.history)
with open('./output/history_'+exp_name+'.csv', "w") as f:
    hist_df.to_csv(f)


