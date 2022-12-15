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


mirrored_strategy = tf.distribute.MirroredStrategy(devices=[ GPU_CHOICE ])   ## Using gpu0[V100] and gpu1[V100]



exp_name="EXP_NAME"

syear=SYEAR_FIT     # Starting year to fit the model
eyear=EYEAR_FIT     # Last year to fit the model
sym_exc = SYM_EXC_FIT    # [yyyymm], starting month to be excluded for test or prediction
eym_exc = EYM_EXC_FIT    # [yyyymm], last month to be excluded for test or prediction

path_data = "./data/"   # Path for Data dir
sample_list = "./sample_list_2012-2019_JJAS_RDR_avg1h_1hrs.csv"  # path for sample list file

pd.set_option('display.max_rows', 8000)

ny = 128   # Y-direction : Dimension size of input data
nx = 128   # X-direction : Dimension size of input data
nch = 1    # Dimension size of channels

input_step = 6    # step number for input sequences
output_step = 6   # step number for output sequences


ckpt_dir='CKPT_DIR'
log_dir='LOG_DIR'




'== Get data =='
data = deep_utils.get_Data(path_data,sample_list,syear,eyear,ny,nx)
data.getsample()
data.exclude_sample(sym_exc,eym_exc)
data_rdr = data.load_radar()   # data_rdr = data_rdr[nsample,nstep,ny,nx]
print(f' Shape of loaded radar data: {data_rdr.shape}')


'== Preprocessing for missing values =='
print(f'**check Min Max data_rdr: min: {np.amin(data_rdr)} and max: {np.amax(data_rdr)}')
data_rdr = np.where(data_rdr==-30000, 0, data_rdr)
print(f'**check Min Max data_rdr: min: {np.amin(data_rdr)} and max: {np.amax(data_rdr)}')
data_rdr = np.where(data_rdr>=120, 120, data_rdr)
print(f'**check Min Max data_rdr: min: {np.amin(data_rdr)} and max: {np.amax(data_rdr)}')



'== Standardization of Input data =='
std_minmax_rdr = deep_utils.processing()
train_rdr = std_minmax_rdr.norm_minmax_fix(data_rdr)


'== Data Augmentation =='
TF_augment = TF_AGMT
if TF_augment is True:
    train_aug = deep_utils.data_aug(train_rdr)
    print(f' Shape of [orignal training data]: {train_rdr.shape}')
    print(f' Shape of [augmented data]: {train_aug.shape}')
# In the fitting process, validation set will be chosen from the last.
# To prevent selecting of augmented data as validation set,
# augmented data will be appended to the reversed original data set.
    train_rdr = train_rdr[::-1,:,:,:]
    train_rdr = np.append(train_rdr, train_aug, axis=0)
    train_rdr = train_rdr[::-1,:,:,:]
    print(f' Shape of [resulted training data]: {train_rdr.shape}')



'== Expand Channel axis in here ==' 
# set [train_all] for default model
train_all = np.expand_dims(train_rdr, axis=4)

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
nch=1
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
inputs = Input(shape=(input_step,ny,nx,nch))

with mirrored_strategy.scope():
    model = deep_model.Model_def(inputs,output_step)
    print(model.summary())

    model.compile(optimizer=OPTIMIZER, loss=LOSS)
    history = model.fit(train_in, train_out, batch_size=BATCH_SIZE, epochs=5, validation_split=0.2,
                    callbacks=callbacks)
#    history = model.fit_generator(generator=training_generator,
#                                  validation_data=validation_generator,
#                                  epochs=300,   # epochs=300 in the case of [def04]
#                                  #use_multiprocessing=True,
#                                  #workers=6,
#                                  callbacks=callbacks)



# save history
hist_df = pd.DataFrame(history.history)
with open('./output/history_'+exp_name+'.csv', "w") as f:
    hist_df.to_csv(f)



