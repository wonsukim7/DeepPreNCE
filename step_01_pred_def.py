import deep_model
import deep_utils
import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta
from netCDF4 import Dataset
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

mirrored_strategy = tf.distribute.MirroredStrategy(devices=[ GPU_CHOICE ])


exp_name="EXP_NAME"

syear=SYEAR_PRED     # Starting year to predict
eyear=EYEAR_PRED     # Last year to predict

sym_exc = SYM_EXC_PRED   # [yyyymm], starting month to be excluded for prediction
eym_exc = EYM_EXC_PRED   # [yyyymm], last month to be excluded for prediction


path_data = "./data/"  # Path for Data dir
sample_list = "./sample_list_2012-2019_JJAS_RDR_avg1h_1hrs.csv"  # path for sample list file


ckpt_path='CKPT_PATH'   # Chosen checkpoint path including the name of checkpoint
ckpt_name='CKPT_NAME'   # Checkpoint name
pred_dir='PRED_DIR'

pd.set_option('display.max_rows', 1000)

ny = 128  # Y-direction : Dimension size of input data
nx = 128  # X-direction : Dimension size of input data
nch = 1   # Dimension size of channels

input_step = 6   # step number for input sequences
output_step = 6  # step numbers for output sequences



'== Get data =='
data = deep_utils.get_Data(path_data,sample_list,syear,eyear,ny,nx)
data.getsample()
datalist = data.exclude_sample(sym_exc,eym_exc)
print(datalist)
print('')
data_rdr = data.load_radar()   # data_rdr = data_rdr[nsample,nstep,ny,nx]
print(f' Shape of loaded radar data: {data_rdr.shape}')

'== Preprocessing for missing values =='
print(f'** check Min Max data_rdr: min: {np.amin(data_rdr)} and max: {np.amax(data_rdr)}')
data_rdr = np.where(data_rdr==-30000, 0, data_rdr)
print(f'** check Min Max data_rdr: min: {np.amin(data_rdr)} and max: {np.amax(data_rdr)}')
data_rdr = np.where(data_rdr>=120, 120, data_rdr)
print(f'** check Min Max data_rdr: min {np.amin(data_rdr)} and max: {np.amax(data_rdr)}')


data_in_rdr = data_rdr[:,0:input_step,:,:]   # Input data
data_out_rdr = data_rdr[:,-output_step:,:,:]   # Ground truth for prediction

'== Standardization of Input data =='
std_minmax_rdr = deep_utils.processing()
test_in_rdr = std_minmax_rdr.norm_minmax_fix(data_in_rdr)

# For Checking 
std_minmax_rdr_check = deep_utils.processing()
check_rdr = std_minmax_rdr_check.norm_minmax_fix(data_rdr)
check_in_rdr = check_rdr[:,0:input_step,:,:]   # Input data
check_out_rdr = check_rdr[:,-output_step:,:,:]   # Ground truth for prediction
print(f' Shape of input [RADAR] sequence for test: {test_in_rdr.shape}')
print(f'**check Min Max test_in_rdr: min: {np.amin(test_in_rdr)} and max: {np.amax(test_in_rdr)}')



'== Expand Channel axis in here == '
test_in = np.expand_dims(test_in_rdr, axis=4)
print(f' Shape of input sequence for test: {test_in.shape}')
# For Checking
check_in = np.expand_dims(check_in_rdr, axis=4)
check_out = np.expand_dims(check_out_rdr, axis=4)



'== Build a new model and load weights =='
inputs = Input(shape=(input_step,ny,nx,nch))
with mirrored_strategy.scope():
    new_model = deep_model.Model_def(inputs,output_step)
    new_model.compile(optimizer=OPTIMIZER, loss=LOSS)
    # Checking
    scores = new_model.evaluate(check_in, check_out)
    print('== Scores after LOAD_MODEL without weights ==')
    print(scores)
    print('')
    print(f'* load weights: {ckpt_path}')
    new_model.load_weights(ckpt_path)

    # Checking
    scores = new_model.evaluate(check_in, check_out)
    print('== Scores after LOAD_MODEL with weights ==')
    print(scores)
    print('')
    
    
    '== Predict =='
    predict = new_model.predict(test_in)
    print(f'* Shape of predict: {predict.shape}')
    
    # Restore predict to original scale
    print(f' Min and Max of predict before restoring: {np.amin(predict)} and {np.amax(predict)}')
    predict = std_minmax_rdr.restore_minmax_fix(predict)
    print(f' Min and Max of predict after restoring: {np.amin(predict)} and {np.amax(predict)}')



'== Write prediction results =='
ndim = predict.shape
# remove right-most dimension
predict = predict.reshape((ndim[0], ndim[1], ndim[2], ndim[3]))
print(f'* Shape of predict: {predict.shape}')

'--test'
print(predict[3,3,:,:])

# Write results to netcdf
fout = pred_dir+'/predict_'+ckpt_name+"_"+exp_name+'.nc'
nc_file = Dataset(fout, 'w', format='NETCDF4')
nc_file.description = "deep-learning based prediction results from ["+exp_name+"]"

# dimension
nc_file.createDimension('case', ndim[0])
nc_file.createDimension('time_step', ndim[1])
nc_file.createDimension('south_north', ndim[2])
nc_file.createDimension('west_east', ndim[3])

# create variables
case = nc_file.createVariable('case','i8', ('case'))
case.long_name = 'Initial time for each samples'
time_step = nc_file.createVariable('time_step','i4',('time_step'))
time_step.long_name = 'hours since [case]'
date_list = nc_file.createVariable('date_list','i8',('case','time_step'))
date_list.long_name = 'date and time string for each predictions'
pred_rainrate = nc_file.createVariable('pred_rainrate','f8',('case','time_step','south_north','west_east'))
pred_rainrate.long_name = 'prediction results for rainrate'
pred_rainrate.units = 'mm/h'

# write data
case[:] = datalist.iloc[:,0].values
print(datalist.iloc[:,0].values)
#time_step[:] = np.arange(1+input_step,1+input_step+output_step,1)
time_step[:] = np.arange(input_step,input_step+output_step,1)
date_list[:,:] = datalist.iloc[:,-output_step:].values
pred_rainrate[:,:,:,:] = predict[:,:,:,:]
print(f'! wrote data: case.shape is {case.shape}')
print(f'! wrote data: time_step.shape is {time_step.shape}')
print(f'! wrote data: datelist.shape is {date_list.shape}')
print(f'! wrote data: pred_rainrate.shape is {pred_rainrate.shape}')

















