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

multi_option="MULTI_OPTION"


syear=SYEAR_PRED     # Starting year to predict
eyear=EYEAR_PRED     # Last year to predict

sym_exc = SYM_EXC_PRED    # [yyyymm], starting month to be excluded for prediction
eym_exc = EYM_EXC_PRED    # [yyyymm], last month to be excluded for prediction

#- Set additional input from ERA5
era_var = [ ERA_VARS ]

path_data = "/data/Data/"  # Path for Data dir
sample_list = "./sample_list_2012-2019_JJAS_RDR_avg1h_1hrs.csv"  # path for sample list file


ckpt_path='CKPT_PATH'    # Chosen checkpoint path including the name of checkpoint
ckpt_name='CKPT_NAME'    # Checkpoint name
pred_dir='PRED_DIR'

pd.set_option('display.max_rows', 1000)

ny = 128  # Y-direction : Dimension size of input data
nx = 128  # X-direction : Dimension size of input data
nch = 1+len(era_var)   # Dimension size of channels
nera = len(era_var)


#-- set the number of channels
if "DIV925" in era_var:
    nch = nch+1
if "VOR925" in era_var:
    nch = nch+1


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

'== load preprocessed ERA5 data =='
data_era = data.load_era(era_var)    # load ERA5 data

print("==== Check data_era ====")
print(f' Shape of loaded era data: {data_era.shape}')



'== Standardization of Input data =='
std_minmax_rdr = deep_utils.processing()
test_rdr = std_minmax_rdr.norm_minmax_fix(data_rdr)
print(f' Shape of [test_rdr] sequence for test: {test_rdr.shape}')
print(f'**check Min Max [test_rdr]: min: {np.amin(test_rdr)} and max: {np.amax(test_rdr)}')



'== Prepare [test_era] =='
# if DIV or VOR is used, it will be devided into 2 channels depending on sign (+ or -)'
test_era = np.ndarray(shape=(data_era.shape[0],data_era.shape[1],ny,nx,nch-1),dtype=np.float32)
aux_ch = 0
for i in range(len(era_var)):
    if era_var[i] in ["DIV925","VOR925"]:   # Actually, 'DIV925' contains 'CONV925'
        if era_var[i] == "DIV925":
            minval = MIN_DIV
            maxval = MAX_DIV
        if era_var[i] == "VOR925":
            minval = MIN_VOR
            maxval = MAX_VOR
        dvvr_pos, dvvr_neg = deep_utils.divide_pos_neg(data_era[:,:,:,:,i])
        print(f'**check Min Max data_era_{i} {era_var[i]}_pos: min: {np.amin(dvvr_pos)} and max: {np.amax(dvvr_pos)}')
        print(f'**check Min Max data_era_{i} {era_var[i]}_neg: min: {np.amin(dvvr_neg)} and max: {np.amax(dvvr_neg)}')
        test_era[:,:,:,:,i] = deep_utils.trim_norm_era(dvvr_pos,0,maxval)
        test_era[:,:,:,:,nera+aux_ch] = deep_utils.trim_norm_era(-dvvr_neg,0,-minval)
        print(f'**check Min Max test_era_{i} {era_var[i]}: min: {np.amin(test_era[:,:,:,:,i])} and max: {np.amax(test_era[:,:,:,:,i])}')
        print(f'**check Min Max test_era_{nera+aux_ch} {era_var[i]}: min: {np.amin(test_era[:,:,:,:,nera+aux_ch])} and max: {np.amax(test_era[:,:,:,:,nera+aux_ch])}')
        aux_ch = aux_ch+1
    if era_var[i] == "CAPE_SFC":
        minval = MIN_CAPE     # 1st try
        maxval = MAX_CAPE
        test_era[:,:,:,:,i] = deep_utils.trim_norm_era(data_era[:,:,:,:,i],minval,maxval)
        print(f'**check Min Max data_era_{i} {era_var[i]}: min: {np.amin(data_era[:,:,:,:,i])} and max: {np.amax(data_era[:,:,:,:,i])}')
        print(f'**check Min Max test_era_{i} {era_var[i]}: min: {np.amin(test_era[:,:,:,:,i])} and max: {np.amax(test_era[:,:,:,:,i])}')
    if era_var[i] == "TCWV_SFC":
        minval = MIN_TCWV
        maxval = MAX_TCWV
        test_era[:,:,:,:,i] = deep_utils.trim_norm_era(data_era[:,:,:,:,i],minval,maxval)
        print(f'**check Min Max data_era_{i} {era_var[i]}: min: {np.amin(data_era[:,:,:,:,i])} and max: {np.amax(data_era[:,:,:,:,i])}')
        print(f'**check Min Max test_era_{i} {era_var[i]}: min: {np.amin(test_era[:,:,:,:,i])} and max: {np.amax(test_era[:,:,:,:,i])}')
        print('')
print(f' Shape of [test_era]: {test_era.shape}')
print('')





'== Construct [ndarray] containing all test data set == '
test_all = np.ndarray(shape=[data_rdr.shape[0],data_rdr.shape[1],ny,nx,nch],dtype=np.float32)
for i in range(0, nch):
    if i == 0:
        test_all[:,:,:,:,i] = test_rdr[:,:,:,:]
    else:
        test_all[:,:,:,:,i] = test_era[:,:,:,:,i-1]
print(f' Shape of [test_all]: {test_all.shape}')
print('')



'== Prepare input for test =='
test_in = test_all[:,0:input_step,:,:,:]
print(f' Shape of input sequence for test: {test_in.shape}')
print('')

'== Prepare input & output for checking =='
check_in = test_in
check_out = np.expand_dims(test_all[:,-output_step:,:,:,0], axis=-1)
print(f' Shape of input sequence for check: {check_in.shape}')
print(f' Shape of output sequnece for check: {check_out.shape}')



'== Build a new model and load weights =='
input_shape = [input_step, ny, nx, nch]
with mirrored_strategy.scope():
    new_model = deep_model.Model_multi(input_shape,output_step,era_var)
    new_model.compile(optimizer=OPTIMIZER, loss=LOSS)
    # Checking
#    scores = new_model.evaluate(check_in, check_out)
    check_in_list = [ np.expand_dims(test_in[:,:,:,:,i], axis=-1) for i in range(nch) ]
    scores = new_model.evaluate(check_in_list, check_out)
    print('== Scores after LOAD_MODEL without weights ==')
    print(scores)
    print('')
    print(f'* load weights: {ckpt_path}')
    new_model.load_weights(ckpt_path)

    # Checking
    scores = new_model.evaluate(check_in_list, check_out)
    print('== Scores after LOAD_MODEL with weights ==')
    print(scores)
    print('')
    
    
    '== Predict =='
    test_in_list = check_in_list
    predict = new_model.predict(test_in_list)
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

















