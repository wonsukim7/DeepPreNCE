#!/bin/bash

# Set variables

# - the name of experiment defined by user
#exp_name="def04_ECA_b20_K533_Nminmax_Lbmse_bmae_AGMT"
exp_name="test_def"

# - Set gpus for fit and prediction
gpu_choice='"/gpu:0", "/gpu:1"'   # e.g., '"/gpu:1", "/gpu:2"'


# - Set Batch Size
batch_size="20"     # e.g., 8, 12, 16, 20

# - Set optimizer
optimizer="Adam(learning_rate=0.0001)" # e.g., [adadelta], [adam], [SGD]

# - Set loss
loss="deep_utils.bmse_bmae_minmax"   # try [binary_crossentropy], [mae], [mse], [my_loss_fn]
#  CBloss: deep_utils.bmse_bmae_minmax
#  MCSloss: deep_utils.MCSLoss


# - Set period for fitting process
syear_fit="2012"      # [yyyy], Starting year to fit the model
eyear_fit="2019"      # [yyyy], Last year to fit the model
sym_exc_fit="201706"  # [yyyymm], starting month to be excluded for test or prediction
eym_exc_fit="201709"  # [yyyymm], last month to be excluded for test or prediction

# - Set period for prediction process
syear_pred="2017"     # [yyyy], Starting year to predict
eyear_pred="2017"     # [yyyy], Last year to predict
sym_exc_pred="201709"  # [yyyymm], starting month to be excluded for test or prediction
eym_exc_pred="201709"  # [yyyymm], last month to be excluded for test or prediction


# - Set option for Data Augmentation
TF_AGMT="True"   # [True] or [False]


# - Choose checkpoint for [pred], [plot], [verity] processes
epoch_num="5"  # Now 200 is default epoch number.
                 # You can change this to any epoch number.

epoch_num_d4=`printf "%04d\n" ${epoch_num}`


# - Set processes need to be conducted
#     
#       [fit] [pred] [plot] [verify]
switch=(  "F"   "F"    "T"    "T" )      # "T": on,  "F": off




# - Set paths 
# directory for scripts
dir_scrp="./output/script/${exp_name}/"
if [ -d $dir_scrp ] && [ ${switch[0]} == "T" ]; then
    rm -rf $dir_scrp
fi
mkdir -p $dir_scrp

# directory for checkpoints
ckpt_dir="./output/ckpt/${exp_name}"
if [ -d $ckpt_dir ] && [ ${switch[0]} == "T" ];then
    rm -rf $ckpt_dir
fi
mkdir -p $ckpt_dir

# directory for log (tensorboard)
log_dir="./output/log/${exp_name}"
if [ -d $log_dir ] && [ ${switch[0]} == "T" ];then
    rm -rf $log_dir
fi
mkdir -p $log_dir

# directory for predict results
pred_dir="./output/predict/${exp_name}"
if [ -d $pred_dir ] && [ ${switch[0]} == "T" ];then
    rm -rf $pred_dir
fi
mkdir -p $pred_dir

# directory for images
image_dir="./output/image/${exp_name}/cp_${epoch_num_d4}"
if [ -d $image_dir ] && [ ${switch[0]} == "T" ];then
    rm -rf $image_dir
fi
mkdir -p $image_dir


source ~/anaconda3/etc/profile.d/conda.sh


#===========================
# 0. [Setting restore] process
#---------------------------

sed "
s/EXP_NAME/$exp_name/g
s/BATCH_SIZE/$batch_size/g
s/OPTIMIZER/$optimizer/g
s/LOSS/$loss/g
s/SYEAR_FIT/${syear_fit}/g
s/EYEAR_FIT/${eyear_fit}/g
s/SYM_EXC_FIT/${sym_exc_fit}/g
s/EYM_EXC_FIT/${eym_exc_fit}/g
s/SYEAR_PRED/${syear_pred}/g
s/EYEAR_PRED/${eyear_pred}/g
s/SYM_EXC_PRED/${sym_exc_pred}/g
s/EYM_EXC_PRED/${eym_exc_pred}/g" settings.txt > settings_${exp_name}.txt

mv settings_${exp_name}.txt ${dir_scrp}



#====================
# 1. [Fit] process
#--------------------
if [ ${switch[0]} == "T" ];then
# activate [venv2]
conda activate venv2

sed "
s/EXP_NAME/$exp_name/g
s@GPU_CHOICE@$gpu_choice@g
s/BATCH_SIZE/$batch_size/g
s/OPTIMIZER/$optimizer/g
s/LOSS/$loss/g
s@CKPT_DIR@${ckpt_dir}@g
s@LOG_DIR@${log_dir}@g
s/TF_AGMT/${TF_AGMT}/g
s/SYEAR_FIT/${syear_fit}/g
s/EYEAR_FIT/${eyear_fit}/g
s/SYM_EXC_FIT/${sym_exc_fit}/g
s/EYM_EXC_FIT/${eym_exc_fit}/g" step_00_fit_def.py > ${exp_name}_00_fit_def.py

python ${exp_name}_00_fit_def.py
mv ${exp_name}_00_fit_def.py ${dir_scrp}
cp deep_*.py ${dir_scrp}

conda deactivate
fi


#====================
# 2. [Pred] process
#--------------------
if [ ${switch[1]} == "T" ];then
# activate [venv2]
conda activate venv2

# - Get chosen checkpoint path including the name of checkpoint
ckpt_path=`ls $ckpt_dir/cp-${epoch_num_d4}* | head -n 1 | awk -F'.' '{print "."$2"."$3}'`
# - Get checkpoint name
#cp_name=`echo ${cp_path} | awk -F"/" '{print $NF}'`
ckpt_name="cp_${epoch_num_d4}"

echo $ckpt_path
sed "
s/EXP_NAME/$exp_name/g
s@GPU_CHOICE@$gpu_choice@g
s/OPTIMIZER/$optimizer/g
s/LOSS/$loss/g
s@CKPT_PATH@${ckpt_path}@g
s/CKPT_NAME/${ckpt_name}/g
s@PRED_DIR@${pred_dir}@g
s/SYEAR_PRED/${syear_pred}/g
s/EYEAR_PRED/${eyear_pred}/g
s/SYM_EXC_PRED/${sym_exc_pred}/g
s/EYM_EXC_PRED/${eym_exc_pred}/g" step_01_pred_def.py > ${exp_name}_01_pred_def.py

python ${exp_name}_01_pred_def.py
mv ${exp_name}_01_pred_def.py ${dir_scrp}
cp deep_*.py ${dir_scrp}

conda deactivate
fi


#====================
# 3. [Plot] process
#--------------------
if [ ${switch[2]} == "T" ];then
# activate [ncl]
conda activate ncl

# - Get checkpoint name
ckpt_name="cp_${epoch_num_d4}"

sed "
s/EXP_NAME/$exp_name/g
s/CKPT_NAME/${ckpt_name}/g
s@PRED_DIR@${pred_dir}@g
s@IMAGE_DIR@${image_dir}@g" step_02_draw_results.ncl > ${exp_name}_02_draw_results.ncl

ncl ${exp_name}_02_draw_results.ncl
mv ${exp_name}_02_draw_results.ncl ${dir_scrp}

conda deactivate
fi








