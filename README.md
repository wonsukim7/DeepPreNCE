# DeepPreNCE

1. install CONDA virtual environment.
```
    $ conda env create -f ./env/conda_env_venv2.yaml 
    $ conda env create -f ./env/conda_env_ncl.yaml
```
2. edit main script.

    - For default run (single-input): edit main_run_def.bash
    - For multi-input run: edit main_run_multi.bash

3. run main script.
```bash
    $ ./main_run_def.bash   # for default run
    $ ./main_run_multi.bash # for multi-input run
```
- Files
    - deep_model.py: contains deep-learning based rain rate nowcasting models
    - deep_utils.py: contains utilities for data processing and loss functions, and etc.
    - main_run_def.bash: main script for '*default (single-input)*' run
    - main_run_multi.bash: main script for '*multi-input*' run
    - sample_list_2012-2019_JJAS_RDR_avg1h_1hrs.csv: list for non-trivial rainfall cases during June-September, 2012-2019 over the Korean Peninsula
    - setting.txt: templete file for recording the settings of each experiment
    - step_00_fit_def.py: program for training a model using single-input
    - step_00_fit_multi.py: program for training a model using multi-input
    - step_01_pred_def.py: program for predicting a rain rate distribution using a model trained with single-input
    - step_01_pred_multi.py: program for predicting a rain rate distribution using a model trained with multi-input
    - step_02_draw_results.ncl: program that draws the predicted rain rate distributions through step_01
    

- Directories
    - ./env: contains conda environments
    - ./data: contains preprocessed radar rainrate data
    - ./output/ckpt: output directories for model checkpoint
    - ./output/log: output directories for log (used by tensorboard)
    - ./output/image: output directories for resulted images (outputs from 'step_02')
    - ./output/script: output directories for used scripts
    - ./output/predict: output directories for prediction (outputs from 'step01')



