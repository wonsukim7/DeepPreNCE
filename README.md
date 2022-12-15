# DeepRainE

1. install CONDA virtual environment.
```
    $ conda env create -f conda_env_venv2.yaml 
    $ conda env create -f cond_env_ncl.yaml
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



- Directories
    
    > env: contains conda environments
    
    > data: contains preprocessed radar rainrate data
    
    > output/ckpt: output directories for model checkpoint
    
    > output/log: output directories for log (used by tensorboard)
    
    > output/image: output directories for resulted images (outputs from 'step_02')
    
    > output/script: output directories for used scripts
    
    > output/predict: output directories for prediction (outputs from 'step01')



