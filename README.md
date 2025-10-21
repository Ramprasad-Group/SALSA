# Extract Line Chart Data
wrapper for LineFormer, ChartDete, and improvement over the code from https://github.com/tdsone/extract-line-chart-data


# Setup
step #1) clone this repo with submodules: 'git clone --recurse-submodules https://github.com/Ramprasad-Group/AEM_OCR'
step #2) create a conda environment using environment.yml: 'conda env create -f environment.yml'
step #3) install openmmlab: 'CUDA_TAG=cu117 bash src/scripts/install_openmmlab.sh'
step #5) install LineFormer and ChartDete repos 'bash src/scripts/install_plugins.sh' -->
step #6) Run this line to patch the config file in ChartDete: 'python src/scripts/patch_config.py extern/ChartDete/configs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'

# How to use
step #1) place the desired figures in scr/input
step #2) run src/plextract/main.py