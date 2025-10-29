######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
######## Set Experiment Configuration ###########
exp_config="$exp_dir/fm_contentstyle.json"
exp_name="fm_contentstyle"

####### Train Model ###########
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --main_process_port 14555 --mixed_precision="bf16" \
    "${work_dir}"/bins/vc/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug