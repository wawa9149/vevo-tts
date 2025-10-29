######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/ar_synthesis.json"
exp_name="ar_synthesis"

######## Optional: Resume Setting ###########
resume=${RESUME:-false}  # 환경변수로 설정하지 않으면 기본은 false
resume_type="resume"
checkpoint_path="/app/data/vevo/models/ar_synthesis/checkpoint/epoch-0001_step-0005000_loss-6.492178"

######## Train Model ###########
if [ "$resume" = true ]; then
    echo "▶ Resuming from checkpoint: $checkpoint_path"
    CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --main_process_port 14557 --mixed_precision="bf16" \
        "${work_dir}"/bins/vc/train.py \
        --config="$exp_config" \
        --exp_name="$exp_name" \
        --log_level debug \
        --resume \
        --resume_type "$resume_type" \
        --checkpoint_path "$checkpoint_path"
else
    echo "▶ Training from scratch"
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --main_process_port 14557 --mixed_precision="bf16" \
        "${work_dir}"/bins/vc/train.py \
        --config="$exp_config" \
        --exp_name="$exp_name" \
        --log_level debug
fi
