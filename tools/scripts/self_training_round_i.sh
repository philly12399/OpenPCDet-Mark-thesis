#!/bin/bash -i

self_training_round=$1
processing_frames=$2
wayside_repo_path=$3
openpcdet_repo_path=$4
datawriter_output_path=$5
device=$6
exp_dir=$7
pcd_detect_script=$8

# tracking_info='/media/mark/adata0/self-training-data/tracking-info/2022-12-29T07:36:35.854785386+08:00/'
# refinement_output='/media/mark/adata0/self-training-data/tracking-refinement-output'
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "self-training-round (must >= 1): $self_training_round"
echo "processing frames: $processing_frames"

set -e

# Run pcd-detect + pcd-tracking + data-writer
(
export RUST_LOG=info
ckpt_path="$openpcdet_repo_path/output/kitti_models/pv_rcnn_ST-$device-r$(($self_training_round - 1))/default/ckpt/checkpoint_epoch_80.pth"
python3 "$openpcdet_repo_path/tools/scripts/revise_json.py" revise_json "$exp_dir/config/modules/pcd-detect-lidar1.json5" ckpt "$ckpt_path"
cd "$wayside_repo_path/exp/mark-exp"
$pcd_detect_script $processing_frames
)


# Get latest directory name
supervisely_ann_path="$datawriter_output_path/$(ls -t $datawriter_output_path| head -n1)/0/pcd/dataset/ann"

# Pipeline: save-kitti-format -> kitti-format-data
# Get latest directory name
(cd "$wayside_repo_path/rust-bin/save-kitti-format" && \
cargo run --release -- -c "$wayside_repo_path/rust-bin/save-kitti-format/config-$device.json5" -a "$supervisely_ann_path" -s 1)

# Prepare KITTI training data
(
source "$openpcdet_repo_path/pcdet-env/bin/activate"
kitti_format_dir="$(dirname "$supervisely_ann_path")/kitti-format"
mkdir -p "$openpcdet_repo_path/data/ST-$device-r$self_training_round"
cd "$openpcdet_repo_path/data/ST-$device-r$self_training_round" 
ln -Ts "$kitti_format_dir" training 
mkdir -p ImageSets 
echo Writing train.txt ... 
python3 "$openpcdet_repo_path/tools/scripts/write_index_file.py" write_index_file ImageSets training/label_2 2
cd "$openpcdet_repo_path"
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml "data/ST-$device-r$self_training_round"
deactivate
)

# PV-RCNN Training
(
source "$openpcdet_repo_path/pcdet-env/bin/activate"
cd "$openpcdet_repo_path/tools/"
python add_two_training_cfg.py add_cfgs "../data/ST-$device-r$self_training_round"
python train.py --cfg_file "cfgs/kitti_models/pv_rcnn_ST-$device-r$self_training_round.yaml" 
deactivate
)
