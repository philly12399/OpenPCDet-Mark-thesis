#!/bin/bash -i

self_training_round=$1
processing_frames=$2
wayside_repo_path=$3
openpcdet_repo_path=$4
tracking_data_dir=$5
device=$6
exp_dir=$7
pcd_detect_script=$8

# tracking_info='/media/mark/adata0/self-training-data/tracking-info/2022-12-29T07:36:35.854785386+08:00/'
# refinement_output='/media/mark/adata0/self-training-data/tracking-refinement-output'
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "self-training-round (must >= 1): $self_training_round"
echo "processing frames: $processing_frames"

set -e

# Run pcd-detect + pcd-tracking
(
ckpt_path="$openpcdet_repo_path/output/kitti_models/pv_rcnn_ST-$device-r$(($self_training_round - 1))/default/ckpt/checkpoint_epoch_80.pth"
python3 "$openpcdet_repo_path/tools/scripts/revise_json.py" revise_json "$exp_dir/config/modules/pcd-detect-lidar1.json5" ckpt "$ckpt_path"
cd "$wayside_repo_path/exp/mark-exp"
$pcd_detect_script $(($processing_frames + 2500))
)


# Get latest directory name
tracking_data_path="$tracking_data_dir/$(ls -t $tracking_data_dir| head -n1)"


# Run module refine-by-track to filter BBoxes by tracking data
(
cd "$wayside_repo_path/rust-bin/refine-by-track"
python3 "$openpcdet_repo_path/tools/scripts/revise_json.py" revise_json ./refine-by-track.json5 input_dir "$tracking_data_path"
python3 "$openpcdet_repo_path/tools/scripts/revise_json.py" revise_json ./refine-by-track.json5 num_output_frames "$processing_frames"
cargo run --release -- -c ./refine-by-track.json5
)

refined_data_path="$tracking_data_path/refined-output/supervisely"

# Run module save-kitti-format to get kitti-format data
# (cd "$wayside_repo_path/rust-bin/save-kitti-format" && \
# cargo run --release -- -c "$wayside_repo_path/rust-bin/save-kitti-format/config-$device.json5" -a "$refined_data_path" -s 1 --keep-raw-points)

# Prepare KITTI training data
(
source "$openpcdet_repo_path/pcdet-env/bin/activate"
kitti_format_dir="$(dirname "$refined_data_path")/kitti-format"
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
