#!/bin/bash -i

self_training_round=$1
cfg_file=$2
wayside_repo_path=$3
openpcdet_repo_path=$4
tracking_data_dir=$5
device=$6
exp_dir=$7
pcd_detect_script=$8

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "self-training-round (must >= 1): $self_training_round"
echo "cfg_file: $cfg_file"
echo "wayside_repo_path: $wayside_repo_path"
echo "openpcdet_repo_path: $openpcdet_repo_path"
set -e
cfg_name=$(basename $cfg_file .yaml)

if [ -d "$openpcdet_repo_path/data/ST-r$self_training_round" ]; then
    (
    source "$openpcdet_repo_path/pcdet-env/bin/activate"
    cd "$openpcdet_repo_path/tools/"
    python train.py --cfg_file "cfgs/kitti_models/$cfg_name.yaml" --extra_tag ST-r$self_training_round \
    --batch_size 6 \
    --set DATA_CONFIG.DATA_PATH "../data/ST-r$self_training_round"
    deactivate
    )
else
    # Run pcd-detect + pcd-tracking
    (
    cfg_name=$(basename $cfg_file .yaml)
    ckpt_path="$openpcdet_repo_path/output/kitti_models/$cfg_name/ST-r$(($self_training_round - 1))/ckpt/checkpoint_epoch_80.pth"
    python3 "$openpcdet_repo_path/tools/scripts/revise_json.py" revise_json "$exp_dir/config/modules/pcd-detect-pvrcnn-nuscenes.json5" ckpt "$ckpt_path"
    cd "$wayside_repo_path/exp/mark-exp"
    $pcd_detect_script
    )


    # Get latest directory name
    tracking_data_path="$tracking_data_dir/$(ls -t $tracking_data_dir| head -n1)"


    # Run module refine-by-track to filter BBoxes by tracking data
    (
    cd "$wayside_repo_path/rust-bin/refine-by-track"
    python3 "$openpcdet_repo_path/tools/scripts/revise_json.py" revise_json ./model-refiner-nuscenes.json5 input_dir "$tracking_data_path"
    cargo run --release -- -c ./model-refiner-nuscenes.json5
    )

    refined_data_path="$tracking_data_path/refined-output/kitti-format"

    # Prepare KITTI training data
    (
    source "$openpcdet_repo_path/pcdet-env/bin/activate"
    mkdir -p "$openpcdet_repo_path/data/ST-r$self_training_round"
    cd "$openpcdet_repo_path/data/ST-r$self_training_round" 
    ln -Tfs "$refined_data_path" training 
    mkdir -p ImageSets 
    echo Writing train.txt ... 
    ln -s "$openpcdet_repo_path/nuscenes-preprocessing/params/train_val.txt" ImageSets/train.txt
    ln -s "$openpcdet_repo_path/nuscenes-preprocessing/params/val.txt" ImageSets/val.txt
    # python3 "$openpcdet_repo_path/tools/scripts/write_index_file.py" write_index_file ImageSets training/label_2 2
    cd "$openpcdet_repo_path"
    python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml "data/ST-r$self_training_round"
    deactivate
    )

    # PV-RCNN Training
    (
    source "$openpcdet_repo_path/pcdet-env/bin/activate"
    cd "$openpcdet_repo_path/tools/"
    python train.py --cfg_file "cfgs/kitti_models/$cfg_name.yaml" --extra_tag ST-r$self_training_round \
    --batch_size 6 \
    --set DATA_CONFIG.DATA_PATH "../data/ST-r$self_training_round"
    deactivate
    )
fi




