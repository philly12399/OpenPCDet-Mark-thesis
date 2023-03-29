#!/bin/bash -i

starting_round=$1
processing_frames=$2
wayside_repo_path="$HOME/wayside-portal"
openpcdet_repo_path="$HOME/OpenPCDet-Mark-thesis"
tracking_data_dir="/mnt/nfs/wayside_team/mark_thesis_data/139-exp-output/tracking-info/"
device="lidar1"
exp_dir="$wayside_repo_path/exp/mark-exp"
pcd_detect_script="$exp_dir/learning-based-detect-from-pcd-files-lidar1.sh"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e

for ((i=starting_round; i<=20; i++)); do
$SCRIPT_DIR/self_training_round_i.sh $i $processing_frames \
$wayside_repo_path $openpcdet_repo_path $tracking_data_dir \
$device $exp_dir $pcd_detect_script
done
