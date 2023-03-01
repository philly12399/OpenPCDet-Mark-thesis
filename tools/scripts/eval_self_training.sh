starting_round=$1
ending_round=$2
ckpt_dir="/mnt/nfs/wayside_team/mark_thesis_data/139-openpcdet-output/kitti_models"
openpcdet_repo_path="$HOME/OpenPCDet-Mark-thesis"
device="lidar1"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e

(
source "$openpcdet_repo_path/pcdet-env/bin/activate"
cd $openpcdet_repo_path/tools
cfg_file=$openpcdet_repo_path/tools/cfgs/kitti_models/pv_rcnn_wayside-gt-dataset-lidar1.yaml
for ((i=starting_round; i<=ending_round; i++)); do
ckpt_path="$ckpt_dir/pv_rcnn_ST-$device-r$i/default/ckpt/checkpoint_epoch_80.pth"
python test.py --cfg_file $cfg_file --ckpt $ckpt_path --extra_tag ST-$device-r$i --eval_range
done
deactivate
)