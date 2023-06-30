#!/bin/bash -i

round=$1

set -e

service rabbitmq-server restart
source ../pcdet-env/bin/activate
bash scripts/dist_train.sh 2 --cfg_file cfgs/kitti_models/pointrcnn_ST-lidar1-r$round.yaml --batch_size 8
deactivate
cd scripts
./self_training_pointrcnn.sh $(($round + 1)) 5000
