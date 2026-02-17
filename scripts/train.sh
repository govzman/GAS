#! /bin/bash

config_name="configs/sd/coco.yaml"
loss_type="GS"
NFE=5
train_size=4

python main.py \
    --config=$config_name \
    --loss_type=$loss_type \
    --student_step=$NFE \
    --train_size=$train_size
