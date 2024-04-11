#!/usr/bin/env bash

training_name=gta2nwpu_20240410-175502
data_dir=/home/featurize/data/nwpu_car/images/test
anno_file=/home/featurize/data/nwpu_car/coco/test.json

python test.py \
  --weights runs/detect/${training_name}/weights/best.pt \
  --dataset $data_dir \
  --anno_file $anno_file \
  --output runs/detect/${training_name}/testing
