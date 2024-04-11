#!/usr/bin/env bash

python train.py \
  --name gta2nwpu_$(date +"%Y%m%d-%H%M%S") \
  --dataset data/gta.yaml \
  --dataset_t data/nwpu.yaml \
  --model yolov8s \
  --imgsz 640 \
  --batch 8 \
  --epochs 100 \
  --device 0 \
  --skip-ins-loss
