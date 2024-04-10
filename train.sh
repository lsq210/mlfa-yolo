#!/usr/bin/env bash

python train.py \
  --name gta2nwpu_$(date +"%Y%m%d-%H%M%S") \
  --dataset data/gta.yaml \
  --dataset_t data/nwpu.yaml \
  --model yolov8s \
  --imgsz 320 \
  --batch 2 \
  --epochs 100 \
  --device 0
