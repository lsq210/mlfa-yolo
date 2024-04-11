import argparse
import json
from os import path
import os
from ultralytics.models import YOLO
from ultralytics.engine.results import Results

from utils import seed_everything

SEED = 20000108

def train(args):
  print('args:', args)
  seed_everything(SEED)
  model = YOLO(args.weights)
  res: list[Results] = model.predict(args.datasets)
  if not path.exists(args.output):
    os.makedirs(args.output)
  with open(args.anno_file, 'r') as f:
    anno = json.load(f)
  img_map = { i['file_name']: i['id'] for i in anno['images'] }
  resultJson = []
  if not path.exists(path.join(args.output, 'images')):
    os.mkdir(path.join(args.output, 'images'))
  for r in res:
    r.save(path.join(args.output, 'images', r.path.rsplit('/', 1)[-1]))
    for bbox in r.boxes:
      xywh = bbox.xywh.cpu().numpy()[0].tolist()
      resultJson.append({
        'image_id': img_map[r.path.rsplit('/', 1)[-1]],
        'category_id': int(bbox.cls.cpu().numpy()[0]) + 1,
        'bbox': [xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[2], xywh[3]],
        'score': float(bbox.conf.cpu().numpy()[0]),
      })
  with open(path.join(args.output, 'result.json'), 'w') as f:
    json.dump(resultJson, f, indent=2)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Multi-Level Feature Alignment YOLO')
  parser.add_argument(
    "--weights",
    required=True,
    metavar="FILE",
    help="path to model weights file",
    type=str,
  )
  parser.add_argument(
    "--datasets",
    required=True,
    metavar="FILE",
    help="path to testing data set config file",
    type=str,
  )
  parser.add_argument(
    "--anno_file",
    required=True,
    metavar="FILE",
    help="path to annotation file",
    type=str,
  )
  parser.add_argument(
    "--output",
    required=True,
    metavar="FILE",
    help="path to output directory",
    type=str,
  )
  train(parser.parse_args())
