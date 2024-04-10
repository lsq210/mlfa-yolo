from os import path
from ultralytics.models import YOLO
from functools import partial
from copy import deepcopy

from trainers import MLFATrainer
from utils import parseArgs, seed_everything

SEED = 20000108

def train(args):
    kwargs = {
        'imgsz': args.imgsz,
        'epochs': args.epochs,
        'val': not args.skip_val,
        'workers': args.workers,
        'batch': args.batch,
        'seed': SEED,
        'device': args.device,
    }
    seed_everything(SEED)
    model = YOLO(args.model + '.yaml').load(args.model + '.pt')
    mlfa_trainer = partial(MLFATrainer, target_domain_data_cfg=args.dataset_t)
    model.train(mlfa_trainer, data=args.dataset, name=args.name, patience=0, **deepcopy(kwargs))
    if not args.skip_val:
        model.val(data=args.dataset_t, name=args.name)

if __name__ == '__main__':
    train(parseArgs())
