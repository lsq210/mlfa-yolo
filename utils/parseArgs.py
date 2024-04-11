
import argparse

def parseArgs():
    parser = argparse.ArgumentParser(description='Multi-Level Feature Alignment YOLO')
    parser.add_argument(
        "--name",
        default="gta2nwpu",
        help="name for this training run",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="data/gta.yaml",
        metavar="FILE",
        help="path to source domain data set config file",
        type=str,
    )
    parser.add_argument(
        "--dataset_t",
        default="data/nwpu.yaml",
        metavar="FILE",
        help="path to target domain data set config file",
        type=str,
    )
    parser.add_argument(
        "--model",
        default="yolov8s",
        help="model name in https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models",
        type=str,
    )
    parser.add_argument(
        "--skip-feat-loss",
        dest="skip_feat_loss",
        action="store_true",
    )
    parser.add_argument(
        "--skip-ins-loss",
        dest="skip_ins_loss",
        action="store_true",
    )

    parser.add_argument('--imgsz', default=640, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--device', default=0, type=int, help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument(
        "--skip-val",
        dest="skip_val",
        action="store_true",
    )
    return parser.parse_args()
