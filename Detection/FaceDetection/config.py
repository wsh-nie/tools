import argparse


def getConfig():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument('--yolo-type', type=str, default='yolov5l', choices=['yolov5s', 'yolov5m', 'yolov5l'],
                        help="yolo type")
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--objThreshold', default=0.3, type=float, help='object confidence')
    args = parser.parse_args()
    return args
