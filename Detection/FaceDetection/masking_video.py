import os
import time

import cv2
from tqdm import tqdm
from config import *
from yolov5_face import YoloV5Face
import numpy as np


def mosaic(img, area_list, kernel_size):
    kernel_matrix = np.repeat(1.0 / (kernel_size ** 2), kernel_size * kernel_size * 3)
    kernel_matrix = kernel_matrix.reshape((kernel_size, kernel_size, 3))
    for area in area_list:
        lx, ly, rx, ry = area[0], area[1], area[2], area[3]
        for i in range(ly, ry, kernel_size):
            for j in range(lx, rx, kernel_size):
                matrix_a = img[i:i + kernel_size, j:j + kernel_size]
                (y, x, c) = matrix_a.shape
                sum_value = (matrix_a * kernel_matrix[0:y, 0:x]).sum(axis=(0, 1))
                sum_value = sum_value.astype(int)
                sum_value = np.concatenate([sum_value] * (kernel_size * kernel_size), axis=0)
                sum_value = sum_value.reshape((kernel_size, kernel_size, 3))
                img[i:i + y, j:j + x] = sum_value[0:y, 0:x]

    return img


class MaskVideo:
    def __init__(self, net):
        self.net = net

    def process_image(self, image):
        assert type(image) is np.ndarray or os.path.isfile(image), print("Please input correct data")
        if os.path.isfile(image):
            image = cv2.imread(image)
        dets = self.net.detect(image)
        detected_boxes = self.net.postprocess(image, dets)
        return detected_boxes

    def process_video(self, input_file, output_file):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        cap = cv2.VideoCapture(input_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

        for idx in tqdm(range(frame_count)):
            success, frame = cap.read()
            if not success:
                continue

            detected_boxes = self.process_image(frame)
            frame = mosaic(frame, detected_boxes, 15)

            out.write(frame)


if __name__ == "__main__":
    args = getConfig()

    yolo_net = YoloV5Face(args.yolo_type, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold,
                          objThreshold=args.objThreshold)
    masking_video = MaskVideo(net=yolo_net)

    input_path = "./video"
    output_path = "./output"
    video_list = os.listdir(input_path)

    for video in video_list:
        if video.endswith('mp4'):
            video_path = os.path.join(input_path, video)
            filename_extension = video.split('.')
            new_video_path = os.path.join(output_path, filename_extension[0]+"_masked."+filename_extension[1])
            masking_video.process_video(video_path, new_video_path)
