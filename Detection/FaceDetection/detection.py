import os
import cv2
from config import *
import numpy as np
from yolov5_face import YoloV5Face
from tqdm import tqdm


class DetectFace:
    def __init__(self, net):
        self.net = net

    def process_image(self, image):
        assert type(image) is np.ndarray or os.path.isfile(image), print("Please input correct data")
        if os.path.isfile(image):
            image = cv2.imread(image)
        dets = self.net.detect(image)
        detected_boxes = self.net.postprocess(image, dets)
        return detected_boxes


if __name__ == "__main__":
    args = getConfig()

    yolo_net = YoloV5Face(args.yolo_type, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold,
                          objThreshold=args.objThreshold)
    detect_face = DetectFace(net=yolo_net)

    input_path = './'
    image_file_list = os.listdir(input_path)
    for image in tqdm(image_file_list):
        if image.endswith('.jpg'):
            image_path = os.path.join(input_path, image)
            src_img = cv2.imread(image_path)

            detected_boxes = detect_face.process(src_img)
            print(len(detected_boxes))
            for box in detected_boxes:
                cv2.rectangle(src_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
            winName = 'Deep learning object detection in OpenCV'
            cv2.namedWindow(winName, 0)
            cv2.imshow(winName, src_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()