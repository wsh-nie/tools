import time

import cv2
import argparse
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class YoloV5Face:
    def __init__(self, yolo_type, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        anchors = [[4, 5, 8, 10, 13, 16], [23, 29, 43, 55, 73, 105], [146, 217, 231, 300, 335, 433]]
        num_classes = 1
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5 + 10
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNet(yolo_type + '-face.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.

        confidences = []
        boxes = []
        landmarks = []
        for detection in outs:
            confidence = detection[15]
            # if confidence > self.confThreshold and detection[4] > self.objThreshold:
            if detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        result_boxes = []
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            result_boxes.append([left, top, left + width, top + height])

        return result_boxes

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True,
                                     crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        # inference output
        outs[..., [0, 1, 2, 3, 4, 15]] = 1 / (1 + np.exp(-outs[..., [0, 1, 2, 3, 4, 15]]))  ###sigmoid
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)

            g_i = np.tile(self.grid[i], (self.na, 1))
            a_g_i = np.repeat(self.anchor_grid[i], h * w, axis=0)
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + g_i) * int(
                self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * a_g_i

            row_ind += length
        return outs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_type', type=str, default='yolov5m', choices=['yolov5s', 'yolov5m', 'yolov5l'],
                        help="yolo type")
    parser.add_argument("--imgpath", type=str, default='selfie.jpg', help="image path")
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--objThreshold', default=0.3, type=float, help='object confidence')
    args = parser.parse_args()

    yolonet = YoloV5Face(args.yolo_type, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold,
                         objThreshold=args.objThreshold)
    srcimg = cv2.imread(args.imgpath)
    start = time.time()
    dets = yolonet.detect(srcimg)
    print("fps is :", (time.time() - start))
    srcimg = yolonet.postprocess(srcimg, dets)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()