import numpy as np
import cv2
import time
from mean_shift import flow_cluster


help_message = '''
USAGE: optical_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
'''
count = 0
color = np.random.randint(0, 255, (100, 3))


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.polylines(vis, lines, 0, (0, 255, 0))
    # for (x1, y1), (x2, y2) in lines:
    #     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == '__main__':
    import sys

    print(help_message)
    try:
        fn = sys.argv[1]
    except:
        fn = "1.mp4"

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(fn)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_cam = cv2.VideoWriter("cam.mp4", fourcc, 30, (w, h))
    out_flow1 = cv2.VideoWriter("flow1.mp4", fourcc, 30, (w, h))
    # out_hsv = cv2.VideoWriter("hsv.mp4", fourcc, 30, (w, h))
    # out_glitch= cv2.VideoWriter("glitch.mp4", fourcc, 30, (w, h))
    # out_flow2 = cv2.VideoWriter("flow2.mp4", fourcc, 30, (w, h))

    ret, prev = cap.read()
    out_cam.write(prev)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prevgray, mask=None, **feature_params)
    show_hsv = True
    show_glitch = True
    cur_glitch = prev.copy()
    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev)

    while True:
        ret, img = cap.read()
        if not ret:
            break
        out_cam.write(img)
        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow1 = None
        flow1 = cv2.calcOpticalFlowFarneback(prev=prevgray, next=gray, flow=flow1, pyr_scale=0.5, levels=5, winsize=15,
                                             iterations=3, poly_n=7, poly_sigma=1.5, flags=10)
        # 开始计算聚类中心
        flow_cluster(flow1)
        # PyrLK Optical Flow
        # p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=prevgray, nextImg=gray, prevPts=p0, nextPts=None, **lk_params)

        # Select good point
        # if p1 is not None:
        #     good_new = p1[st==1]
        #     good_old = p0[st==1]
        #     # draw the tracks
        # p0 = p1
        # for i, (new, old) in enumerate(zip(good_new, good_old)):
        #     a, b = new.ravel()
        #     c, d = old.ravel()
        #     mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        #     frame = cv2.circle(img.copy(), (int(a), int(b)), 5, color[i].tolist(), -1)
        # flow2 = cv2.add(frame, mask)
        # cv2.imshow('frame', flow2)
        # out_flow2.write(flow2)
        mask = np.zeros_like(img)

        prevgray = gray
        img_flow1 = draw_flow(gray, flow1)
        cv2.imshow('flow', img_flow1)
        out_flow1.write(img_flow1)
        # if show_hsv:
        #     gray1 = cv2.cvtColor(draw_hsv(flow1), cv2.COLOR_BGR2GRAY)
        #     thresh = cv2.threshold(gray1, 25, 255, cv2.THRESH_BINARY)[1]
        #     thresh = cv2.dilate(thresh, None, iterations=2)
        #     (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        #     # loop over the contours
        #     for c in cnts:
        #         # if the contour is too small, ignore it
        #         (x, y, w, h) = cv2.boundingRect(c)
        #         if 100 < w < 900 and 100 < h < 680:
        #             cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)
        #             cv2.putText(vis, str(time.time()), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        #
        #     cv2.imshow('Image', vis)
        #     out_hsv.write(vis)
        # if show_glitch:
        #     cur_glitch = warp_flow(cur_glitch, flow1)
        #     cv2.imshow('glitch', cur_glitch)
        #     out_glitch.write(cur_glitch)
        #     cur_glitch = img.copy()
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        # if ch == ord('1'):
        #     show_hsv = not show_hsv
        #     print('HSV flow visualization is', ['off', 'on'][show_hsv])
        # if ch == ord('2'):
        #     show_glitch = not show_glitch
        #     if show_glitch:
        #         cur_glitch = img.copy()
        #     print('glitch is', ['off', 'on'][show_glitch])

    cv2.destroyAllWindows()