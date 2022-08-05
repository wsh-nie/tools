import cv2
import numpy as np

color = np.random.randint(0, 255, (1000, 3))


def getFeature(gray):
    orb = cv2.ORB_create(nfeatures=200)
    kp, des = orb.detectAndCompute(gray, None)
    p0 = []
    # Point2f to int
    # Remove redundancy
    mp = []
    for k in kp:
        tp = str(float(int(k.pt[0]))) + str(float(int(k.pt[1])))
        if tp not in mp:
            mp.append(tp)
            p0.append([float(int(k.pt[0])), float(int(k.pt[1]))])
    p0 = np.array(p0, dtype='float32')
    p0 = p0.reshape(-1, 1, 2)
    return p0


if __name__ == '__main__':
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_cam = cv2.VideoWriter("cam.mp4", fourcc, 30, (w, h))
    out_flow = cv2.VideoWriter("flow.mp4", fourcc, 30, (w, h))

    success, prev = cap.read()
    out_cam.write(prev)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    p0 = getFeature(prevgray)

    while True:
        success, frame = cap.read()
        if not success:
            break
        out_cam.write(frame)
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=prevgray, nextImg=cur_gray, prevPts=p0, nextPts=None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # draw the tracks
        p0 = getFeature(gray=cur_gray) # 每一帧都要重新检测关键点
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            flow = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            img = cv2.circle(frame.copy(), (int(a), int(b)), 5, color[i].tolist(), -1)
        # flow = cv2.add(img, mask)
        out_flow.write(flow)
        prevgray = cur_gray
        # mask = np.zeros_like(prev)
        cv2.imshow("s", flow)
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()


