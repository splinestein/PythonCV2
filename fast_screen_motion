# Fastest motion detector at 1440p, averages at 20+ FPS.
from mss import mss
import cv2
import numpy as np
from time import time
#bounding_box1 = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440} # top left
bbox = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440} # top left
object_detector = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=5, detectShadows=False)

def scaler(percentage):
    scale_percent = percentage # percent of original size
    width = int(sc_np.shape[1] * scale_percent / 100)
    height = int(sc_np.shape[0] * scale_percent / 100)
    return (width, height)

with mss() as sct:
    loop_time = time()
    for i in range(2000):
        shot = sct.grab(bbox)
        sc_np = np.array(shot)
        dim = scaler(50)
        resized = cv2.resize(sc_np, dim, interpolation = cv2.INTER_AREA)
        mask = object_detector.apply(resized)
        dim = scaler(80)
        resized = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Frame1", resized)
        print('FPS {}'.format(1 / (time() - loop_time)))
        loop_time = time()
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
