import cv2
import numpy as np
from time import time
from mss import mss

object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=3, detectShadows=True
)

with mss() as sct:
    while True:
        sc_np = np.asarray(sct.grab(sct.monitors[1]))

        cv2.imshow(
            'Movement Detector',
            object_detector.apply(
                cv2.resize(
                    sc_np,
                    (
                        int((sc_np.shape[1] * 50) * 0.01),
                        int((sc_np.shape[0] * 50) * 0.01),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            ),
        )

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
