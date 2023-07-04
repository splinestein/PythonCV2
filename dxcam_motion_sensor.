import dxcam
import cv2
from time import time

"""
NOTE: This doesn't work well if your GPU is under load. 
Check out mss_screen_motion.py for better overall performance.
"""

camera = dxcam.create()
camera.start(target_fps=120)

object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=3, detectShadows=False
)

scale_percentage = 50
loop_time = time()

while True:
    frame = camera.get_latest_frame()

    cv2.imshow(
        'Movement Detector',
        object_detector.apply(
            cv2.resize(
                frame,
                (
                    int((frame.shape[1] * scale_percentage) * 0.01),
                    int((frame.shape[0] * scale_percentage) * 0.01),
                ),
                interpolation=cv2.INTER_AREA,
            ),
        ),
    )

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

camera.stop()
