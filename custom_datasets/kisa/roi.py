import cv2
import numpy as np


class RoI:
    def __init__(self, img_size, default_roi, alarm_start_frame, window_name="roi"):
        self.img_size = img_size
        self.default_roi = default_roi
        self.alarm_start_frame = alarm_start_frame
        self.window_name = window_name

        self.ref_img = np.zeros(self.img_size, np.uint8)
        cv2.namedWindow(self.window_name)

    def imshow(self, img, pos_frame):
        if pos_frame >= self.alarm_start_frame:
            cv2.fillPoly(self.ref_img, [self.default_roi.reshape((-1, 1, 2))], (0, 0, 225))
        else:
            cv2.fillPoly(self.ref_img, [self.default_roi.reshape((-1, 1, 2))], (0, 225, 225))
        result = cv2.addWeighted(img, 1, self.ref_img, 0.5, 0)
        cv2.imshow(self.window_name, result)
        return result
