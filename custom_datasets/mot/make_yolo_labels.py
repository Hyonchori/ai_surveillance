# make labels for training YOLO
# (frame, id, top_left_x, top_left_y, width, height, confidence, class, visibility)
# -> (class, center_x, center_y, width, height)

import os
import time

import cv2
import numpy as np
from tqdm import tqdm


def make_yolo_labels(root_dir, vis=False, save=False):
    print(os.listdir(root_dir))


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/MOT17"
    train_root = os.path.join(root, "train")

    make_yolo_labels(train_root)
