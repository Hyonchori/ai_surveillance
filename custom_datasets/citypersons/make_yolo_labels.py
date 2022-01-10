# make labels for training YOLO
# (class, id(-1), center_x, center_y, width, height) -> (class, center_x, center_y, width, height)

import os
import time

import cv2
import numpy as np
from tqdm import tqdm


def make_yolo_labels(img_root, annot_root, out_root, vis=False, save=False):
    cities = [x for x in os.listdir(annot_root) if os.path.isdir(os.path.join(annot_root, x))]
    for city in cities:
        print(f"\n--- Processing {annot_root} - {city}")
        time.sleep(0.5)
        city_path = os.path.join(annot_root, city)
        out_city_path = os.path.join(out_root, city)
        os.makedirs(out_city_path, exist_ok=True)
        labels = os.listdir(city_path)
        for label in tqdm(labels):
            img_path = os.path.join(img_root, city, label.replace(".txt", ".png"))
            label_path = os.path.join(city_path, label)
            with open(label_path) as f:
                gtboxes = f.readlines()

            if vis:
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                for gtbox in gtboxes:
                    cpwhn = [eval(x.replace("\n", "")) for x in gtbox.split(" ")][2:]
                    draw_cpwhn(cpwhn, w, h, img, color=(255, 0, 0))
                cv2.imshow("img", img)
                cv2.waitKey(0)

            out_label = [x.replace(" -1", "") for x in gtboxes]
            out_label_path = os.path.join(out_city_path, label)
            if save:
                with open(out_label_path, "w") as o:
                    o.write(''.join(out_label))


def cpwhn2xyxy(cpwhn, w, h):
    xyxy = [int((cpwhn[0] - cpwhn[2] / 2) * w),
            int((cpwhn[1] - cpwhn[3] / 2) * h),
            int((cpwhn[0] + cpwhn[2] / 2) * w),
            int((cpwhn[1] + cpwhn[3] / 2) * h)]
    return xyxy


def draw_xyxy(xyxy, img, color=None, thickness=2):
    color = np.random.choice(range(256), size=3).tolist() if color is None else color
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=color, thickness=thickness)


def draw_cpwhn(cpwhn, w, h, img, color=None, thickness=2):
    xyxy = cpwhn2xyxy(cpwhn, w, h)
    draw_xyxy(xyxy, img, color, thickness)


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/citypersons"
    train_img_root = os.path.join(root, "images", "train")
    valid_img_root = os.path.join(root, "images", "val")
    train_annot_root = os.path.join(root, "labels_with_ids", "train")
    valid_annot_root = os.path.join(root, "labels_with_ids", "val")
    train_out_root = os.path.join(root, "labels", "train")
    valid_out_root = os.path.join(root, "labels", "val")

    make_yolo_labels(train_img_root, train_annot_root, train_out_root, vis=False, save=True)
    make_yolo_labels(valid_img_root, valid_annot_root, valid_out_root, vis=False, save=True)
