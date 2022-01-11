# make labels for training YOLO
# (frame, id, top_left_x, top_left_y, width, height, confidence, class, visibility)
# -> (class, center_x, center_y, width, height)

import os
import time

import cv2
import numpy as np
from tqdm import tqdm


def split_and_make_labels(root_dir, train_split=False, random_state=42, vis=False, save=False):
    vid_list = list(set(["-".join(x.split("-")[:-1]) for x in os.listdir(root_dir) if "MOT" in x]))
    if train_split:
        np.random.seed(random_state)
        train_idx = np.random.choice(range(len(vid_list)), len(vid_list) // 2 + 1, replace=False).tolist()
        val_idx = [x for x in range(len(vid_list)) if x not in train_idx]
    else:
        train_idx = list(range(len(vid_list)))
        val_idx = []
    train_vid_list = [vid_list[x] for x in sorted(train_idx)]
    val_vid_list = [vid_list[x] for x in val_idx]

    out_img_dir = os.path.join(root_dir)
    out_label_dir = os.path.join(root_dir)
    make_yolo_labels(root_dir, train_vid_list, out_img_dir, out_label_dir, verbose="train", vis=vis, save=save)
    make_yolo_labels(root_dir, val_vid_list, out_img_dir, out_label_dir, verbose="val", vis=vis, save=save)


def make_yolo_labels(root_dir, vid_lsit, out_img_dir, out_label_dir, verbose=None, vis=False, save=False):
    out_img_dir = os.path.join(out_img_dir, verbose, "images") if verbose is not None else out_img_dir
    out_label_dir = os.path.join(out_label_dir, verbose, "labels") if verbose is not None else out_label_dir
    if save:
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)
    for vid in vid_lsit:
        vid_dir = os.path.join(root_dir, [x for x in os.listdir(root_dir) if vid in x][0])
        img_dir = os.path.join(vid_dir, "img1")
        img_names = os.listdir(img_dir)
        gt_path = os.path.join(vid_dir, "gt", "gt.txt")
        gt = get_gt(gt_path)
        print(f"\n--- {vid}")
        time.sleep(0.5)
        for img_name in tqdm(img_names):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            img_num = int(img_name.replace(".jpg", ""))
            gtboxes = gt[img_num]
            label = ""
            for gtbox in gtboxes:
                cls = gtbox[0]
                xywh = gtbox[1:]
                if vis:
                    draw_xywh(xywh, img)
                xyxye = xywh2xyxye(xywh, w, h)
                if check_xyxy(xyxye):
                    cpwhn = xyxy2cpwhn(xyxye, w, h)
                    label += f"0 {cpwhn[0]} {cpwhn[1]} {cpwhn[2]} {cpwhn[3]}\n"
            if vis:
                cv2.imshow("img", img)
                cv2.waitKey(0)
            if save:
                out_img_path = os.path.join(out_img_dir, f"{vid}-{img_name}")
                out_label_path = os.path.join(out_label_dir, f"{vid}-{img_name}".replace(".jpg", ".txt"))
                cv2.imwrite(out_img_path, img)
                with open(out_label_path, "w") as o:
                    o.write(label)


def get_gt(gt_path, vis_thr=0.13):
    gt = {}
    target_clss = [1, 2, 7]  # 1: pedestrian, 2: person on vehicle, 3: static person
    with open(gt_path) as f:
        gt_lines = f.readlines()
    for gt_line in gt_lines:
        frame, _,  x, y, w, h, conf, cls, vis = gt_line.split(",")
        if int(cls) not in target_clss:
            continue
        if float(vis) < vis_thr:
            continue
        if int(frame) in gt:
            gt[int(frame)].append([0, int(x), int(y), int(w), int(h)])
        else:
            gt[int(frame)] = [[0, int(x), int(y), int(w), int(h)]]
    return gt


def xywh2xyxy(xywh):
    xyxy = [int(xywh[0]),
            int(xywh[1]),
            int(xywh[0] + xywh[2]),
            int(xywh[1] + xywh[3])]
    return xyxy


def draw_xyxy(xyxy, img, color=None, thickness=2):
    color = np.random.choice(range(256), size=3).tolist() if color is None else color
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=color, thickness=thickness)


def draw_xywh(xywh, img, color=None, thickness=2):
    xyxy = xywh2xyxy(xywh)
    draw_xyxy(xyxy, img, color, thickness)


def xywh2xyxye(box, w, h):
    xyxy = xywh2xyxy(box)
    xyxy_e = [max(0, xyxy[0]),
              max(0, xyxy[1]),
              min(w, xyxy[2]),
              min(h, xyxy[3])]
    return xyxy_e


def check_xyxy(xyxy):
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    if w > 0 and h > 0:
        return True
    else:
        return False


def xyxy2cpwhn(xyxy, w, h):
    cpwhn = [round(((xyxy[0] + xyxy[2]) / 2 / w), 6),
             round(((xyxy[1] + xyxy[3]) / 2 / h), 6),
             round(((xyxy[2] - xyxy[0]) / w), 6),
             round(((xyxy[3] - xyxy[1]) / h), 6)]
    return cpwhn


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/MOT17"
    train_root = os.path.join(root, "train")

    split_and_make_labels(train_root, train_split=True, vis=False, save=True)

