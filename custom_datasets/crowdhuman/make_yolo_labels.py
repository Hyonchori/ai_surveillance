# make labels for training YOLO: (class, center_x, center_y, width, height)

import os
import time

import cv2
import numpy as np
from tqdm import tqdm


def make_yolo_labels(root_dir, annot_file, out_name="labels", vis=False, save=False):
    img_dir = os.path.join(root_dir, "Images")
    label_dir = os.path.join(root_dir, out_name)
    os.makedirs(label_dir, exist_ok=True)
    with open(annot_file, "r") as f:
        annot = f.readlines()
        num_body = 0
        print(f"\n--- Processing {annot_file}")
        time.sleep(0.5)
        for tmp_annot in tqdm(annot):
            tmp_annot = eval(tmp_annot)
            img_name = tmp_annot["ID"]
            img_path = os.path.join(img_dir, img_name + ".jpg")
            label_path = os.path.join(label_dir, img_name + ".txt")
            img = cv2.imread(img_path)
            im0 = img.copy()
            h, w, _ = img.shape

            label = ""
            gtboxes = tmp_annot["gtboxes"]
            gtboxes = [x for x in gtboxes if x["tag"] == "person"]
            ignores = []
            for gtbox in gtboxes:
                ignore = False
                if "ignore" in gtbox["extra"]:
                    if gtbox["extra"]["ignore"] == 1:
                        ignore = True
                        #continue
                        ignores.append(ignore)

                fbox = gtbox["fbox"]
                fxyxy_e = xywh2xyxye(fbox, w, h)

                if check_xyxy(fxyxy_e):
                    if vis:
                        if not ignore:
                            draw_xyxy(fxyxy_e, img, (255, 0, 0))
                        else:
                            draw_xyxy(fxyxy_e, img, (0, 0, 255))
                    body_cpwh = xyxy2cpwhn(fxyxy_e, w, h)
                    label += f"0 {body_cpwh[0]} {body_cpwh[1]} {body_cpwh[2]} {body_cpwh[3]}\n"
                    num_body += 1

            if save:
                with open(label_path, "w") as s:
                    s.write(label)
            if vis:
                if any(ignores):
                    joint_img = np.hstack((img, im0))
                    cv2.imshow("img", joint_img)
                    cv2.waitKey(0)


def xywh2xyxy(xywh):
    xyxy = [int(xywh[0]),
            int(xywh[1]),
            int(xywh[0] + xywh[2]),
            int(xywh[1] + xywh[3])]
    return xyxy


def draw_xyxy(xyxy, img, color=None, thickness=2):
    color = np.random.choice(range(256), size=3).tolist() if color is None else color
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=color, thickness=thickness)


def draw_xywh(cpwh, img, color=None, thickness=2):
    xyxy = xywh2xyxy(cpwh)
    draw_xyxy(xyxy, img, color, thickness)


def check_xyxy(xyxy):
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    if w > 0 and h > 0:
        return True
    else:
        return False


def xywh2xyxye(box, w, h):
    xyxy = xywh2xyxy(box)
    xyxy_e = [max(0, xyxy[0]),
              max(0, xyxy[1]),
              min(w, xyxy[2]),
              min(h, xyxy[3])]
    return xyxy_e


def xyxy2cpwhn(xyxy, w, h):
    cpwhn = [round(((xyxy[0] + xyxy[2]) / 2 / w), 6),
             round(((xyxy[1] + xyxy[3]) / 2 / h), 6),
             round(((xyxy[2] - xyxy[0]) / w), 6),
             round(((xyxy[3] - xyxy[1]) / h), 6)]
    return cpwhn


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/crowdhuman"
    train_root = os.path.join(root, "Crowdhuman_train")
    valid_root = os.path.join(root, "Crowdhuman_val")
    train_annot = os.path.join(root, "annotation_train.odgt")
    valid_annot = os.path.join(root, "annotation_val.odgt")

    make_yolo_labels(train_root, train_annot, vis=False, save=True)
    make_yolo_labels(valid_root, valid_annot, vis=False, save=True)
