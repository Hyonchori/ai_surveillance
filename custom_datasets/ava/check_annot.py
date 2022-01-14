import os
import argparse

import cv2
import pandas as pd
import numpy as np


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def main(args):
    annot_file = args.annot_file
    label_map = args.label_map
    vid_dir = args.vid_dir
    vid_name = args.vid_name

    vid_names = os.listdir(vid_dir)
    if vid_name is not None:
        target_vids = [x for x in vid_names if vid_name in x]
    else:
        target_vids = vid_names

    annot = pd.read_csv(annot_file, names=["vid_name", "frame", "x1", "y1", "x2", "y2", "action", "id"])
    with open(label_map) as f:
        label_map = [eval(x.split(": ")[-1][:-1]) for x in f.readlines() if "name" in x]

    for target_vid_name in target_vids:
        tmp_idx = annot["vid_name"] == target_vid_name.replace(".mkv", "").replace(".mp4", "").replace(".webm", "")
        tmp_annot = annot[tmp_idx]
        start_sec = tmp_annot["frame"].iloc[0]
        end_sec = tmp_annot["frame"].iloc[-1]

        vid_path = os.path.join(vid_dir, target_vid_name)
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int((start_sec - 2) * fps)
        end_frame = int(end_sec * fps)
        print(start_frame, end_frame, fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while True:
            tmp_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if tmp_frame == end_frame:
                break

            tmp_sec = int(tmp_frame / fps)
            ret, img = cap.read()
            h, w, _ = img.shape
            bboxes_idx = tmp_annot["frame"] == tmp_sec
            bboxes = tmp_annot[bboxes_idx][["x1", "y1", "x2", "y2", "action", "id"]].values
            for bbox in bboxes:
                xyxy = xyxyn2xyxy(bbox[:4], w, h)
                action = label_map[int(bbox[4] - 1)]
                id = bbox[5]
                draw_xyxy(xyxy, img, color=colors(id, True))
            cv2.imshow("img", img)
            cv2.waitKey(0)



def xyxyn2xyxy(xyxyn, w, h):
    xyxy = [int(xyxyn[0] * w),
            int(xyxyn[1] * h),
            int(xyxyn[2] * w),
            int(xyxyn[3] * h)]
    return xyxy


def draw_xyxy(xyxy, img, color=None, thickness=2):
    color = np.random.choice(range(256), size=3).tolist() if color is None else color
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=color, thickness=thickness)



def parse_args():
    parser = argparse.ArgumentParser()

    ava_root = "/media/daton/Data/datasets/ava"
    annot_file = os.path.join(ava_root, "annotations", "ava_train_v2.2.csv")
    label_map = os.path.join(ava_root, "annotations", "ava_action_list_v2.2.pbtxt")
    vid_dir = os.path.join(ava_root, "videos")
    vid_name = "-IELREHX_js"

    parser.add_argument("--annot-file", default=annot_file)
    parser.add_argument("--label-map", default=label_map)
    parser.add_argument("--vid-dir", default=vid_dir)
    parser.add_argument("--vid-name", default=vid_name)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
