import argparse
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import cv2

FILE = Path(__file__).absolute()
sys.path.append(os.path.join(FILE.parents[0].as_posix(), "yolor"))
from yolor.utils.google_utils import attempt_load
from yolor.utils.datasets import LoadStreams, LoadImages
from yolor.utils.general import (check_img_size, non_max_suppression,
                                 apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolor.utils.plots import plot_one_box
from yolor.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolor.models.models import Darknet

sys.path.append(os.path.join(FILE.parents[0].as_posix(), "ByteTrack"))
from ByteTrack.yolox.data.data_augment import ValTransform, preproc
from ByteTrack.yolox.exp import get_exp as get_yolox_exp
from ByteTrack.yolox.utils import fuse_model, get_model_info, postprocess
from ByteTrack.yolox.utils.visualize import plot_tracking
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer

from custom_utils import increment_path



def main(opt):
    yolor_weights = opt.yolor_weights
    yolor_cfg = opt.yolor_cfg
    yolor_imgsz = opt.yolor_imgsz
    yolor_conf_thr = opt.yolor_conf_thr
    yolor_iou_thr = opt.yolor_iou_thr

    yolox_weights = opt.yolox_weights
    yolox_fuse = opt.yolox_fuse

    track_thr = opt.track_thresh
    track_buffer = opt.track_buffer
    track_match_thr = opt.match_thresh
    track_aspect_ratio_thr = opt.aspect_ratio_thr
    track_min_box_area = opt.min_box_area
    tracK_mot20 = opt.mot20

    source = opt.source
    device = opt.device
    dir_path = opt.dir_path
    run_name = opt.run_name
    is_video_frames = opt.is_video_frames
    show_cls = opt.show_cls
    save_vid = opt.save_vid
    hide_labels = opt.hide_labels
    hide_conf = opt.hide_conf
    use_model = opt.use_model
    show_model = {key: use_model[key] & opt.show_model[key] for key in use_model}

    webcam = source == "0" or source.startswith("rtsp") or source.startswith("http") or source.endswith(".txt")

    device = select_device(device)
    save_dir = increment_path(Path(dir_path) / run_name, exist_ok=False)
    if save_vid:
        save_dir.mkdir(parents=True, exist_ok=True)

    yolor_model = Darknet(yolor_cfg, yolor_imgsz).cuda()
    yolor_model.load_state_dict(torch.load(yolor_weights, map_location=device)["model"])
    yolor_model.to(device).eval()

    tracker = BYTETracker(opt, frame_rate=30)

    if webcam:
        datasets = LoadStreams(source, img_size=yolor_imgsz)
        bs = len(datasets)
    else:
        datasets = LoadImages(source, img_size=yolor_imgsz, auto_size=64)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs





def parse_opt():
    parser = argparse.ArgumentParser()

    # YOLOR
    yolor_weights = f"{FILE.parents[0]}/weights/yolor/yolor_p6.pt"
    parser.add_argument("--yolor-weights", type=str, default=yolor_weights)
    parser.add_argument("--yolor-cfg", type=str, default="yolor/cfg/yolor_p6.cfg")
    parser.add_argument("--yolor-imgsz", type=int, default=1280)
    parser.add_argument("--yolor-conf-thr", type=float, default=0.4)
    parser.add_argument("--yolor-iou-thr", type=float, default=0.5)

    # YOLOX
    yolox_weights = f"{FILE.parents[0]}/weights/yolox/yolox_l.pth"
    parser.add_argument("--yolox-weights", type=str, default=yolox_weights)
    parser.add_argument("--yolox-fuse", type=bool, default=True)

    # Byte tracker
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--aspect-ratio-thr", type=float, default=1.6)
    parser.add_argument("--min-box-area", type=float, default=10)
    parser.add_argument("--mot20", type=bool, default=False)

    source = "0"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dir-path", default="runs/inference")
    parser.add_argument("--run-name", default="exp")
    parser.add_argument("--is-video-frames", type=bool, default=True)  # use when process images from video
    parser.add_argument("--show-cls", type=int, default=[0])
    parser.add_argument("--save-vid", type=bool, default=True)
    parser.add_argument("--hide-labels", type=bool, default=False)
    parser.add_argument("--hide-conf", type=bool, default=False)
    parser.add_argument("--use-model", type=dict,
                        default={"yolor": True,
                                 "yolox": False,
                                 "tracker": False})
    parser.add_argument("--show-model", type=dict,
                        default={"yolor": True,
                                 "yolox": False,
                                 "tracker": False})

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
