# Person recognizer: detection(yolox), tracking(byte), action-recognition(spatio-temporal action localization)

import argparse
import sys
import os
from pathlib import Path
import warnings

import cv2
import torch

from yolox_byte.yolox.data.data_augment import ValTransform
from yolox_byte.yolox.exp import get_exp as get_yolox_exp
from yolox_byte.yolox.utils import fuse_model, get_model_info, postprocess, vis

from yolox_byte.yolox.tracker.byte_tracker import BYTETracker
from yolox_byte.yolox.utils.visualize import plot_tracking

warnings.filterwarnings("ignore")
FILE = Path(__file__).absolute()
if os.path.join(FILE.parents[0], "custom_lib") not in sys.path:
    sys.path.append(os.path.join(FILE.parents[0], "custom_lib"))
from custom_lib.custom_utils import LOGGER, select_device, increment_path, check_file
from custom_lib.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from custom_lib.names import PERSON_CLASSES
from custom_lib.predictor import Predictor


def main(opt):
    # Load arguments of YOLOv5(main detector)
    yolo_exp = opt.yolo_exp
    yolo_name = opt.yolo_name
    yolo_weights = opt.yolo_weights
    yolo_imgsz = opt.yolo_imgsz
    yolo_conf_thr = opt.yolo_conf_thr
    yolo_iou_thr = opt.yolo_iou_thr
    yolo_fuse = opt.yolo_fuse

    # Load arguments of Bytetracker(tracker)
    track_thresh = opt.track_thresh
    track_buffer = opt.track_buffer
    match_thresh = opt.match_thresh
    aspect_ratio_thresh = opt.aspect_ratio_thresh
    min_box_area = opt.min_box_area
    mot20 = opt.mot20

    # Load general arguments
    source = opt.source
    device = opt.device
    half = opt.half
    save_dir = opt.save_dir
    run_name = opt.run_name
    is_video_frames = opt.is_video_frames
    save_vid = opt.save_vid
    hide_labels = opt.hide_labels
    hide_conf = opt.hide_conf
    use_model = opt.use_model
    show_model = {key: use_model[key] & opt.show_model[key] for key in use_model}

    device = select_device(device)
    save_dir = increment_path(Path(save_dir) / run_name, exist_ok=False)
    if save_vid:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize YOLOX model
    yolo_exp = get_yolox_exp(yolo_exp, yolo_name)
    yolo_model = yolo_exp.get_model().to(device)
    yolo_model.eval()
    if os.path.isfile(yolo_weights):
        ckpt = torch.load(yolo_weights)
        yolo_model.load_state_dict(ckpt["model"])
    if yolo_fuse:
        yolo_model = fuse_model(yolo_model)
    print(f"YOLOX model is loaded... {get_model_info(yolo_model, yolo_imgsz)}")

    if half:
        yolo_model.half()

    person_detector = Predictor(yolo_model, yolo_exp, PERSON_CLASSES, device, half, normalize=True)
    tracker = BYTETracker(opt)

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    if webcam:
        dataset = LoadStreams(source, img_size=yolo_imgsz)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=yolo_imgsz)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs
    for path, im, im0s, vid_cap, s in dataset:
        print("\n---")
        print(im.shape)


def parse_opt():
    parser = argparse.ArgumentParser("Person action recognizer")

    # Arguments for YOLOX(main person detector)
    yolo_exp = f"{FILE.parents[0]}/yolox_byte/exps/example/mot/yolox_l_mix_det.py"
    yolo_weights = f"{FILE.parents[0]}/weights/yolox/bytetrack_l_mot17.pth.tar"
    parser.add_argument("--yolo-exp", type=str, default=yolo_exp)
    parser.add_argument("--yolo-name", type=str, default=None)
    parser.add_argument("--yolo-weights", type=str, default=yolo_weights)
    parser.add_argument("--yolo-imgsz", type=int, default=[640])
    parser.add_argument("--yolo-conf-thr", type=float, default=0.01)
    parser.add_argument("--yolo-iou-thr", type=float,default=0.7)
    parser.add_argument("--yolo-fuse", default=True, action="store_true")

    # Arguments for ByteTracker(person tracker)
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--aspect-ratio-thresh", type=float, default=1.6)
    parser.add_argument("--min-box-area", type=float, default=10)
    parser.add_argument("--mot20", default=False, action="store_true")

    # General arguments
    source = "/home/daton/Downloads/daton_office_02-people_counting.mp4"
    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    source = "https://youtu.be/WNIccic_178"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--half", default=True, action="store_true")
    parser.add_argument("--save-dir", type=str, default=f"{FILE.parents[0]}/runs/inference")
    parser.add_argument("--run-name", type=str, default="exp")
    parser.add_argument("--is-video-frames", default=True, action="store_true")
    parser.add_argument("--save-vid", default=True, action="store_true")
    parser.add_argument("--hide-labels", default=False, action="store_true")
    parser.add_argument("--hide-conf", default=False, action="store_true")
    parser.add_argument("--use-model", default={"yolox": True,
                                                "byte": True,
                                                "stder": True})
    parser.add_argument("--show-model", default={"yolox": True,
                                                "byte": True,
                                                "stder": True})
    opt = parser.parse_args()
    opt.yolo_imgsz *= 2 if len(opt.yolo_imgsz) == 1 else 1
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

