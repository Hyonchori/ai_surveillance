# Person recognizer: detection(yolox), tracking(byte), action-recognition(spatio-temporal action localization)

import argparse
import sys
import os
from pathlib import Path
import warnings
import time
from collections import deque

import cv2
import mmcv
import torch
import numpy as np

from yolox_byte.yolox.data.data_augment import ValTransform
from yolox_byte.yolox.exp import get_exp as get_yolox_exp
from yolox_byte.yolox.utils import fuse_model, get_model_info, postprocess, vis

from yolox_byte.yolox.tracker.byte_tracker import BYTETracker
from yolox_byte.yolox.utils.visualize import plot_tracking

warnings.filterwarnings("ignore")
FILE = Path(__file__).absolute()
if os.path.join(FILE.parents[0], "custom_lib") not in sys.path:
    sys.path.append(os.path.join(FILE.parents[0], "custom_lib"))
from custom_lib.custom_utils import LOGGER, select_device, increment_path, check_file, colors
from custom_lib.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from custom_lib.names import PERSON_CLASSES

from mmcv import Config as get_stdet_cfg
from custom_lib.stdet import StdetPredictor, get_action_dict, plot_actions


@torch.no_grad()
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

    # Load arguments of Spatio-temporal action detector
    stdet_cfg = get_stdet_cfg.fromfile(opt.stdet_cfg)
    stdet_cfg.merge_from_dict(opt.stdet_cfg_options)
    stdet_img_norm_cfg = stdet_cfg["img_norm_cfg"]
    stdet_weights = opt.stdet_weights
    stdet_imgsz = opt.stdet_imgsz
    stdet_interval = opt.stdet_interval
    stdet_action_score_thr = opt.stdet_action_score_thr
    stdet_action_dict = get_action_dict(opt.stdet_action_list_path)
    stdet_label_map_path = opt.stdet_label_map_path

    # Load general arguments
    source = opt.source
    device = opt.device
    normalize = opt.normalize
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

    # Initialize STDet model
    stdet_model = StdetPredictor(
        config=stdet_cfg,
        checkpoint=stdet_weights,
        device=device,
        score_thr=stdet_action_score_thr,
        label_map_path=stdet_label_map_path
    )

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

    trackers = [BYTETracker(opt) for _ in range(bs)]
    stdet_input_imgs = [deque([], maxlen=8) for _ in range(bs)]
    if device.type != "cpu":
        yolo_model(torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(yolo_model.parameters())))
        stdet_input = {
            "img": [torch.zeros(1, 3, 8, *stdet_imgsz).to(device).type_as(next(stdet_model.model.parameters()))],
            "img_metas": [[{"img_shape": stdet_imgsz}]],
            "proposals": [[torch.tensor([[0, 0, 5, 5]], device=device).type_as(next(stdet_model.model.parameters()))]],
            "return_loss": False
        }
        stdet_model.model(**stdet_input)
    for path, im, im0s, vid_cap, s, resize_params in dataset:
        print("\n---")
        ts = time.time()

        # Image preprocessing
        if not webcam:
            path, im0s = [path], [im0s]
        t1 = time.time()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        if normalize:
            im -= torch.Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(device)
            im /= torch.Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(device)
        t2 = time.time()
        print(f"img preproc: {t2 - t1:.4f}")

        if use_model["yolox"]:
            t1 = time.time()
            yolo_preds = yolo_model(im)
            yolo_preds = postprocess(yolo_preds, yolo_exp.num_classes, yolo_exp.test_conf, yolo_exp.nmsthre, True)
            t2 = time.time()
            print(f"yolo predict: {t2 - t1:.4f}")
        else:
            yolo_preds = []

        for i, yolo_pred in enumerate(yolo_preds):
            p, im0, imv = path[i], im0s[i].copy(), im0s[i].copy()
            p = Path(p)
            save_path = str(save_dir / "video") if is_video_frames else str(save_dir / p.name)

            proposals = []
            if use_model["stdet"]:
                t1 = time.time()
                stdet_input_size = mmcv.rescale_size((im0.shape[1], im0.shape[0]), (stdet_imgsz[0], np.inf))
                if "to_rgb" not in stdet_img_norm_cfg and "to_bgr" in stdet_img_norm_cfg:
                    to_bgr = stdet_img_norm_cfg.pop("to_bgr")
                    stdet_img_norm_cfg["to_rgb"] = to_bgr
                stdet_img_norm_cfg["mean"] = np.array(stdet_img_norm_cfg["mean"])
                stdet_img_norm_cfg["std"] = np.array(stdet_img_norm_cfg["std"])
                stdet_input_img = mmcv.imresize(im0, stdet_input_size).astype(np.float32)
                _ = mmcv.imnormalize_(stdet_input_img, **stdet_img_norm_cfg)
                stdet_input_imgs[i].append(stdet_input_img)
                ratio = (stdet_input_size[0] / im0.shape[1], stdet_input_size[1] / im0.shape[0])
                t2 = time.time()
                print(f"stdet preproc: {t2 - t1:.4f}")

            if use_model["byte"]:
                if len(yolo_pred) > 0:
                    t1 = time.time()
                    _, height, width = im[i].shape
                    online_targets = trackers[i].update(yolo_pred, im0.shape[:2], [height, width])
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                            tlwh[0] = tlwh[0] - resize_params[i][1][0] / resize_params[i][0][0]
                            tlwh[1] = tlwh[1] - resize_params[i][1][1] / resize_params[i][0][1]
                            if use_model["stdet"]:
                                xyxy = np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]) * ratio[0]
                                proposals.append(xyxy)
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                    t2 = time.time()
                    print(f"byte predict: {t2 - t1:.4f}")

            if use_model["stdet"]:
                if len(proposals) > 0:
                    t1 = time.time()
                    proposals = np.stack(proposals)
                    proposals = [[torch.from_numpy(proposals).to(device).float()]]
                    if len(stdet_input_imgs[i]) == 8:
                        imgs = np.stack(stdet_input_imgs[i]).transpose(3, 0, 1, 2)
                        imgs = [torch.from_numpy(imgs).unsqueeze(0).to(device)]
                        img_meta = [[{"img_shape": stdet_input_img.shape[:2]}]]
                        return_loss = False
                        stdet_input = {
                            "img": imgs,
                            "img_metas": img_meta,
                            "proposals": proposals,
                            "return_loss": return_loss
                        }
                        stdet_pred = stdet_model.model(**stdet_input)[0]
                        stdet_result = []
                        for _ in range(proposals[0][0].shape[0]):
                            stdet_result.append([])
                        for class_id in range(len(stdet_pred)):
                            if class_id + 1 not in stdet_model.label_map:
                                continue
                            for bbox_id in range(proposals[0][0].shape[0]):
                                if len(stdet_pred[class_id]) != proposals[0][0].shape[0]:
                                    continue
                                if stdet_pred[class_id][bbox_id, 4] > stdet_model.score_thr:
                                    stdet_result[bbox_id].append((stdet_model.label_map[class_id + 1],
                                                                  stdet_pred[class_id][bbox_id, 4]))
                    t2 = time.time()
                    print(f"stdet: {t2 - t1:.4f}")

            t1 = time.time()
            if show_model["yolox"] and not show_model["byte"]:
                if yolo_preds[i] is None:
                    continue
                yolo_bbox = yolo_preds[i][:, :4]
                yolo_bbox[:, 0] = (yolo_bbox[:, 0] - resize_params[i][1][0]) / resize_params[i][0][0]
                yolo_bbox[:, 1] = (yolo_bbox[:, 1] - resize_params[i][1][1]) / resize_params[i][0][1]
                yolo_bbox[:, 2] = (yolo_bbox[:, 2] - resize_params[i][1][0]) / resize_params[i][0][0]
                yolo_bbox[:, 3] = (yolo_bbox[:, 3] - resize_params[i][1][1]) / resize_params[i][0][1]
                yolo_cls = yolo_preds[i][:, 6]
                yolo_scores = yolo_preds[i][:, 4] * yolo_preds[i][:, 5]
                imv = vis(imv,  yolo_bbox, yolo_scores, yolo_cls, 0.5, PERSON_CLASSES)

            elif show_model["byte"]:
                imv = plot_tracking(imv, online_tlwhs, online_ids)

            if show_model["stdet"]:
                plot_actions(imv, proposals[0][0], stdet_result, ratio, colors, stdet_action_dict)

            if any(show_model.values()):
                cv2.imshow(f"img {i}", imv)
                cv2.waitKey(1)
            t2 = time.time()
            print(f"plot: {t2 - t1:.4f}")

            if save_vid:
                if dataset.mode == "image" and not is_video_frames:
                    cv2.imwrite(save_path, imv)
                elif dataset.mode == "image" and is_video_frames:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        fps, w, h = 30, imv.shape[1], imv.shape[0]
                        save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(imv)
                else:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, imv.shape[1], imv.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(imv)
        tf = time.time()
        print(f"\ttotal: {tf - ts:.4f}")


def parse_opt():
    parser = argparse.ArgumentParser("Person action recognizer")

    # Arguments for YOLOX(main person detector)
    yolo_exp = f"{FILE.parents[0]}/yolox_byte/exps/example/mot/yolox_l_mix_det.py"
    yolo_weights = f"{FILE.parents[0]}/weights/yolox/bytetrack_l_mot17.pth.tar"
    parser.add_argument("--yolo-exp", type=str, default=yolo_exp)
    parser.add_argument("--yolo-name", type=str, default=None)
    parser.add_argument("--yolo-weights", type=str, default=yolo_weights)
    parser.add_argument("--yolo-imgsz", type=int, default=[1280])
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

    # Arguments for STDet(action recognizer)
    stdet_cfg = f"{FILE.parents[0]}/mmaction2/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.py"
    stdet_weights = f"{FILE.parents[0]}/weights/stdet/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom/epoch_20.pth"
    parser.add_argument("--stdet-cfg", type=str, default=stdet_cfg)
    parser.add_argument("--stdet-weights", type=str, default=stdet_weights)
    parser.add_argument("--stdet-imgsz", type=int, default=[256])
    parser.add_argument("--stdet-interval", type=int, default=1)
    parser.add_argument("--stdet-action-score-thr", type=float, default=0.4)
    parser.add_argument("--stdet-action-list-path", default=f"{FILE.parents[0]}/weights/stdet/ava_action_list_v2.2.pbtxt")
    parser.add_argument("--stdet-label-map-path", default=f"{FILE.parents[0]}/mmaction2/tools/data/ava/label_map.txt")
    parser.add_argument("--stdet-cfg-options", default={})

    # General arguments
    source = "/home/daton/Downloads/daton_office_02-people_counting.mp4"
    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    #source = "https://youtu.be/WNIccic_178"
    source = "source_list.txt"
    source = "/media/daton/Data/datasets/MOT17/train/MOT17-04-FRCNN/img1"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--normalize", default=True, action="store_true")
    parser.add_argument("--half", default=True, action="store_true")
    parser.add_argument("--save-dir", type=str, default=f"{FILE.parents[0]}/runs/inference")
    parser.add_argument("--run-name", type=str, default="exp")
    parser.add_argument("--is-video-frames", default=True, action="store_true")
    parser.add_argument("--save-vid", default=True, action="store_true")
    parser.add_argument("--hide-labels", default=False, action="store_true")
    parser.add_argument("--hide-conf", default=False, action="store_true")
    parser.add_argument("--use-model", default={"yolox": True,
                                                "byte": True,
                                                "stdet": False})
    parser.add_argument("--show-model", default={"yolox": True,
                                                 "byte": True,
                                                 "stdet": True})
    opt = parser.parse_args()
    opt.yolo_imgsz *= 2 if len(opt.yolo_imgsz) == 1 else 1
    opt.stdet_imgsz *= 2 if len(opt.stdet_imgsz) == 1 else 1
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

