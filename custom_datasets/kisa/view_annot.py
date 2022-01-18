import os
import argparse
import datetime

import cv2
import numpy as np
import xmltodict

from roi import RoI


ROOT = "/media/daton/SAMSUNG/3. 연구개발분야/1. 해외환경(1500개)"
CLASSES = {1: "배회", 2: "침입", 3: "유기", 4: "쓰러짐", 5: "싸움", 6: "방화"}
EVENTS = ["Loitering", "Intrusion", "Abandonment", "FireDetection", "Violence", "Falldown"]


def get_annot(annot_path):
    with open(annot_path) as f:
        annot = xmltodict.parse(f.read())
    return annot


def get_event_area(raw_area):
    event_area = []
    for pt in raw_area:
        if pt not in event_area:
            event_area.append(pt)
    event_area = np.array([[int(x.split(",")[0]), int(x.split(",")[1])] for x in event_area])
    return event_area


def time2hms(time):
    h, m, s = [int(x) for x in time.split(":")]
    return h, m, s


def main(args):
    target_cls = args.target_cls
    target_vid = args.target_vid

    target_cls = CLASSES[target_cls]
    target_dir = os.path.join(ROOT, [x for x in os.listdir(ROOT) if target_cls in x][0])
    target_vids = [x for x in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, x)) and
                   (x.endswith(".mp4")) and (target_vid in x if target_vid is not None else True)]
    target_annots = [x.replace(".mp4", ".xml") for x in target_vids]

    for vid_name, annot_name in zip(target_vids, target_annots):
        print(f"\n--- {vid_name}")
        vid_path = os.path.join(target_dir, vid_name)
        annot_path = os.path.join(target_dir, annot_name)

        vid_cap = cv2.VideoCapture(vid_path)
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        annot = get_annot(annot_path)
        clip = annot["KisaLibraryIndex"]["Library"]["Clip"]
        event = [x for x in clip["Header"].keys() if x in EVENTS]
        event = event[0] if len(event) else "DetectArea"
        raw_area = clip["Header"][event]["Point"]
        event_area = get_event_area(raw_area)
        vid_time = time2hms(clip["Header"]["Duration"])

        alarm = clip["Alarms"]["Alarm"]
        start_time = time2hms(alarm["StartTime"])
        alarm_description = alarm["AlarmDescription"]
        duration = time2hms(alarm["AlarmDuration"])

        vid_time_delta = datetime.timedelta(hours=vid_time[0], minutes=vid_time[1], seconds=vid_time[2])
        start_time_delta = datetime.timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
        duration_delta = datetime.timedelta(hours=duration[0], minutes=duration[1], seconds=duration[2])
        start_frame_rate = start_time_delta / vid_time_delta
        end_frame_rate = (start_time_delta + duration_delta) / vid_time_delta
        start_frame = max(0, int(total_frames * start_frame_rate - fps * 13))
        end_frame = min(total_frames, int(total_frames * end_frame_rate + fps * 3))
        alarm_start_frame = int(total_frames * start_frame_rate)
        alarm_end_frame = int(total_frames * end_frame_rate)
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        roi = RoI(img_size=(height, width, 3), default_roi=event_area, alarm_start_frame=alarm_start_frame)
        while True:
            pos_frame = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos_frame == end_frame:
                vid_cap.release()
                break
            ret, img = vid_cap.read()
            if not ret:
                vid_cap.release()
                break
            roi.imshow(img, pos_frame)
            cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-cls", type=int, default=2)
    parser.add_argument("--target_vid", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
