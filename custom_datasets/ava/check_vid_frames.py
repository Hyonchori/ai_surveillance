import os

import cv2


if __name__ == "__main__":
    vid_dir = "/media/daton/Data/datasets/ava/videos_15min"
    vid_list = os.listdir(vid_dir)
    '''for vid in vid_list:
        vid_path = os.path.join(vid_dir, vid)
        cap = cv2.VideoCapture(vid_path)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"\n--- {vid}")
        print(total_frames)
        while True:
            ret, img = cap.read()
            if not ret:
                break
        print(cap.get(cv2.CAP_PROP_POS_FRAMES))'''

    vid = "Vmef_8MY46w.mkv"
    cap = cv2.VideoCapture(os.path.join(vid_dir, vid))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, img = cap.read()
        pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print("\n---")
        print(pos_frame)
        if pos_frame == total_frames:
            break
