import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Lock, Pool

import cv2
import numpy as np


def extract_frame(vid_item):
    full_path, vid_path, vid_id, method, task, report_file = vid_item
    if "/" in vid_path:
        act_name = osp.basename(osp.dirname(vid_path))
        out_full_path = osp.join(args.out_dir, act_name)
    else:
        out_full_path = args.out_dir

    run_success = -1
    if task == "rgb":
        if args.use_opencv:
            vid_name = osp.splitext(osp.basename(vid_path))[0]
            out_full_path = osp.join(out_full_path, vid_name)
            raw_full_path = full_path.replace("videos_15min", "videos")
            raw_cap = cv2.VideoCapture(raw_full_path)
            raw_fps = raw_cap.get(cv2.CAP_PROP_FPS)
            target_fps = 30

            if not osp.isdir(out_full_path):
                os.makedirs(out_full_path)
            cap = cv2.VideoCapture(full_path)
            fps_gap = target_fps - raw_fps
            cnt = 0
            tik = 0
            if "mp4" not in full_path:
                while True:
                    ret, img = cap.read()
                    if not ret:
                        run_success = 0
                        break
                    cv2.imwrite(f"{out_full_path}/img_{cnt + 1:05d}.jpg", img)
                    cnt += 1
                    tik += fps_gap
                    if tik >= raw_fps:
                        cv2.imwrite(f"{out_full_path}/img_{cnt + 1:05d}.jpg", img)
                        tik -= raw_fps
                        cnt += 1
            else:
                while True:
                    ret, img = cap.read()
                    if not ret:
                        run_success = 0
                        break
                    cv2.imwrite(f"{out_full_path}/img_{cnt + 1:05d}.jpg", img)
                    cnt += 1

    if run_success == 0:
        print(f'{task} {vid_id} {vid_path} {method} done')
        sys.stdout.flush()

        lock.acquire()
        with open(report_file, 'a') as f:
            line = full_path + '\n'
            f.write(line)
        lock.release()
    else:
        print(f'{task} {vid_id} {vid_path} {method} got something wrong')
        sys.stdout.flush()


def init(lock_):
    global lock
    lock = lock_


def main(args):
    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    if args.mixed_ext:
        print('Extension of videos is mixed')
        fullpath_list = glob.glob(args.src_dir + '/*' * args.level)
    else:
        print('Extension of videos: ', args.ext)
        fullpath_list = glob.glob(args.src_dir + '/*' * args.level + '.' +
                                  args.ext)
    print('Total number of videos found: ', len(fullpath_list))

    if args.level == 1:
        vid_list = list(map(osp.basename, fullpath_list))

    pool = Pool(args.num_worker, initializer=init, initargs=(lock, ))
    pool.map(
        extract_frame,
        zip(fullpath_list, vid_list, range(len(vid_list)),
            len(vid_list) * [args.flow_type],
            len(vid_list) * [args.task],
            len(vid_list) * [args.report_file])
    )
    pool.close()
    pool.join()
    #print(vid_list)
    #extract_frame((fullpath_list[5], vid_list[5], 0, args.flow_type, args.task, args.report_file))


def parse_args():
    parser = argparse.ArgumentParser()

    src_dir = "/media/daton/Data/datasets/ava/videos_15min"
    out_dir = "/media/daton/Data/datasets/ava/rawframes_re"
    parser.add_argument("--src-dir", type=str, default=src_dir)
    parser.add_argument("--out-dir", type=str, default=out_dir)
    parser.add_argument("--task", default="rgb", choices=["rgb", "flow", "both"])
    parser.add_argument("--level", default=1, choices=[1, 2])
    parser.add_argument("--num-worker", default=8)
    parser.add_argument("--flow-type", default=None, choices=[None, "tvl1", 'warp_tvl1', 'farn', 'brox'])
    parser.add_argument("--out-format", default="jpg", choices=["jpg", "h5", "png"])
    parser.add_argument("--ext", default="avi")
    parser.add_argument("--mixed-ext", default=True)
    parser.add_argument("--new-width", default=0)
    parser.add_argument("--new-height", default=0)
    parser.add_argument("--new-short", default=0)
    parser.add_argument("--num-gpu", default=1)
    parser.add_argument("--resume", default=False)
    parser.add_argument("--use-opencv", default=True)
    parser.add_argument("--input-frames", default=False)
    parser.add_argument("--report-file", default="build_report.txt")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    lock = Lock()
    main(args)
