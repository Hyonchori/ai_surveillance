import sys
import time
import os
from pathlib import Path

import torch
import cv2

from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess, vis

FILE = Path(__file__).absolute()
if os.path.join(FILE.parents[1].as_posix(), "custom_lib") not in sys.path:
    sys.path.append(os.path.join(FILE.parents[1].as_posix(), "custom_lib"))
from names import PERSON_CLASSES


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=PERSON_CLASSES,
        device="cpu",
        fp16=False,
        normalize=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=normalize)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        t3 = time.time()
        img, _ = self.preproc(img, None, self.test_size)
        t4 = time.time()
        print(img.shape)
        print(f"\tpreproc: {t4 - t3:.4}")
        cv2.imshow("preproc", img.transpose(1, 2, 0)[..., ::-1])

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16
        print(img.shape)
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            print(self.confthre, self.nmsthre)
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res
