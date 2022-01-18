import torch
import cv2
import numpy as np

from mmcv.runner import load_checkpoint
from mmaction.models import build_detector


class StdetPredictor:
    """Wrapper for MMAction2 spatio-temporal action models.

    Args:
        config (str): Path to stdet config.
        ckpt (str): Path to stdet checkpoint.
        device (str): CPU/CUDA device option.
        score_thr (float): The threshold of human action score.
        label_map_path (str): Path to label map file. The format for each line
            is `{class_id}: {class_name}`.
    """

    def __init__(self, config, checkpoint, device, score_thr, label_map_path):
        self.score_thr = score_thr

        # load model
        config.model.backbone.pretrained = None
        model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location=device)
        model.to(device)
        model.eval()
        self.model = model
        self.device = device

        # init label map, aka class_id to class_name dict
        with open(label_map_path) as f:
            lines = f.readlines()
        lines = [x.strip().split(': ') for x in lines]
        self.label_map = {int(x[0]): x[1] for x in lines}
        try:
            if config['data']['train']['custom_classes'] is not None:
                self.label_map = {
                    id + 1: self.label_map[cls]
                    for id, cls in enumerate(config['data']['train']
                                             ['custom_classes'])
                }
        except KeyError:
            pass

    def predict(self, task):
        """Spatio-temporval Action Detection model inference."""
        # No need to do inference if no one in keyframe
        if len(task.stdet_bboxes) == 0:
            return task

        with torch.no_grad():
            result = self.model(**task.get_model_inputs(self.device))[0]

        # pack results of human detector and stdet
        preds = []
        for _ in range(task.stdet_bboxes.shape[0]):
            preds.append([])
        for class_id in range(len(result)):
            if class_id + 1 not in self.label_map:
                continue
            for bbox_id in range(task.stdet_bboxes.shape[0]):
                if result[class_id][bbox_id, 4] > self.score_thr:
                    preds[bbox_id].append((self.label_map[class_id + 1],
                                           result[class_id][bbox_id, 4]))

        # update task
        # `preds` is `list[list[tuple]]`. The outter brackets indicate
        # different bboxes and the intter brackets indicate different action
        # results for the same bbox. tuple contains `class_name` and `score`.
        task.add_action_preds(preds)

        return task


def get_action_dict(action_list_path):
    with open(action_list_path) as f:
        data = f.read().replace("\n", "")
        data_split = data.split("}")

        action_dict = {"PERSON_MOVEMENT": [],
                       "OBJECT_MANIPULATION": [],
                       "PERSON_INTERACTION": []}
        for ds in data_split:
            ds = ds.replace("name", "'name'").replace("label_id", ",'label_id'").replace("label_type", ",'label_type'")
            ds = ds.replace("PERSON_MOVEMENT", "'PERSON_MOVEMENT'")
            ds = ds.replace("OBJECT_MANIPULATION", "'OBJECT_MANIPULATION'")
            ds = ds.replace("PERSON_INTERACTION", "'PERSON_INTERACTION'") + "}"
            ds = ds.replace("label {", "{")
            if len(ds) < 5:
                continue
            tmp_dict = eval(ds)
            action_dict[tmp_dict["label_type"]].append(tmp_dict["name"])
    return action_dict


def plot_action_label(img, actions, st, colors, verbose):
    location = (0 + st[0], 18 + verbose * 18 + st[1])
    diag0 = (location[0] + 20, location[1] - 14)
    diag1 = (location[0], location[1] + 2)
    cv2.rectangle(img, diag0, diag1, colors(verbose + 110, True), -1)
    if len(actions) > 0:
        for (label, score) in actions:
            text = f"{label} {score:.2f}"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            textwidth = textsize[0]
            diag0 = (location[0] + textwidth, location[1] - 14)
            cv2.rectangle(img, diag0, diag1, colors(verbose + 110, True), -1)
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, 1)
            break


def plot_actions(img, bboxes, actions, ratio, colors, action_dict):
    for bbox, action in zip(bboxes, actions):
        bbox = bbox.cpu().numpy() / ratio[0]
        bbox = bbox.astype(np.int64)
        st, ed = tuple(bbox[:2]), tuple(bbox[2:])
        action = sorted(action, key=lambda x: x[1], reverse=True)
        action_pm = list(filter(lambda x: x[0] in action_dict["PERSON_MOVEMENT"], action))
        action_om = list(filter(lambda x: x[0] in action_dict["OBJECT_MANIPULATION"], action))
        action_pi = list(filter(lambda x: x[0] in action_dict["PERSON_INTERACTION"], action))
        plot_action_label(img, action_pm, st, colors, 0)
        plot_action_label(img, action_om, st, colors, 1)
        plot_action_label(img, action_pi, st, colors, 2)