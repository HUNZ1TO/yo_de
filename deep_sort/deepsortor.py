from deep_sort.deep_sort import DeepSort
import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            # self.update(yaml.load(fo.read()))
            self.update(yaml.load(fo.read(), Loader=yaml.FullLoader))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)

def get_config(config_file=None):
    return YamlParser(config_file=config_file)

class Deepsortor:
    def __init__(self, configFile):
        cfg = get_config()
        cfg.merge_from_file(configFile)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    def update(self, xywhs, confss, image):
        bboxes2draw = []
        # Pass detections to deepsort
        outputs = self.deepsort.update(xywhs, confss, image)

        for value in list(outputs):
            x1, y1, x2, y2, track_id = value
            bboxes2draw.append(
                (x1, y1, x2, y2, '', track_id)
            )

        return image, bboxes2draw
