import cv2
import os
from ultralytics import YOLO
from deep_sort.deepsortor import Deepsortor
import numpy as np
import time
import torch
from tqdm import tqdm


"""
测试单个序列的追踪图片，生成追踪视频并记录追踪数据
"""

# 更换成自定义数据集的种类
COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

# 需要测试的追踪文件夹路径
test_dir = "MOT16/train"
# 保存txt文件的文件夹路径
result_txt_dir = "data/trackers/mot_challenge/MOT16-train/MOT/data"

"""
递归创建保留追踪数据的文件夹
"""
def judge_path(path, d=1):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)
    else:
        if d == 1:
            path = path + '({})'.format(d)
        else:
            path = path.replace('({})'.format(d - 1), '({})'.format(d))
        d = d + 1
        path = judge_path(path, d)
    return path

"""
模型权重
"""
weights_path = "runs/trains/ym/weights/best.pt"

tracker_name = "Deepsort"
DEEPSORT_CONFIG_PATH = "deep_sort/configs/deep_sort.yaml"

result_txt_dir = os.path.join(result_txt_dir, tracker_name)
result_txt_dir = judge_path(result_txt_dir)

img_w = 1920
img_h = 1080

if __name__ == '__main__':

    # 加载YOLO模型
    yolo = YOLO(weights_path)

    # 读取测试文件夹
    ffs = [file for file in os.listdir(test_dir)]
    ffs.sort()
    video_num = 0 # 已经处理过的序列数量
    FPS = 0 # 总的执行时间，不包含画框
    for ff in ffs: #遍历得到每一个序列的文件夹名
        # 每次初始化一个新的追踪器对象！！！
        tracker = Deepsortor(configFile=DEEPSORT_CONFIG_PATH)
        # 初始化参数
        frame_number = 0
        print("The tracking file is ",ff)
        video_txt = os.path.join(result_txt_dir,ff+".txt") #对应视频文件名的txt文件路径
        if os.path.exists(video_txt):
            os.remove(video_txt)
        video_path = os.path.join(test_dir,ff,"img1")
        # 读取视频路径中的每一张图片
        files = [file for file in os.listdir(video_path)]
        files.sort() #将图片从头开始排序，保证从第一张开始读取图片
        total_frames = len(files) #视频帧数=文件夹中照片数量
        # 读取文件夹中的图像文件
        for file in tqdm(files):
            frame_number += 1
            im0 = cv2.imread(os.path.join(video_path,file))

            t1 = time.time()  # 开始时间
            dets = yolo.predict(source=im0)

            bbox_xywh = []
            confs = []

            for det in dets:  # 每一个检测目标框 torch
                boxes = det.boxes.xyxy
                confs = det.boxes.conf
                for box in boxes:
                    obj = [
                        int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2),
                        box[2] - box[0], box[3] - box[1]
                    ]

                    bbox_xywh.append(obj)
                xywhs = torch.Tensor(bbox_xywh)

            print(f"当前帧号: {frame_number}/{total_frames}")

            im, obj_bboxes = tracker.update(xywhs, confs, im0)  # --> (x, y, x, y, id, conf, cls, ind)

            t2 = time.time()  # 单个序列的FPS，只包含推理时间（没有画框)
            FPS += (t2 - t1)

            for (x1, y1, x2, y2, cls, id) in obj_bboxes:
                bbox_top_x = x1
                bbox_top_y = y1
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                with open(video_txt, 'a+') as f:
                    f.write(('%g ' * 10 + '\n') % (frame_number, id, bbox_top_x,
                                                   bbox_top_y, bbox_w, bbox_h, -1, -1, -1, -1))
        video_num += total_frames

    print("average fps: ", video_num/FPS)







