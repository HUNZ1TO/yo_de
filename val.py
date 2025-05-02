import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/voc-C2f-DCNV2/weights/best.pt')
    model.val(#data='tt100k/tt100k.yaml',
              data='datasets\mydate.yaml',
              split='val',
              imgsz=640,
              batch=16,
              device='1',
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )