import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/(3)C2f-ghost2-TADDH/weights/best.pt') # select your model.pt path
    model.predict(source='dateset/images/images/test',
                  imgsz=640,
                  project='runs/detect',
                  name='GDI',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )