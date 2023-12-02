from ultralytics import YOLO
import argparse

def parse_option():
    parser = argparse.ArgumentParser('Yolov8 detection & tracking for NOAA fish.', add_help=False)
    parser.add_argument("--inference_folder", type=str, default='frames', help='path to your folder of frames waiting to be run with detector.')
    parser.add_argument("--save_folder", type=str, default='results', help='path to your save folder for the detection results.')

    args, unparsed = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_option()
    
    # Step-1: Load a NOAA-full dateset trained yolov8
    model = YOLO('/work/u1436961/JieMei/ultralytics/runs/detect/noaa_full(yolov8x)/weights/best.pt')

    # Step-2: 
    results = model.track(source='/work/u1436961/JieMei/datasets/yolov8_rail_data_full_mltiprocess/images/val',
                          stream=True,  # use stream=True to avoid out of memory.
                          tracker='botsort.yaml', # it has camera motion compensation. or try "bytetrack.yaml"
                          show=True,
                          conf=0.167, # searched during detector training.
                          iou=0.7, # default
                          imgsz=640, # default, also is my trained model input size.
                        #   epochs=50,
                        #   imgsz=640,
                        #   batch=32*8,
                        #   device=[0,1,2,3,4,5,6,7],
                        #   name='noaa_full(yolov8x)'
                        )


    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        from IPython import embed
        embed()