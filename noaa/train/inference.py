# command: python noaa/train/inference.py --save_tracking_video --inference_folder '/home/jiemei/Documents/ultralytics/noaa/test_data' --confidence_threshold 0.167 --save_folder 'noaa/test_result'

from ultralytics import YOLO
import argparse
import cv2
from tqdm import tqdm
import os
import csv

def parse_option():
    parser = argparse.ArgumentParser('Yolov8 detection & tracking for NOAA fish.', add_help=False)
    parser.add_argument("--inference_folder", type=str, default='/home/jiemei/Documents/ultralytics/noaa/test_data', help='path to your folder of frames waiting to be run with detector.')
    parser.add_argument("--save_folder", type=str, default='noaa/test_result', help='path to your save folder for the detection & tracking results.')
    parser.add_argument("--save_tracking_video", action="store_true", help="Flag to save the tracking video.")
    parser.add_argument("--confidence_threshold", type=float, default=0.167, help='detection confidence threshold')

    args, unparsed = parser.parse_known_args()
    return args

def creat_csv(result_save_dir):
    f = open(result_save_dir+'/tracking_results.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["id", "filename", "xmin", "ymin", "xmax", "ymax", "conf","class", "length", "kept"])

    return f, csv_writer

if __name__ == '__main__':
    args = parse_option()
    
    # Step-1: Load yolov8 trained on NOAA-full dateset.
    model = YOLO('runs/detect/noaa_full(yolov8x)/weights/best.pt')
    f, csv_writer = creat_csv(args.save_folder)

    # Step-2: Run detection and tracking. Get results for all frames.
    results = model.track(source=args.inference_folder,
                          stream=True,  # use stream=True to avoid out of memory.
                          tracker='botsort.yaml', # it has camera motion compensation. or try "bytetrack.yaml"
                          show=False,
                          conf=args.confidence_threshold, # 0.167 is searched during detector training.
                          iou=0.7, # default
                          imgsz=640, # default, also is my trained model input size.
                          verbose=True, # disable showing frames
                          persist=True, 
                        #   epochs=50,
                        #   imgsz=640,
                        #   batch=32*8,
                        #   device=[0,1,2,3,4,5,6,7],
                        #   name='noaa_full(yolov8x)'
                        )

    # Step-3: Save results for each frame.
    for i, r in enumerate(tqdm(results)): # each r is all detections from a single frame.

        # Step-3.1: save tracking demo video, all detections of 5 classes, some boxes are tracked, some are not due to unstable detection/low conf.
        if args.save_tracking_video:
            # Visualize the results on the frame
            vis_img = r.plot()
            w, h = vis_img.shape[1], vis_img.shape[0]

            if i==0:
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'XVID' to other codecs as needed
                frame_rate = 30  # or the frame rate of your source video
                frame_size = (w, h)  # Replace with the size of your frames
                out = cv2.VideoWriter(os.path.join(args.save_folder,'output.mp4'), fourcc, frame_rate, frame_size)

            # Write frame to video
            out.write(vis_img)
        
        # Step-3.2: save tracked boxes into csv file, here 5 classes are tracked.
        if not r.boxes.is_track:
            continue

        for each_box, each_conf, each_cls, each_id in zip(r.boxes.xyxyn, r.boxes.conf, r.boxes.cls,r.boxes.id):
            # Only save fish tracks into csv file.
            # if you want to save all 5 classes into csv file, just remove this if
            if r.names[int(each_cls.item())]=='fish':
                xmin, ymin, xmax, ymax = each_box
                csv_writer.writerow([each_id.item(),os.path.basename(r.path),xmin.item(),ymin.item(), xmax.item(), ymax.item(), each_conf.item(), r.names[int(each_cls.item())], 0, 1])

    # close file writer.
    f.close()