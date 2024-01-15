# command: python noaa/train/detection_inference.py --save_tracking_video --inference_folder '/home/jiemei/Documents/ultralytics/noaa/test_data' --confidence_threshold 0.167 --save_folder 'noaa/test_result'
# command: python noaa/train/inference_detection.py --save_tracking_video --inference_folder '/home/jiemei/Documents/ultralytics/noaa/test_video/output.avi' --confidence_threshold 0.167 --save_folder 'noaa/test_result_video'

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
    f = open(result_save_dir+'/detection_result_with_blank_frames_batch.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame_id", "xmin", "ymin", "xmax", "ymax", "confidence","class"])

    return f, csv_writer

if __name__ == '__main__':
    args = parse_option()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # Step-1: Load yolov8 trained on NOAA-full dateset.
    model = YOLO('runs/detect/noaa_full(yolov8x)/weights/best.pt')
    f, csv_writer = creat_csv(args.save_folder)

    # Step-2: Run detection. Get results for all frames.
    results = model.predict(source=args.inference_folder,
                          stream=True,  # use stream=True to avoid out of memory.
                          show=False,
                          conf=args.confidence_threshold, # 0.167 is searched during detector training.
                          iou=0.7, # default
                          imgsz=640, # default, also is my trained model input size.
                          verbose=False, # disable showing frames
                        #   persist=True, 
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
        
        
        # No detections in this frame, but still keep its info in the csv for later tracking purpose.
        if r.boxes.xyxyn.shape[0] == 0:
            csv_writer.writerow([os.path.basename(r.path), '', '', '', '', 0, ''])
            continue

        # Step-3.2: save all frames including detections/blank frames into csv file, here 5 classes are detected.
        for each_box, each_conf, each_cls in zip(r.boxes.xyxyn, r.boxes.conf, r.boxes.cls):
            # save 5 classes detections into csv file.
            xmin, ymin, xmax, ymax = each_box
            csv_writer.writerow([os.path.basename(r.path),xmin.item(),ymin.item(), xmax.item(), ymax.item(), each_conf.item(), r.names[int(each_cls.item())]])

    # close file writer.
    f.close()