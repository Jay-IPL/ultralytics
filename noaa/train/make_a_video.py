# command: python noaa/train/make_a_video.py --folder '/home/jiemei/Documents/ultralytics/noaa/test_data' --save_folder '/home/jiemei/Documents/ultralytics/noaa/test_video'

import cv2
import os
from tqdm import tqdm 
import argparse

def parse_option():
    parser = argparse.ArgumentParser('Yolov8 detection & tracking for NOAA fish.', add_help=False)
    parser.add_argument("--folder", type=str, default='/home/jiemei/Documents/ultralytics/noaa/test_data', help='path to your folder of frames waiting to be combined into a video.')
    parser.add_argument("--save_folder", type=str, default='/home/jiemei/Documents/ultralytics/noaa/test_video', help='path to your save folder for the generated video.')

    args, unparsed = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_option()
    folder = args.folder
    save_folder = args.save_folder

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')


    frame_list = os.listdir(folder)# your frame file paths
    frame_list.sort()

    for id, frame in enumerate(tqdm(frame_list)):
        image = cv2.imread(os.path.join(folder, frame))

        h = image.shape[0]
        w = image.shape[1]

        if id ==0:
            out = cv2.VideoWriter(os.path.join(save_folder, 'output.avi'), fourcc, 30.0, (w, h))
        out.write(image)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
