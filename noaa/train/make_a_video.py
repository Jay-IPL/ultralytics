import cv2
import os
from tqdm import tqdm 

folder = '/home/jiemei/Documents/ultralytics/noaa/test_data'
save_folder = '/home/jiemei/Documents/ultralytics/noaa/test_video'
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
