import os
import shutil

# Source and destination directories
source_dir = '/media/jiemei/FieldData-AukBay_201807/AukBay_2018/20180808T131848-0800/GO-2400C-PGE+09-88-36'
dest_dir = '/home/jiemei/Documents/ultralytics/noaa/test_data'

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Define a function to determine whether to copy a file
def should_copy(filename, start_id, end_id):
    # Implement your logic here
    if filename.endswith('.jpg'):
        id = int(filename.split('_')[0])
    else:
        return False
    # For example, copy only PNG files:
    return id>start_id and id<end_id

# Iterate over files in the source directory
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    # Check if the file meets your criteria
    if should_copy(filename, start_id=60050, end_id=61000):
        # Copy the file to the destination directory
        shutil.copy(file_path, dest_dir)

print("Frames copied successfully.")
