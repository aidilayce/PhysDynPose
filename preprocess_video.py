import time
import os
import argparse
import shutil

def move_images_to_rgb_subfolder(image_folder):
    # Define the subfolder path
    rgb_folder = os.path.join(image_folder, 'rgb')
    
    # Create the 'rgb' subfolder if it doesn't exist
    if not os.path.exists(rgb_folder):
        os.makedirs(rgb_folder)
    
    # List all image files in the folder (supports .png, .jpg, .jpeg)
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Move each image to the 'rgb' subfolder
    for image in images:
        source_path = os.path.join(image_folder, image)
        destination_path = os.path.join(rgb_folder, image)
        
        # Move the file
        shutil.move(source_path, destination_path)

    print(f"Moved {len(images)} images to {rgb_folder}.")

def generate_rgb_txt(image_folder, output_file="rgb.txt", fps=25):
    # List all image files in the folder, sorted by filename
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])

    # Get the current time as the starting timestamp
    start_time = time.time()

    # Generate timestamps based on fps (frames per second)
    timestamps = [start_time + i / fps for i in range(len(images))]

    output_file = str(image_folder)[:-3] + 'rgb.txt'

    # Create the rgb.txt file
    with open(output_file, "w") as f:
        f.write("# color images\n")
        f.write("# timestamp filename\n")
        
        for i, (ts, image) in enumerate(zip(timestamps, images)):
            # Assuming images are located directly in the folder, adjust if necessary
            filename = f"rgb/{image}"
            f.write(f"{ts:.6f} {filename}\n")

    print(f"rgb.txt generated with {len(images)} entries.")

# Usage example:
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='mov', help='folder path')
args = parser.parse_args()
path = args.path

move_images_to_rgb_subfolder(path)
sub_path  = path + '/' + 'rgb'
generate_rgb_txt(sub_path)
