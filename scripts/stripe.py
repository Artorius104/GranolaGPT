import numpy as np
import cv2
import os
import random

def random_crop(image, crop_size):
    height, width, _ = image.shape
    crop_height, crop_width = crop_size
    
    if crop_height > height or crop_width > width:
        raise ValueError("Crop size must be smaller than the image size")
    
    top = random.randint(0, height - crop_height)
    left = random.randint(0, width - crop_width)
    
    return image[top:top + crop_height, left:left + crop_width]

def process_images(input_folder, output_folder, crop_fraction=0.1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            height, width, _ = img.shape
            crop_height = int(height * (1 - crop_fraction))
            crop_width = int(width * (1 - crop_fraction))
            
            cropped_img = random_crop(img, (crop_height, crop_width))
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_img)

# # Example usage:
# emotion = "happy"
# input_folder = f'dataset/{emotion}'
# output_folder = f'dataset/{emotion}'
# process_images(input_folder, output_folder, crop_fraction=0.15)


emotion = "neutral"
input_folder = f'{emotion}'
output_folder = f'{emotion}'
process_images(input_folder, output_folder, crop_fraction=0.15)