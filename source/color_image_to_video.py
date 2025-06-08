#!/usr/bin/env python3
import os
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm

def create_video_from_images(image_folder, output_file, fps=30, sort_numerically=True, extension='jpg'):
    image_files = glob.glob(os.path.join(image_folder, f'*.color.png'))
    
    if not image_files:
        print("no images found in the specified folder.")
        return
    
    if sort_numerically:
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    else:
        image_files.sort()
    
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    print(f"dealing {len(image_files)} images...")
    for image_file in tqdm(image_files):
        image = cv2.imread(image_file)
        
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height))
            
        video_writer.write(image)
    
    video_writer.release()
    print(f"video saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='image to video converter')
    parser.add_argument('--input', '-i', type=str, required=True, help='path to image folder')
    parser.add_argument('--output', '-o', type=str, required=True, help='output video file path')
    parser.add_argument('--fps', type=int, default=30, help='fps for the output video (default: 30)')
    parser.add_argument('--extension', '-e', type=str, default='png', help='file extension of images (default: png)')
    parser.add_argument('--no-numeric-sort', action='store_true', help='sort images alphabetically instead of numerically')
    
    args = parser.parse_args()
    
    create_video_from_images(
        args.input, 
        args.output, 
        args.fps, 
        not args.no_numeric_sort,
        args.extension
    )

if __name__ == "__main__":
    main()