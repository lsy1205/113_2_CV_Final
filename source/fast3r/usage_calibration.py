import os
import argparse
import numpy as np

import torch

from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

from invert import invert_se3_np

# --- Setup ---
# Load the model from Hugging Face
model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create a lightweight lightning module wrapper for the model.
# This provides functions to estimate camera poses, evaluate 3D reconstruction, etc.
lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

# Set model to evaluation mode
model.eval()
lit_module.eval()

# --- Load Images ---
# Provide a list of image file paths. Images can come from different cameras and aspect ratios.
parser = argparse.ArgumentParser(description="Generate poses and point clouds from a sequence of images using Fast3R.")
parser.add_argument('--seq_path', type=str, required=True, help='Path to the sequence directory')
parser.add_argument('--pose_path', type=str, required=True, help='Path to save the pose.txt')
args = parser.parse_args()

image_folder = args.seq_path
filelist = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.color.jpg', '.color.jpeg', '.color.png'))]
filelist.sort()
img_0 = filelist[0]  # The first image is used as the anchor view.
filelist = filelist[1:]

NUM = 7
i   = 0
P0_gt_path = os.path.join(args.seq_path, 'frame-000000.pose.txt')
print(f"Loading ground truth pose from {P0_gt_path}")
P0_gt = np.loadtxt(P0_gt_path, dtype=np.float32)
frame_0_pose_path = os.path.join(args.pose_path, 'frame-000000.pose.txt')
print(f"Saving ground truth pose to {frame_0_pose_path}")
with open(frame_0_pose_path, 'w') as f:
    # Write the pose matrix in the specified scientific notation format
    for row in P0_gt:
        formatted_row = [f"{val:15.7e}" for val in row]
        f.write('\t'.join(formatted_row) + '\n')

while i < len(filelist):
    print(f"Processing images {i} to {i + NUM}...")
    if i+NUM > len(filelist):
        batch_files = filelist[i:]
    else: 
        batch_files = filelist[i:i+NUM]
    
    img_list = [img_0] + batch_files
    images = load_images(img_list, size=512, verbose=True)

    # --- Run Inference ---
    # The inference function returns a dictionary with predictions and view information.
    output_dict, profiling_info = inference(
        images,
        model,
        device,
        dtype=torch.float32,  # or use torch.bfloat16 if supported
        verbose=True,
        profiling=True,
    )

    # print(output_dict.keys())
    # print(output_dict['views'][0].keys())

    # --- Estimate Camera Poses ---
    # This step estimates the camera-to-world (c2w) poses for each view using PnP.
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'],
        niter_PnP=500,
        focal_length_estimation_method='first_view_from_global_head'
    )
    # poses_c2w_batch is a list; the first element contains the estimated poses for each view.
    camera_poses = poses_c2w_batch[0]

    # Print camera poses for all views.
    P0_pred = camera_poses[0]  # The first pose is the anchor view's pose.
    C = P0_gt @ invert_se3_np(P0_pred)  # C = T0_gt @ P0_pred⁻¹

    for view_idx, pose in enumerate(camera_poses):
        if view_idx == 0:
            # Skip the first view since it's the anchor view.
            continue
        # print(f"Camera Pose for view {view_idx}:")

        image_basename = os.path.basename(img_list[view_idx])
        base_name = image_basename.split('.')[0]
        pose_file = os.path.join(args.pose_path, f"{base_name}.pose.txt")

        corrected_pose = C @ pose

        # print(f"Saving pose to {pose_file}")
        with open(pose_file, 'w') as f:
            # Write the pose matrix in the specified scientific notation format
            for row in corrected_pose:
                formatted_row = [f"{val:15.7e}" for val in row]
                f.write('\t'.join(formatted_row) + '\n')
        # print(pose.shape)  # np.array of shape (4, 4), the camera-to-world transformation matrix
        # print(pose)

    # --- Extract 3D Point Clouds for Each View ---
    # Each element in output_dict['preds'] corresponds to a view's point map.
    # for view_idx, pred in enumerate(output_dict['preds']):
    #     point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
    #     print(f"Point Cloud Shape for view {view_idx}: {point_cloud.shape}")  # shape: (1, 368, 512, 3), i.e., (1, Height, Width, XYZ)
    
    i = i + NUM