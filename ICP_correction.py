#!/usr/bin/env python3
"""incremental_icp_pose_refine.py — **v1.1**
Fixes TypeError caused by passing RGBDImage directly to `registration_icp`.
Now converts RGB‑D frames into down‑sampled point clouds and runs point‑to‑plane
ICP on them.
"""
from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
from pathlib import Path
from typing import List

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_intrinsics(width: int = 640, height: int = 480,
                    fx: float = 525.0, fy: float = 525.0,
                    cx: float = 320.0, cy: float = 240.0) -> o3d.camera.PinholeCameraIntrinsic:
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width, height, fx, fy, cx, cy)
    return intr


def read_depth(path: str) -> np.ndarray:
    if path.endswith((".exr", ".EXR")):
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth /= 1000.0
    depth[np.isinf(depth)] = 0
    depth[np.isnan(depth)] = 0
    depth[(depth < 1e-3) | (depth > 10.0)] = 0
    return depth.astype(np.float32)


def read_color(path: str) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img


def build_rgbd(depth_f: str, color_f: str | None, intr: o3d.camera.PinholeCameraIntrinsic) -> o3d.geometry.RGBDImage:
    depth = read_depth(depth_f)
    o3d_depth = o3d.geometry.Image(depth)

    if color_f and osp.isfile(color_f):
        color = read_color(color_f)
        o3d_color = o3d.geometry.Image(color)
    else:
        o3d_color = o3d.geometry.Image(np.zeros_like(depth, dtype=np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=1.0, convert_rgb_to_intensity=False)
    return rgbd


def rgbd_to_down_pc(rgbd: o3d.geometry.RGBDImage, intr: o3d.camera.PinholeCameraIntrinsic,
                    voxel: float) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    pc_down = pc.voxel_down_sample(voxel)
    pc_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=30))
    return pc_down


def icp_pair(src_pc: o3d.geometry.PointCloud, tgt_pc: o3d.geometry.PointCloud,
             init_T: np.ndarray, max_dist: float, max_iter: int = 50) -> tuple[np.ndarray, float]:
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    reg = o3d.pipelines.registration.registration_icp(
        src_pc, tgt_pc, max_dist, init_T.astype(np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)
    return reg.transformation.astype(np.float32), reg.fitness


# -----------------------------------------------------------------------------
# Main refinement loop
# -----------------------------------------------------------------------------

def refine_sequence(color_dir: str, depth_dir: str, pred_pose_dir: str,
                    out_pose_dir: str, voxel: float = 0.02,
                    max_icp_dist: float = 0.07, min_fitness: float = 0.03,
                    intr: o3d.camera.PinholeCameraIntrinsic | None = None):

    Path(out_pose_dir).mkdir(parents=True, exist_ok=True)
    intr = intr or load_intrinsics()

    pose_files = sorted(glob.glob(osp.join(pred_pose_dir, "*.pose.txt")))
    if not pose_files:
        raise FileNotFoundError("No *.pose.txt in pred_pose_dir")

    frame_ids = [osp.splitext(osp.basename(p))[0].replace('.pose', '') for p in pose_files]

    # Frame 0: trusted anchor
    anchor_id = frame_ids[0]
    refined = {anchor_id: np.loadtxt(pose_files[0]).astype(np.float32)}
    np.savetxt(osp.join(out_pose_dir, f"{anchor_id}.pose.txt"), refined[anchor_id], fmt='%16.7e')

    def get_down_pc(fid: str):
        depth_f = osp.join(depth_dir, f"{fid}.depth.proj.png")
        color_f = osp.join(color_dir, f"{fid}.color.png")
        rgbd = build_rgbd(depth_f, color_f, intr)
        return rgbd_to_down_pc(rgbd, intr, voxel)

    prev_fid = anchor_id
    tgt_pc = get_down_pc(prev_fid)
    
    for fid in tqdm(frame_ids[1:], desc="Incremental ICP"):
        src_pc = get_down_pc(fid)

        # initial relative transform
        T_pred = np.loadtxt(osp.join(pred_pose_dir, f"{fid}.pose.txt")).astype(np.float32)
        T_prev_abs = refined[prev_fid]
        T_init_rel = np.linalg.inv(T_prev_abs) @ T_pred

        ΔT, fitness = icp_pair(src_pc, tgt_pc, T_init_rel, max_icp_dist)

        if fitness < min_fitness:
            print(f"[WARN] ICP poor fit ({fitness:.2f}) on {fid}; using predicted pose.")
            refined_T = T_pred
        else:
            refined_T = ΔT @ T_prev_abs
        refined[fid] = refined_T.astype(np.float32)

        np.savetxt(osp.join(out_pose_dir, f"{fid}.pose.txt"), refined_T, fmt='%16.7e')

        # update target for next iteration
        # tgt_pc = src_pc
        # prev_fid = fid

    print(f"Refinement complete → poses saved to {out_pose_dir}")


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Incrementally refine predicted poses using pair‑wise ICP.")
    parser.add_argument("--color_dir", required=True,
                        help="Directory with *.color.png images")
    parser.add_argument("--depth_dir", required=True,
                        help="Directory with *.depth.proj.png images")
    parser.add_argument("--pred_pose_dir", required=True,
                        help="Directory with predicted *.pose.txt files")
    parser.add_argument("--out_pose_dir", required=True,
                        help="Output directory for refined *.pose.txt files")
    parser.add_argument("--voxel", type=float, default=0.02,
                        help="Voxel size for down‑sampling (metres)")
    args = parser.parse_args()

    refine_sequence(args.color_dir, args.depth_dir, args.pred_pose_dir,
                    args.out_pose_dir, voxel=args.voxel)
