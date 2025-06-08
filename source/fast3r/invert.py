import torch
import numpy as np
import os
import argparse

def invert_se3_np(T):
    R = T[:3, :3]
    t = T[:3, 3:4]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3:4] = t_inv
    return T_inv

def pose_errors_np(T_pred, T_gt):
    """
    計算平移 (cm) RMSE 與旋轉 (deg) MAE
    T_pred, T_gt: (...,4,4)
    """
    # --- translation error --- #
    t_pred = T_pred[..., :3, 3]
    t_gt   = T_gt[...,   :3, 3]
    trans_err = np.linalg.norm(t_pred - t_gt, axis=-1)      # (batch,) m
    trans_rmse = np.sqrt(np.mean(trans_err ** 2)) * 100     # → cm

    # --- rotation error (geodesic) --- #
    R_pred = T_pred[..., :3, :3]
    R_gt   = T_gt[...,   :3, :3]
    # R_rel = R_pred R_gtᵀ
    R_rel = np.matmul(R_pred, np.swapaxes(R_gt, -1, -2))
    # cosθ = (trace(R_rel) - 1) / 2，夾到 [-1,1] 以防數值誤差
    cos_theta = (np.trace(R_rel, axis1=-2, axis2=-1) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    rot_err_deg = np.degrees(np.arccos(cos_theta))          # (batch,) deg
    rot_mae = np.mean(rot_err_deg)

    return trans_rmse, rot_mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert SE(3) transformation matrix")
    parser.add_argument("--gt_pose0_path", type=str, required=True, help="Path to ground truth pose file")
    parser.add_argument("--pred_pose0_path", type=str, required=True, help="Path to the predicted reference pose file")
    parser.add_argument("--pred_posei_path", type=str, required=True, help="Path to the predicted pose file")
    parser.add_argument("--gt_posei_path", type=str, required=True, help="Path to ground truth pose file")
    args = parser.parse_args()

    T0_gt = np.loadtxt(args.gt_pose0_path, dtype=np.float32)
    P0_pred = np.loadtxt(args.pred_pose0_path, dtype=np.float32)
    Pi_pred = np.loadtxt(args.pred_posei_path, dtype=np.float32)
    Pi_gt = np.loadtxt(args.gt_posei_path, dtype=np.float32)

    print("T0_gt shape:", T0_gt.shape)  # Should be (4, 4)
    print(T0_gt)
    
    print("P0_pred shape:", P0_pred.shape)  # Should be (4, 4)
    print(P0_pred)
    
    print("Pi_pred shape:", Pi_pred.shape)  # Should be (N, 4, 4)
    print(Pi_pred)

    print("Pi_gt shape:", Pi_gt.shape)  # Should be (N, 4, 4)
    print(Pi_gt)

    C = T0_gt @ invert_se3_np(P0_pred)
    print("C shape:", C.shape)  # Should be (4, 4)\
    print(C)
    
    # Correct the predicted poses
    Pi_corrected = C @ Pi_pred    # Pi_pred shape: (N,4,4)

    print("Pi_corrected shape:", Pi_corrected.shape)  # Should be (N, 4, 4)
    print(Pi_corrected)

    # Calculate the error
    Original_rmse, Original_mae = pose_errors_np(Pi_pred, Pi_gt)
    Corrected_rmse, Corrected_mae = pose_errors_np(Pi_corrected, Pi_gt)
    print(f"Original RMSE: {Original_rmse:.4f} cm, Original MAE: {Original_mae:.4f} deg")
    print(f"Corrected RMSE: {Corrected_rmse:.4f} cm, Corrected MAE: {Corrected_mae:.4f} deg")