
import os
import sys
import json
import numpy as np
from scipy.spatial import cKDTree as KDTree
from plyfile import PlyData



def accuracy(gt_points, rec_points):
    """
    Input: 
        gt_points: ground truth points, shape (N, 3)
        rec_points: reconstructed points, shape (M, 3)
    Output:
        acc: accuracy of the reconstructed points, mean distance to the nearest ground truth point
        acc_median: median distance to the nearest ground truth point
    """
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)
    acc_median = np.median(distances)

    return acc, acc_median

def completion(gt_points, rec_points):
    """
    Input: 
        gt_points: ground truth points, shape (N, 3)
        rec_points: reconstructed points, shape (M, 3)
    Output:
        comp: completeness of the reconstructed points, mean distance to the nearest reconstructed point
        comp_median: median distance to the nearest reconstructed point
    """
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    return comp, comp_median


def load_points_from_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']
    points = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
    return points


