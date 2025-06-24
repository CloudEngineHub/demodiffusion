import os
import cv2
import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.utils import check_random_state
import torch
from hamer.configs import CACHE_DIR_HAMER
import os
import cv2
import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.utils import check_random_state
import torch
from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel
import json
from typing import Dict, Optional
import glob
from natsort import natsorted
from transformations import *


# Configuration constants
DEFAULT_DATA_ROOT = '../human_data'
MAX_GRIPPER_WIDTH = 0.08500000089406967  # robotiq 2f-85

def parse_args():
    parser = argparse.ArgumentParser(description="Full hand tracking pipeline")
    parser.add_argument(
        "--task_name", type=str, default="closelaptop", help="Name of task (e.g. grasp)"
    )
    parser.add_argument(
        "--traj_num", type=int, default=0, help="trajectory number"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing task folders",
    )
    return parser.parse_args()




def exponential_moving_average(data, alpha=0.1):
    """Apply EMA smoothing to 3D keypoints trajectory"""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def save_eef_gripper(traj_path, cam_id=1, alpha=0.1):
    # save eef pose, gripper action.
    extrinsics = load_transforms(path=CALIBRATION_PATH, cam_idx_list=[cam_id])[0]
    extrinsics_R = extrinsics[
        :3, :3
    ]  # this is rotation of cam, represented from world frame.

    # normalize rotation
    extrinsics_R[:, 0] = extrinsics_R[:, 0] / (
        np.linalg.norm(extrinsics_R[:, 0]) + 1e-8
    )
    extrinsics_R[:, 1] = extrinsics_R[:, 1] / (
        np.linalg.norm(extrinsics_R[:, 1]) + 1e-8
    )

    # left hand to right hand frame
    extrinsics_R[:, 2] = np.cross(extrinsics_R[:, 0], extrinsics_R[:, 1])

    keypoints_path = os.path.join(traj_path, "processed_3d/righthand_3d_keypoints.npy")

    keypoints_3d_list = np.load(keypoints_path)

    # Apply EMA to each keypoint coordinate independently
    smoothed_kps = np.zeros_like(keypoints_3d_list)
    for kp_idx in range(21):  # For each keypoint
        for coord in range(3):  # For x,y,z coordinates
            smoothed_kps[:, kp_idx, coord] = exponential_moving_average(
                keypoints_3d_list[:, kp_idx, coord], alpha
            )

    keypoints_3d_list = smoothed_kps


    eef_pose_list = []
    gripper_action_list = []
    prev_orient = np.eye(3)
    for keypoints_3d in keypoints_3d_list:
        eef_pose = np.zeros(7)


        A = keypoints_3d[0]  # Wrist (smoothed)
        B = keypoints_3d[4]  # Thumb tip (smoothed)
        C = keypoints_3d[[8, 12, 16, 20]].mean(axis=0)  # Other tips average (smoothed)

        # Calculate orthogonal basis vectors
        vec_BA = B - A
        vec_CA = C - A

        # z axis is mean of the two vectors
        z_axis = vec_CA + vec_BA
        z_axis /= np.linalg.norm(z_axis) + 1e-8

        x_axis = -np.cross(vec_CA, vec_BA)  # - is just for microwave
        x_axis /= np.linalg.norm(x_axis) + 1e-8  # Prevent division by zero

        y_axis = np.cross(z_axis, x_axis)

        try:
            ori_robot = np.column_stack(
                (x_axis.squeeze(), y_axis.squeeze(), z_axis.squeeze())
            )
        except:
            import ipdb

            ipdb.set_trace()

        prev_orient = ori_robot

        quat = rmat_to_quat(ori_robot)

        eef_pose[:3] = keypoints_3d[0]
        eef_pose[3:] = quat

        eef_pose_list.append(eef_pose)

        gripper_width = np.linalg.norm(B - C)

        gripper_width_norm = np.clip(gripper_width / MAX_GRIPPER_WIDTH, 0, 1)

        print(gripper_width_norm)

        gripper_action = 1 - gripper_width_norm
        gripper_action_list.append(gripper_action)

    eef_pose_list = np.array(eef_pose_list)

    # update offset for robotiq gripper.
    for i in range(len(eef_pose_list)):
        eef_pose = eef_pose_list[i]    
        eef_rotation = quat_to_rmat(eef_pose[3:])
        eef_pose_list[i][:3] -= eef_rotation @ np.array([0,0,0.062]).T
 
    gripper_action_list = np.array(gripper_action_list)

    save_path = os.path.join(traj_path, "processed_3d")
    os.makedirs(save_path, exist_ok=True)

    np.save(f"{save_path}/eef_pose.npy", eef_pose_list)
    np.save(f"{save_path}/retarget_gripper_action.npy", gripper_action_list)


def main():
    args = parse_args()
    task_path = os.path.join(args.data_root, args.task_name)
    traj_path = os.path.join(task_path, f"traj_{args.traj_num}")

    print("saving end effector pose and gripper actions..")
    save_eef_gripper(traj_path)

    print(f"Processing complete for {args.task_name}/traj_{args.traj_num}")


if __name__ == "__main__":
    main()
