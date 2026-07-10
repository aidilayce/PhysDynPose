import argparse

import joblib
import numpy as np
import torch
from pytorch3d import transforms
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from smplx import SMPL

from config import paths, sequence_frame_range
from one_euro_filter import OneEuroFilter
from release_paths import (
    aligned_slam_output_path,
    camera_ext_npy,
    camera_exts_txt,
    fourdhumans_out_list,
    legacy_orb_trajectory_txt,
    pip_input_dir,
    slam_first_npy,
    slam_trajectory_txt,
)


FPS = 60
TRANSLATION_SIGMA = 6.0
POSE_MIN_CUTOFF = 0.04
POSE_BETA = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_no", type=int, required=True, help="Sequence number")
    parser.add_argument("--surface", type=int, required=True, help="Surface type: 31 or 24")
    parser.add_argument("--slam", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_camera_exts_txt(path):
    with open(path) as handle:
        content = handle.readlines()
    intrinsic_lines = [line.strip().split()[1:] for line in content[1::3]]
    extrinsic_lines = [line.strip().split()[1:] for line in content[2::3]]
    intrinsics = np.array([list(map(float, line)) for line in intrinsic_lines]).reshape((-1, 4, 4))
    extrinsics = np.array([list(map(float, line)) for line in extrinsic_lines]).reshape((-1, 4, 4))
    return intrinsics, extrinsics


def pose_to_extrinsics(translation, quaternion):
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R.from_quat(quaternion).as_matrix()
    pose_matrix[:3, 3] = translation
    return np.linalg.inv(pose_matrix)


def read_orb_trajectory(path):
    extrinsics = []
    with open(path) as handle:
        for line in handle:
            timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, line.strip().split())
            del timestamp
            extrinsics.append(pose_to_extrinsics([tx, ty, tz], [qx, qy, qz, qw]))
    return np.stack(extrinsics)


def align_orb_poses(orb_poses, gt_poses):
    frame_count = min(len(orb_poses), len(gt_poses))
    orb_poses = orb_poses[:frame_count].copy()
    gt_poses = gt_poses[:frame_count].copy()

    if frame_count < 3:
        raise ValueError("Need at least 3 frames to align ORB-SLAM trajectory.")

    gt_scale = np.linalg.norm(gt_poses[2, :3, 3] - gt_poses[1, :3, 3])
    orb_scale = np.linalg.norm(orb_poses[2, :3, 3] - orb_poses[1, :3, 3])
    if orb_scale == 0:
        raise ValueError("ORB-SLAM trajectory scale is zero.")

    scale = gt_scale / orb_scale * 0.6
    orb_poses[:, :3, 3] *= scale

    rotation_fix = np.array(
        [
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
        ]
    )
    for index in range(frame_count):
        orb_poses[index, :3, :3] = orb_poses[index, :3, :3] @ rotation_fix

    anchor = np.linalg.inv(orb_poses[0])
    for index in range(frame_count):
        orb_poses[index] = anchor @ orb_poses[index]
        orb_poses[index] = gt_poses[0] @ orb_poses[index]

    return orb_poses


def load_slam_poses(seq_no: int, surface: int) -> np.ndarray:
    aligned_path = aligned_slam_output_path(seq_no, surface)
    if aligned_path.exists():
        return np.load(aligned_path)

    trajectory_path = slam_trajectory_txt()
    if not trajectory_path.exists():
        trajectory_path = legacy_orb_trajectory_txt()

    if trajectory_path.exists():
        orb_poses = read_orb_trajectory(trajectory_path)
        gt_poses = np.load(camera_ext_npy(seq_no, surface))
        aligned = align_orb_poses(orb_poses, gt_poses)
        np.save(aligned_path, aligned)
        return aligned

    return np.load(slam_first_npy(seq_no, surface))


def select_track_tensor(value):
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim > 1 and tensor.shape[0] in (1, 2):
        tensor = tensor[0]
    return tensor.squeeze()


def compute_translation_offset(smpl_model: SMPL, pose_aa: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    smpl_output = smpl_model(
        betas=betas.unsqueeze(0),
        body_pose=pose_aa[1:].reshape(1, -1),
        global_orient=pose_aa[0:1].reshape(1, -1),
    )
    joints = smpl_output.joints.detach().cpu()[0]
    return (joints[1] + joints[2]) / 2


def smooth_pose_axis_angle(pose_aa: torch.Tensor) -> torch.Tensor:
    pose_np = pose_aa.numpy()
    pose_filter = OneEuroFilter(
        np.zeros_like(pose_np[0]),
        pose_np[0],
        min_cutoff=POSE_MIN_CUTOFF,
        beta=POSE_BETA,
    )

    pose_smooth = np.zeros_like(pose_np)
    pose_smooth[0] = pose_np[0]
    for index, pose in enumerate(pose_np[1:], start=1):
        time_value = np.ones_like(pose) * index
        pose_smooth[index] = pose_filter(time_value, pose)

    return torch.from_numpy(pose_smooth).float()


def build_pip_inputs(seq_no: int, surface: int, slam: bool):
    start_frame, end_frame = sequence_frame_range(seq_no, surface)
    intrinsics, _ = load_camera_exts_txt(camera_exts_txt(seq_no, surface))
    intrinsics = intrinsics[start_frame : end_frame + 1, :3, :3]

    camera_poses = load_slam_poses(seq_no, surface) if slam else np.load(camera_ext_npy(seq_no, surface))
    tracks = joblib.load(fourdhumans_out_list(seq_no, surface))

    frame_count = min(len(intrinsics), len(camera_poses), len(tracks))
    intrinsics = intrinsics[:frame_count]
    camera_poses = torch.from_numpy(camera_poses[:frame_count]).float()
    tracks = tracks[:frame_count]

    rotation_world = camera_poses[:, :3, :3]
    translation_world = camera_poses[:, :3, 3]

    smpl_model = SMPL(paths.smpl_file)
    first_track = tracks[0]["pred_smpl_params"]
    first_betas = select_track_tensor(first_track["betas"]).float()
    first_body_pose = transforms.matrix_to_axis_angle(select_track_tensor(first_track["body_pose"]).float())
    first_global_orient = select_track_tensor(first_track["global_orient"]).float()
    first_world_orient = rotation_world[0].T @ first_global_orient
    first_pose_aa = torch.cat(
        (
            transforms.matrix_to_axis_angle(first_world_orient).reshape(1, 3),
            first_body_pose,
        ),
        dim=0,
    )
    translation_offset = compute_translation_offset(smpl_model, first_pose_aa, first_betas)

    pose_list = []
    joint_list = []
    root_rot_list = []
    world_translation_list = []
    last_betas = first_betas

    for frame_index, track in enumerate(tracks):
        smpl_params = track["pred_smpl_params"]
        betas = select_track_tensor(smpl_params["betas"]).float()
        body_pose = select_track_tensor(smpl_params["body_pose"]).float()
        global_orient = select_track_tensor(smpl_params["global_orient"]).float()
        camera_translation = select_track_tensor(track["pred_cam_t_full"]).float().reshape(3)

        world_orient_matrix = rotation_world[frame_index].T @ global_orient
        world_translation = rotation_world[frame_index].T @ (camera_translation - translation_world[frame_index])

        body_pose_aa = transforms.matrix_to_axis_angle(body_pose)
        world_orient_aa = transforms.matrix_to_axis_angle(world_orient_matrix).reshape(1, 3)

        smpl_output = smpl_model(
            betas=betas.unsqueeze(0),
            body_pose=body_pose_aa.reshape(1, -1),
            global_orient=world_orient_aa,
            transl=world_translation.reshape(1, 3),
            return_full_pose=True,
        )

        joints = smpl_output.joints.detach().cpu()[0, :24] - translation_offset
        pose_aa = torch.cat((world_orient_aa, body_pose_aa), dim=0)
        pose_matrix = transforms.axis_angle_to_matrix(pose_aa)

        pose_list.append(pose_matrix)
        joint_list.append(joints)
        root_rot_list.append(pose_matrix[0])
        world_translation_list.append(world_translation)
        last_betas = betas

    pose = torch.stack(pose_list)
    joints = torch.stack(joint_list)
    root_rot = torch.stack(root_rot_list)
    world_translation = torch.stack(world_translation_list)
    joint_velocity = (joints[1:] - joints[:-1]).bmm(root_rot[1:]).flatten(1) * FPS

    smooth_translation = torch.from_numpy(
        gaussian_filter1d(world_translation.numpy(), sigma=TRANSLATION_SIGMA, axis=0, mode="nearest")
    ).float()
    smooth_translation = smooth_translation - translation_offset

    pose_aa = transforms.matrix_to_axis_angle(pose)
    smooth_pose_aa = smooth_pose_axis_angle(pose_aa)
    repeated_betas = last_betas.unsqueeze(0).repeat(len(smooth_pose_aa), 1)

    smooth_output = smpl_model(
        betas=repeated_betas,
        body_pose=smooth_pose_aa[:, 1:].reshape(len(smooth_pose_aa), -1),
        global_orient=smooth_pose_aa[:, 0].reshape(len(smooth_pose_aa), -1),
        transl=smooth_translation,
        return_full_pose=True,
    )

    smooth_pose = transforms.axis_angle_to_matrix(smooth_pose_aa)
    smooth_joints = smooth_output.joints.detach().cpu()[:, :24]
    smooth_root_rot = smooth_pose[:, 0]
    smooth_joint_velocity = (smooth_joints[1:] - smooth_joints[:-1]).bmm(smooth_root_rot[1:]).flatten(1) * FPS

    return {
        "pred_pose.pt": pose,
        "pred_joint_velocity.pt": joint_velocity,
        "pred_joints.pt": joints,
        "root_rots.pt": root_rot,
        "pred_world_translation.pt": world_translation,
        "pred_beta.pt": last_betas,
        "pred_pose_smooth.pt": smooth_pose,
        "pred_joint_velocity_smooth.pt": smooth_joint_velocity,
        "pred_joints_smooth.pt": smooth_joints,
        "root_rots_smooth.pt": smooth_root_rot,
    }


def main():
    args = parse_args()
    output_dir = pip_input_dir(args.seq_no, args.surface, args.slam)
    outputs = build_pip_inputs(args.seq_no, args.surface, args.slam)

    for filename, tensor in outputs.items():
        torch.save(tensor, output_dir / filename)

    print(f"Saved preprocessed PIP inputs to {output_dir}")


if __name__ == "__main__":
    main()
