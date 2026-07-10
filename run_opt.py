import argparse

import numpy as np
import torch
import articulate as art

from config import joint_set, paths
from net import PIP
from release_paths import contact_labels_npy, opt_result_path, pip_input_dir


JI_MASK = torch.tensor([18, 19, 4, 5, 15, 0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_no", type=int, required=True, help="Sequence number")
    parser.add_argument("--surface", type=int, required=True, help="Surface type: 31 or 24")
    parser.add_argument("--slam", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_full_local_pose(inverse_kinematics_fn, root_rotation, global_reduced_pose):
    global_reduced_pose = art.math.r6d_to_rotation_matrix(global_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
    global_full_pose = torch.eye(3).repeat(global_reduced_pose.shape[0], 24, 1, 1)
    global_full_pose[:, joint_set.reduced] = global_reduced_pose
    pose = inverse_kinematics_fn(global_full_pose).view(-1, 24, 3, 3)
    pose[:, joint_set.ignored] = torch.eye(3)
    pose[:, 0] = root_rotation.view(-1, 3, 3)
    return pose


def main():
    args = parse_args()
    input_dir = pip_input_dir(args.seq_no, args.surface, args.slam)

    pose_matrix = torch.load(input_dir / "pred_pose_smooth.pt", map_location="cpu")
    joint_positions = torch.load(input_dir / "pred_joints_smooth.pt", map_location="cpu")
    joint_velocity = torch.load(input_dir / "pred_joint_velocity_smooth.pt", map_location="cpu")
    shape = torch.load(input_dir / "pred_beta.pt", map_location="cpu").float()
    contacts = np.load(contact_labels_npy(args.seq_no, args.surface))

    if shape.ndim == 1:
        shape = shape.unsqueeze(0)

    sequence_name = f"seq{args.seq_no}"
    pose_axis_angle = art.math.rotation_matrix_to_axis_angle(pose_matrix).view(-1, 24, 3)
    root_translation = joint_positions[:, 0]

    body_model = art.ParametricModel(paths.smpl_file)
    pose_rotation = art.math.axis_angle_to_rotation_matrix(pose_axis_angle).view(-1, 24, 3, 3)
    pose_rotation[:, 0] = torch.eye(3)
    global_pose = body_model.forward_kinematics_R(pose_rotation)
    global_reduced_pose = art.math.rotation_matrix_to_r6d(global_pose[:, joint_set.reduced]).view(
        -1, len(joint_set.reduced) * 6
    )

    pose_rotation = art.math.axis_angle_to_rotation_matrix(pose_axis_angle).view(-1, 24, 3, 3)
    global_rotation, _ = body_model.forward_kinematics(pose_rotation, shape, root_translation, calc_mesh=False)
    global_imu_rotation = global_rotation[:, JI_MASK]

    contact_count = min(len(contacts), len(pose_axis_angle))
    contacts = contacts[:contact_count]
    pose_axis_angle = pose_axis_angle[:contact_count]
    joint_positions = joint_positions[:contact_count]
    global_reduced_pose = global_reduced_pose[:contact_count]
    global_imu_rotation = global_imu_rotation[:contact_count]

    full_local_pose = build_full_local_pose(
        body_model.inverse_kinematics_R,
        global_imu_rotation.view(-1, 6, 3, 3)[:, -1],
        global_reduced_pose,
    )

    contacts = contacts[:contact_count] > 0.5
    left_contact = contacts[:, 0] | contacts[:, 1]
    right_contact = contacts[:, 2] | contacts[:, 3]
    contact_tensor = torch.tensor(np.column_stack((left_contact, right_contact)), dtype=torch.float32)

    result = PIP().predict(
        global_imu_rotation,
        full_local_pose,
        joint_velocity,
        contact_tensor,
        sequence_name,
        joint_positions,
    )

    save_path = opt_result_path(args.seq_no, args.surface, args.slam)
    torch.save(result, save_path)
    print(f"Saved optimized motion to {save_path}")


if __name__ == "__main__":
    main()
