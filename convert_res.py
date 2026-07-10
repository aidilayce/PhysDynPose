import argparse

import numpy as np
import torch
from pytorch3d import transforms

from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from release_paths import converted_joints_path, opt_result_path, pip_reference_shape_pt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_no", type=int, required=True, help="Sequence number")
    parser.add_argument("--surface", type=int, required=True, help="Surface type: 31 or 24")
    parser.add_argument("--method", type=str, default="PIP", help="Only PIP is supported in this release")
    parser.add_argument("--slam", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.method != "PIP":
        raise NotImplementedError("This release supports only --method PIP.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    smpl = SMPL(SMPL_MODEL_DIR, pose_type="body26fk", create_transl=False).to(device)

    pose, translation = torch.load(opt_result_path(args.seq_no, args.surface, args.slam), map_location="cpu")
    pose_aa = transforms.matrix_to_axis_angle(pose)

    shape = torch.load(pip_reference_shape_pt(args.seq_no, args.surface), map_location="cpu")[0][:, :10]
    shape = torch.repeat_interleave(shape, len(pose_aa), dim=0)

    smpl_output = smpl(
        global_orient=pose_aa[:, 0, :].to(device),
        body_pose=pose_aa[:, 1:, :].reshape(len(pose_aa), -1).to(device),
        betas=shape.to(device),
        root_trans=translation.to(device),
        return_full_pose=True,
        orig_joints=True,
    )

    save_path = converted_joints_path(args.seq_no, args.surface, args.slam)
    np.save(save_path, smpl_output.joints.detach().cpu().numpy())
    print(f"Saved converted joints to {save_path}")


if __name__ == "__main__":
    main()
