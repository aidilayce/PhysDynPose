import argparse
import gzip
import shutil
import time
from pathlib import Path

import pybullet as p
import torch

from articulate.utils.bullet import change_color
from config import paths
from release_paths import opt_result_path, pip_input_dir
from utils import set_pose, smpl_to_rbdl


COLORS = {
    "pip": [198 / 255, 238 / 255, 0, 1.0],
    "4dhumans": [1.0, 0.0, 0.0, 1.0],
}

ASSET_DIR = Path(__file__).resolve().parent / "asset"
SCENE_URDF = ASSET_DIR / "scene.urdf"
SCENE_OBJ = ASSET_DIR / "no_wall_flip.obj"
SCENE_OBJ_GZ = ASSET_DIR / "no_wall_flip.obj.gz"
SCENE_ROTATION = [-0.7071068, 0.0, 0.0, 0.7071068]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_no", type=int, required=True, help="Sequence number")
    parser.add_argument("--surface", type=int, required=True, help="Surface type: 31 or 24")
    parser.add_argument("--method", choices=["pip", "4dhumans"], default="pip")
    parser.add_argument("--slam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--scene", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_motion(args):
    if args.method == "pip":
        pose, translation = torch.load(opt_result_path(args.seq_no, args.surface, args.slam), map_location="cpu")
        return pose, translation

    input_dir = pip_input_dir(args.seq_no, args.surface, args.slam)
    pose = torch.load(input_dir / "pred_pose_smooth.pt", map_location="cpu")
    translation = torch.load(input_dir / "pred_joints_smooth.pt", map_location="cpu")[:, 0, :]
    return pose, translation


def load_scene(args) -> bool:
    if not args.scene:
        print("Scene disabled.")
        return False

    if not SCENE_OBJ.exists() and SCENE_OBJ_GZ.exists():
        with gzip.open(SCENE_OBJ_GZ, "rb") as src, SCENE_OBJ.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    if SCENE_URDF.exists():
        p.loadURDF(str(SCENE_URDF), [0.0, 0.0, 0.0], SCENE_ROTATION)
        print(f"Loaded scene URDF: {SCENE_URDF}")
        return True

    print("No scene URDF found.")
    return False


def main():
    args = parse_args()
    pose, translation = load_motion(args)

    p.connect(p.GUI)
    p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
    p.configureDebugVisualizer(flag=p.COV_ENABLE_SHADOWS, enable=0)
    load_scene(args)
    p.setRealTimeSimulation(0)

    robot_id = p.loadURDF(
        paths.physics_model_file,
        translation[0],
        useFixedBase=False,
        flags=p.URDF_MERGE_FIXED_LINKS,
    )
    change_color(robot_id, COLORS[args.method])

    for pose_frame, translation_frame in zip(pose, translation):
        q = smpl_to_rbdl(pose_frame, translation_frame)[0]
        set_pose(robot_id, q)
        p.stepSimulation()
        time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    main()
