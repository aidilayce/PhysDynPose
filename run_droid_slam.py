import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import sequence_frame_range
from release_paths import camera_exts_txt, ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid_root", type=Path, required=True, help="Path to the external DROID-SLAM checkout")
    parser.add_argument("--imagedir", type=Path, required=True, help="Directory containing extracted frames")
    parser.add_argument("--seq_no", type=int, required=True, help="Sequence number")
    parser.add_argument("--surface", type=int, required=True, help="Surface type: 31 or 24")
    parser.add_argument("--output", type=Path, required=True, help="Output trajectory path")
    parser.add_argument("--weights", type=Path, default=None, help="Path to droid.pth")
    parser.add_argument("--t0", type=int, default=0, help="Starting frame inside the extracted frame folder")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride passed to DROID-SLAM")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    parser.add_argument("--asynchronous", action="store_true")
    return parser.parse_args()


def load_camera_exts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as handle:
        content = handle.readlines()
    intrinsic_lines = [line.strip().split()[1:] for line in content[1::3]]
    extrinsic_lines = [line.strip().split()[1:] for line in content[2::3]]
    intrinsics = np.array([list(map(float, line)) for line in intrinsic_lines]).reshape((-1, 4, 4))
    extrinsics = np.array([list(map(float, line)) for line in extrinsic_lines]).reshape((-1, 4, 4))
    return intrinsics, extrinsics


def write_droid_calibration(seq_no: int, surface: int, output_path: Path) -> None:
    start_frame, _ = sequence_frame_range(seq_no, surface)
    intrinsics, _ = load_camera_exts(camera_exts_txt(seq_no, surface))
    camera_matrix = intrinsics[start_frame]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    ensure_parent(output_path)
    output_path.write_text(f"{fx} {fy} {cx} {cy}\n")


def image_stream(imagedir: Path, calib: Path, stride: int):
    calibration = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calibration[:4]

    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = fx
    camera_matrix[0, 2] = cx
    camera_matrix[1, 1] = fy
    camera_matrix[1, 2] = cy

    image_list = sorted(p for p in imagedir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[::stride]
    for timestamp, image_path in enumerate(image_list):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        if len(calibration) > 4:
            image = cv2.undistort(image, camera_matrix, calibration[4:])

        height, width, _ = image.shape
        scale = np.sqrt((384 * 512) / (height * width))
        resized_height = int(height * scale)
        resized_width = int(width * scale)
        image = cv2.resize(image, (resized_width, resized_height))
        image = image[: resized_height - resized_height % 8, : resized_width - resized_width % 8]

        tensor = torch.as_tensor(image).permute(2, 0, 1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= resized_width / width
        intrinsics[1::2] *= resized_height / height
        yield timestamp, tensor[None], intrinsics


def save_trajectory(path: Path, trajectory: np.ndarray) -> None:
    ensure_parent(path)
    with path.open("w") as handle:
        for timestamp, pose in enumerate(trajectory):
            if pose.shape[0] != 7:
                raise ValueError(f"Unexpected DROID-SLAM pose shape: {pose.shape}")
            tx, ty, tz, qx, qy, qz, qw = pose.tolist()
            handle.write(f"{timestamp:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def main() -> None:
    args = parse_args()

    droid_root = args.droid_root.resolve()
    sys.path.insert(0, str(droid_root))
    sys.path.insert(0, str(droid_root / "droid_slam"))

    from droid import Droid
    from droid_async import DroidAsync

    weights_path = args.weights.resolve() if args.weights is not None else droid_root / "droid.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find DROID-SLAM weights: {weights_path}")

    calib_path = args.output.parent / "droid_calib.txt"
    write_droid_calibration(args.seq_no, args.surface, calib_path)

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    args.stereo = False
    args.disable_vis = True
    args.image_size = [240, 320]
    args.weights = str(weights_path)

    droid = None
    for timestamp, image, intrinsics in tqdm(image_stream(args.imagedir, calib_path, args.stride)):
        if timestamp < args.t0:
            continue
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)
        droid.track(timestamp, image, intrinsics=intrinsics)

    if droid is None:
        raise RuntimeError(f"No frames found in {args.imagedir}")

    trajectory = droid.terminate(image_stream(args.imagedir, calib_path, args.stride))
    save_trajectory(args.output, trajectory)
    print(f"Saved DROID-SLAM trajectory to {args.output}")


if __name__ == "__main__":
    main()
