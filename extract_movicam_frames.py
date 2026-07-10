import argparse
from pathlib import Path

import cv2

from config import sequence_frame_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=Path, required=True, help="Path to the synced moving-camera video")
    parser.add_argument("--seq_no", type=int, required=True, help="Sequence number")
    parser.add_argument("--surface", type=int, required=True, help="Surface type: 31 or 24")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory where cropped frames will be saved")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_frame, end_frame = sequence_frame_range(args.seq_no, args.surface)
    expected_count = end_frame - start_frame + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = 0
    frame_index = start_frame
    while frame_index <= end_frame:
        ok, frame = capture.read()
        if not ok:
            break

        frame_path = args.output_dir / f"{saved:06d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Failed to write frame: {frame_path}")

        saved += 1
        frame_index += 1

    capture.release()

    if saved != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} frames for seq{args.seq_no}, surface {args.surface}, but saved {saved}."
        )

    print(f"Saved {saved} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
