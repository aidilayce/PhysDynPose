from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUTS_DIR = ROOT_DIR / "outputs"

RECORDINGS_DIR = INPUTS_DIR / "recordings"
FOURD_HUMANS_DIR = INPUTS_DIR / "4d_humans"
ORB_SLAM3_DIR = INPUTS_DIR / "orb_slam3"
SLAM_DIR = INPUTS_DIR / "slam"
PIP_REFERENCE_DIR = INPUTS_DIR / "pip_reference" / "dataset_captury"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    ensure_dir(path.parent)
    return path


def pip_reference_seq_no(seq_no: int, surface: int) -> int:
    if surface == 24:
        mapping = {1: 8, 2: 9}
        if seq_no not in mapping:
            raise ValueError(f"Unsupported flat-surface sequence: seq{seq_no}")
        return mapping[seq_no]
    return seq_no


def recording_dir(seq_no: int, surface: int) -> Path:
    return RECORDINGS_DIR / f"Recordings_{surface}_03_23" / f"seq{seq_no}_captury"


def camera_exts_txt(seq_no: int, surface: int) -> Path:
    return recording_dir(seq_no, surface) / "moving_cam" / "camera_exts_full.txt"


def camera_ext_npy(seq_no: int, surface: int) -> Path:
    return recording_dir(seq_no, surface) / "moving_cam" / "cam_ext.npy"


def slam_first_npy(seq_no: int, surface: int) -> Path:
    return recording_dir(seq_no, surface) / "moving_cam" / "cam_ext_slam_first.npy"


def gt_joints_npy(seq_no: int, surface: int) -> Path:
    return recording_dir(seq_no, surface) / "smpl" / "3d_joints.npy"


def contact_labels_npy(seq_no: int, surface: int) -> Path:
    return recording_dir(seq_no, surface) / "labels_con_sta" / "pred_con_gt.npy"


def pip_reference_dir(seq_no: int, surface: int) -> Path:
    return PIP_REFERENCE_DIR / f"seq{pip_reference_seq_no(seq_no, surface)}"


def pip_reference_pose_pt(seq_no: int, surface: int) -> Path:
    return pip_reference_dir(seq_no, surface) / "pose.pt"


def pip_reference_shape_pt(seq_no: int, surface: int) -> Path:
    return pip_reference_dir(seq_no, surface) / "shape.pt"


def fourdhumans_out_list(seq_no: int, surface: int) -> Path:
    out_seq = pip_reference_seq_no(seq_no, surface)
    return FOURD_HUMANS_DIR / f"outputs_{out_seq:02d}" / "out_list.pkl"


def slam_trajectory_txt() -> Path:
    return SLAM_DIR / "Trajectory.txt"


def legacy_orb_trajectory_txt() -> Path:
    return ORB_SLAM3_DIR / "Trajectory.txt"


def aligned_slam_output_path(seq_no: int, surface: int) -> Path:
    return ensure_parent(OUTPUTS_DIR / "slam" / f"cleaned_slam_seq{seq_no}_{surface}.npy")


def pip_input_dir(seq_no: int, surface: int, slam: bool) -> Path:
    suffix = "_slam" if slam else ""
    return ensure_dir(OUTPUTS_DIR / "pip_input" / f"seq{seq_no}_{surface}_4dhuman{suffix}")


def opt_result_path(seq_no: int, surface: int, slam: bool) -> Path:
    suffix = "_slam" if slam else ""
    return ensure_parent(
        OUTPUTS_DIR
        / "opt_result"
        / f"seq{seq_no}_{surface}_4dhuman"
        / "PIP"
        / f"seq{seq_no}_smoothed{suffix}.pt"
    )


def converted_joints_path(seq_no: int, surface: int, slam: bool) -> Path:
    suffix = "_slam" if slam else ""
    return ensure_parent(
        OUTPUTS_DIR
        / "convert_res"
        / f"seq{seq_no}_{surface}"
        / "eval"
        / f"seq{seq_no}_{surface}{suffix}.npy"
    )
