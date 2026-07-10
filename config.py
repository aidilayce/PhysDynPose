from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

FRAME_RANGES = {
    31: {
        1: (2908, 6240),
        2: (692, 4466),
        3: (2171, 5767),
        4: (1778, 5309),
        6: (1576, 3899),
        7: (1900, 5400),
    },
    24: {
        1: (700, 4190),
        2: (800, 2721),
    },
}


def sequence_frame_range(seq_no: int, surface: int) -> tuple[int, int]:
    try:
        return FRAME_RANGES[surface][seq_no]
    except KeyError as exc:
        raise ValueError(f"Unsupported sequence/surface pair: seq{seq_no}, surface {surface}") from exc


class paths:
    root_dir = str(ROOT_DIR)
    asset_dir = str(ROOT_DIR / "asset")
    smpl_file = str(ROOT_DIR / "models" / "SMPL_male_10PCs.pkl")
    physics_model_file = str(ROOT_DIR / "models" / "urdf" / "physics.urdf")
    plane_file = str(ROOT_DIR / "models" / "plane.urdf")
    height_map_file = str(ROOT_DIR / "asset" / "height_map.npy")
    physics_parameter_file = str(ROOT_DIR / "physics_parameters.json")


class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


vel_scale = 3
