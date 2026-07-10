# MoviCam Dataset Layout

This document describes the released MoviCam dataset archive and how it relates to the PhysDynPose code release.

## Release Archives

The release uses two data archives:

- [MoviCam raw dataset archive](https://edmond.mpg.de/file.xhtml?fileId=345390&version=1.0)
- [PhysDynPose inputs archive](https://edmond.mpg.de/file.xhtml?fileId=345391&version=1.0)

### MoviCam Raw Dataset Archive

This is the Edmond archive. After extraction it should contain:

```text
Data/
  Recordings_31_03_23/
  Recordings_24_03_23/
  indoor_31_03_23/
  indoor_24_03_23/
```

This archive contains raw moving-camera recordings, Captury recording exports, processed image sequences with ground truth, camera summaries, and the non-flat scene mesh.

### PhysDynPose Inputs Archive

This smaller archive is extracted inside the PhysDynPose repository:

```text
PhysDynPose/
  inputs/
    recordings/
    pip_reference/dataset_captury/
```

It contains contact labels and PIP reference pose/shape files. It does not contain `outputs/`.

## Connecting `Data/` To The Code

- `Data/Recordings_*` is the raw Edmond dataset layout. It contains the raw recording-side material: synchronized moving-camera videos, Captury exports, moving-camera calibration/trajectory files, SMPL joints, and contact labels.
- `PhysDynPose/inputs/recordings/Recordings_*` is the code-side location expected by the release scripts. In the small PhysDynPose inputs archive, this tree is only a minimal processed overlay. It is not a second full copy of the raw Edmond dataset. You can use these inputs directly to run PhysDynPose.

For the full pipeline, make the raw Edmond recording folders available at the code-side path:

```bash
cd /path/to/PhysDynPose
mkdir -p inputs/recordings
ln -s /path/to/Data/Recordings_31_03_23 inputs/recordings/Recordings_31_03_23
ln -s /path/to/Data/Recordings_24_03_23 inputs/recordings/Recordings_24_03_23
```

Alternatively, copy those two raw folders into `inputs/recordings/`.

If the small `inputs/` archive has already been extracted, do not blindly overwrite it. Either use the raw Edmond `Data/Recordings_*` folders as the authoritative `inputs/recordings/Recordings_*` trees, or merge only the missing raw subfolders into the existing code-side trees. The important non-raw part of the small inputs archive is:

```text
inputs/pip_reference/dataset_captury/
```

## Released Sequences

Non-flat scene:

```text
surface=31: seq1, seq2, seq3, seq4, seq6, seq7
```

Flat scene:

```text
surface=24: seq1, seq2
```

The code maps the flat-scene sequences to PIP reference sequence IDs:

```text
surface=24, seq1 -> seq8
surface=24, seq2 -> seq9
```

## Raw Recording Folders

The raw recording folders are:

```text
Data/Recordings_31_03_23/seq1_captury/
Data/Recordings_31_03_23/seq2_captury/
Data/Recordings_31_03_23/seq3_captury/
Data/Recordings_31_03_23/seq4_captury/
Data/Recordings_31_03_23/seq6_captury/
Data/Recordings_31_03_23/seq7_captury/
Data/Recordings_24_03_23/seq1_captury/
Data/Recordings_24_03_23/seq2_captury/
```

Each sequence may include:

- `stream*.mp4`: static multiview Captury videos
- `camera.calib`, `camSorting.txt`, `settings.txt`, `meta.txt`: Captury recording metadata
- `seq*.motion` and `seq*.skeleton`: Captury/Skeletool motion exports
- `moving_cam/`: moving camera trajectory files
- `smpl/`: SMPL-derived ground-truth joints and related exports
- `labels_con_sta/`: foot contact labels

## Files Used By The PhysDynPose Pipeline

For each released sequence, the PhysDynPose preprocessing and optimization code uses:

```text
seq*_captury/moving_cam/camera_exts_full.txt
seq*_captury/moving_cam/cam_ext.npy
seq*_captury/moving_cam/cam_ext_slam_first.npy
seq*_captury/labels_con_sta/pred_con_gt.npy
seq*_captury/smpl/3d_joints.npy
```

The synced moving-camera videos are:

```text
Data/Recordings_31_03_23/SonyRX0_31_03_23/sync/seq*_captury/SonyRX0_motion_00*.mp4
Data/Recordings_24_03_23/SonyRX0/sync/seq*_captury/SonyRX0_motion_00*.mp4
```

The `SonyRX0_shot_00*.mp4` files are included as additional synchronized captures but the default pipeline examples use `SonyRX0_motion_00*.mp4`.

## Processed Benchmark Folders

The processed MoviCam image/ground-truth benchmark folders are:

```text
Data/indoor_31_03_23/Cam_seq/mov01/
Data/indoor_31_03_23/Cam_seq/mov02/
Data/indoor_31_03_23/Cam_seq/mov03/
Data/indoor_31_03_23/Cam_seq/mov04/
Data/indoor_31_03_23/Cam_seq/mov06/
Data/indoor_31_03_23/Cam_seq/mov07/
Data/indoor_24_03_23/Cam_seq/mov01/
Data/indoor_24_03_23/Cam_seq/mov02/
```

Each `movXX/` folder contains image frames and:

```text
all_pose_gt.pkl
```

For `mov07`, image frames are stored under:

```text
Data/indoor_31_03_23/Cam_seq/mov07/0/
```

The processed benchmark camera summaries are:

```text
Data/indoor_31_03_23/Cameras/hand_eye_calibration.pkl
Data/indoor_31_03_23/Cameras/moving_cam.pkl
Data/indoor_24_03_23/Cameras/hand_eye_calibration.pkl
Data/indoor_24_03_23/Cameras/moving_cam.pkl
```

## Scene Data

The processed non-flat scene mesh is:

```text
Data/indoor_31_03_23/scene_gt/no_wall.obj
Data/indoor_31_03_23/scene_gt/no_wall.obj.mtl
Data/indoor_31_03_23/scene_gt/model.jpg
```

The PhysDynPose optimizer itself uses the compact scene representation bundled with the code:

```text
asset/height_map.npy
asset/scene.urdf
asset/no_wall_flip.obj.gz
```

`asset/height_map.npy` is queried during optimization. The compressed scene mesh is unpacked by [visualizer.py](visualizer.py) as `asset/no_wall_flip.obj` when scene visualization is enabled. The raw `scene_gt/no_wall.obj` mesh is useful for inspection, benchmark documentation, and regenerating scene-derived assets, but it is not loaded directly by `run_opt.py`.
