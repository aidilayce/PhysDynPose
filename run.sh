#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEQ_NO=""
SURFACE=""
VIDEO_PATH=""
FRAMES_DIR=""
VISUALIZE=0
SLAM=1

while [ $# -gt 0 ]; do
    case "$1" in
        --seq_no)
            SEQ_NO="$2"
            shift 2
            ;;
        --surface)
            SURFACE="$2"
            shift 2
            ;;
        --video_path)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --frames_dir|--video_dir)
            FRAMES_DIR="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE=1
            shift
            ;;
        --no-slam)
            SLAM=0
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$SEQ_NO" ] || [ -z "$SURFACE" ]; then
    echo "Usage: $0 --seq_no <id> --surface <24|31> (--video_path <video> | --frames_dir <dir>) [--visualize] [--no-slam]"
    exit 1
fi

if [ -n "${RBDL_PYTHON_PATH:-}" ]; then
    export PYTHONPATH="${PYTHONPATH:-}:$RBDL_PYTHON_PATH"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

FOURD_HUMANS_ROOT="${FOURD_HUMANS_ROOT:-$SCRIPT_DIR/../4D-Humans}"
DROID_SLAM_ROOT="${DROID_SLAM_ROOT:-$SCRIPT_DIR/../DROID-SLAM}"

EXTRACT_PYTHON="${EXTRACT_PYTHON:-python3}"
FOURD_HUMANS_PYTHON="${FOURD_HUMANS_PYTHON:-python3}"
DROID_SLAM_PYTHON="${DROID_SLAM_PYTHON:-python3}"
ALIGN_PYTHON="${ALIGN_PYTHON:-python3}"
OPT_PYTHON="${OPT_PYTHON:-python3}"
CONVERT_PYTHON="${CONVERT_PYTHON:-python3}"
VIS_PYTHON="${VIS_PYTHON:-python3}"

if [ -z "$VIDEO_PATH" ] && [ -z "$FRAMES_DIR" ]; then
    echo "Either --video_path or --frames_dir is required."
    exit 1
fi

if [ ! -d "$FOURD_HUMANS_ROOT" ]; then
    echo "4D-Humans checkout not found at $FOURD_HUMANS_ROOT"
    echo "Set FOURD_HUMANS_ROOT to your installed 4D-Humans folder."
    exit 1
fi

if [ "$SLAM" -eq 1 ] && [ ! -d "$DROID_SLAM_ROOT" ]; then
    echo "DROID-SLAM checkout not found at $DROID_SLAM_ROOT"
    echo "Set DROID_SLAM_ROOT to your installed DROID-SLAM folder."
    exit 1
fi

map_sequence_id() {
    if [ "$SURFACE" = "24" ]; then
        case "$SEQ_NO" in
            1) echo "8" ;;
            2) echo "9" ;;
            *)
                echo "Unsupported flat-scene sequence: seq$SEQ_NO" >&2
                exit 1
                ;;
        esac
    else
        echo "$SEQ_NO"
    fi
}

OUTPUT_SEQ="$(map_sequence_id)"
OUTPUT_SEQ_PADDED="$(printf "%02d" "$OUTPUT_SEQ")"
FOURD_OUTPUT_DIR="$SCRIPT_DIR/inputs/4d_humans/outputs_$OUTPUT_SEQ_PADDED"
SLAM_OUTPUT="$SCRIPT_DIR/inputs/slam/Trajectory.txt"
DEFAULT_FRAMES_DIR="$SCRIPT_DIR/outputs/raw_frames/seq${SEQ_NO}_${SURFACE}"

if [ -z "$FRAMES_DIR" ]; then
    FRAMES_DIR="$DEFAULT_FRAMES_DIR"
    "$EXTRACT_PYTHON" extract_movicam_frames.py \
        --video_path "$VIDEO_PATH" \
        --seq_no "$SEQ_NO" \
        --surface "$SURFACE" \
        --output_dir "$FRAMES_DIR"
fi

mkdir -p "$FOURD_OUTPUT_DIR" "$(dirname "$SLAM_OUTPUT")"

"$FOURD_HUMANS_PYTHON" "$FOURD_HUMANS_ROOT/demo.py" \
    --img_folder "$FRAMES_DIR" \
    --out_folder "$FOURD_OUTPUT_DIR"

if [ "$SLAM" -eq 1 ]; then
    "$DROID_SLAM_PYTHON" run_droid_slam.py \
        --droid_root "$DROID_SLAM_ROOT" \
        --imagedir "$FRAMES_DIR" \
        --seq_no "$SEQ_NO" \
        --surface "$SURFACE" \
        --output "$SLAM_OUTPUT"
    SLAM_FLAG="--slam"
else
    SLAM_FLAG="--no-slam"
fi

"$ALIGN_PYTHON" align_slam.py --seq_no "$SEQ_NO" --surface "$SURFACE" "$SLAM_FLAG"
"$OPT_PYTHON" run_opt.py --seq_no "$SEQ_NO" --surface "$SURFACE" "$SLAM_FLAG"
"$CONVERT_PYTHON" convert_res.py --seq_no "$SEQ_NO" --surface "$SURFACE" --method PIP "$SLAM_FLAG"

if [ "$VISUALIZE" -eq 1 ]; then
    "$VIS_PYTHON" visualizer.py --seq_no "$SEQ_NO" --surface "$SURFACE" --method pip "$SLAM_FLAG"
fi
