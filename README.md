# Camera Calibration Tool User Guide

This guide explains how to use the camera calibration tool to estimate intrinsic and extrinsic camera parameters from court point correspondences and physical measurements.

## Prerequisites

- Python 3.6+
- Required packages: numpy, opencv-python, plotly, pillow, matplotlib, pandas, joblib, scipy

Install dependencies:
```bash
pip install numpy opencv-python plotly pillow matplotlib pandas joblib scipy
```

## Input Files

The calibration tool requires the following input files:

1. **Court Points File**: A pickle file containing 2D image coordinates of court points
   - Format: Numpy array or list of shape (N, 2) containing (x, y) pixel coordinates

2. **Measurements File**: A JSON file containing physical measurements 
   - Format: Contains camera height and distance measurements to known 3D points
   - Example structure:
     ```json
     {
       "camera_id": {
         "3": [2.1, 6.45, 7.12, 5.89, 4.76]
       }
     }
     ```
     where 2.1 is the camera height, and other values are measured distances to reference points

3. **Reference Image** (optional): An image for visualization of reprojection results

4. **Ball Position Files** (optional): CSV files with 3D ball positions and corresponding 2D detections

## Usage

### Basic Usage

```bash
python camera_calibration.py \
  --court_points [PATH_TO_COURT_POINTS.pkl] \
  --measurements [PATH_TO_MEASUREMENTS.json] \
  --camera_id [CAMERA_ID] \
  --image [PATH_TO_REFERENCE_IMAGE] \
  --output_dir [OUTPUT_DIRECTORY]
```

### Example

```bash
python calib_cam4.py \
  --court_points ./source_data/view4/4.pkl \
  --measurements ./source_data/measurement.json \
  --camera_id 4 \
  --image ./source_data/view4/cam4-000001.png \
  --output_dir ./calib_results/view4
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--court_points` | Path to the pickle file containing court coordinates |
| `--measurements` | Path to the JSON file containing distance measurements |
| `--camera_id` | Camera ID (e.g. '0', '2', '3', '7') |
| `--image` | Path to reference image for visualization (optional) |
| `--ball_3d` | Path to 3D ball positions CSV (optional) |
| `--ball_2d` | Path to 2D ball detections CSV (optional) |
| `--output_dir` | Directory to save calibration results (default: current directory) |

## Output Files

The tool generates the following output files in the specified output directory:

1. **Extrinsic Matrix**: `Cam_[CAMERA_ID]_extrinsic.npy`
   - A 3×4 matrix combining rotation (R) and translation (t)

2. **Intrinsic Parameters**: `Cam_[CAMERA_ID]_intrinsic.npy`
   - A 1×9 array containing [fx, fy, cx, cy, k1, k2, k3, p1, p2]
   - fx, fy: Focal lengths
   - cx, cy: Principal point coordinates
   - k1, k2, k3: Radial distortion coefficients
   - p1, p2: Tangential distortion coefficients

3. **Visualizations** (if reference image is provided):
   - `camera_pose.html`: 3D visualization of camera position relative to court
   - `reprojection.png`: 2D visualization showing original and reprojected points

## Calibration Process

The tool performs calibration in two main stages:

1. **Focal Length Optimization**: Grid search to find optimal fx and fy values
2. **Distortion Coefficient Optimization**: Grid search for k1, k2, p1, p2 values

Progress for each stage is displayed with progress bars and status updates.

## Troubleshooting

- **"Failed to load data"**: Check that input files exist and have the correct format
- **"Calibration failed"**: Try adjusting the court point selections or measurement values
- **Slow performance**: Reduce the `steps` parameter in the code for faster but less precise results

## Example Output

After successful calibration, you'll see output similar to:

```
Loading data for camera 3...
Starting calibration process...
Starting grid search for focal lengths (30x30 grid)...
[====================>] 100%
Best focal lengths: fx=1835.42, fy=1872.16 (cost=0.0082)
Resulting camera position: [-2.59  2.1  -3.42]
Starting parallel grid search for distortion coefficients...
Total parameter combinations to evaluate: 50625
Evaluating distortion parameters: 100%|██████████| 50625/50625
Best distortion parameters: [0.12, -0.24, 0.01, -0.02] (cost=0.0067)
Final camera position: [-2.59  2.1  -3.42]
Extrinsic matrix saved to ./calibration_results/Cam_3_extrinsic.npy
Intrinsic parameters saved to ./calibration_results/Cam_3_intrinsic.npy
3D visualization saved to ./calibration_results/camera_pose.html
2D visualization saved to ./calibration_results/reprojection.png
Calibration completed successfully.
```
