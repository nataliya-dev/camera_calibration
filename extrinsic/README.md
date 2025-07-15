# Camera-Robot Extrinsic Calibration

This package provides a complete solution for calibrating the extrinsic parameters of a dual-camera robot system with a hand-mounted camera and an external camera.

## Overview

The system solves for two transformation matrices:
1. **T_hand_eye**: Transformation from robot end-effector to hand-mounted camera
2. **T_base_external**: Transformation from robot base to external camera

## Requirements

```bash
pip install numpy opencv-python scipy matplotlib json glob dataclasses
```

## Setup

### 1. Hardware Requirements
- Robot arm with known forward/inverse kinematics
- Hand-mounted camera on robot end-effector
- External camera with fixed position
- ChArUco calibration board

### 2. Camera Calibration
First, calibrate both cameras individually using checkerboard patterns:

```python
from calibration_utils import calibrate_camera

# Calibrate hand camera
hand_camera_matrix, hand_dist_coeffs = calibrate_camera("hand_camera_calib/*.jpg")

# Calibrate external camera  
external_camera_matrix, external_dist_coeffs = calibrate_camera("external_camera_calib/*.jpg")
```

### 3. ChArUco Board Generation
Generate and print the ChArUco board:

```python
from calibration_utils import generate_charuco_board
generate_charuco_board(squares_x=7, squares_y=5, square_length=40, marker_length=20)
```

**Important**: Print at exactly 300 DPI for correct scale. Measure the printed squares to verify size.

## Data Collection

### Directory Structure
```
calibration_data/
├── ee_poses.json
├── hand_camera/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── ...
└── external_camera/
    ├── sample_001.jpg
    ├── sample_002.jpg
    └── ...
```

### End-Effector Poses Format (ee_poses.json)
```json
{
  "sample_001": {
    "position": [0.5, 0.2, 0.3],
    "orientation": [0.0, 0.0, 0.0, 1.0]
  },
  "sample_002": {
    "position": [0.4, 0.3, 0.35], 
    "orientation": [0.1, 0.0, 0.0, 0.995]
  }
}
```

**Note**: 
- Position in meters [x, y, z]
- Orientation as quaternion [x, y, z, w]

### Data Collection Strategy

1. **Workspace Coverage**: Move robot to 15-30 different positions covering the workspace
2. **Orientation Diversity**: Vary end-effector orientation significantly
3. **Board Visibility**: Ensure ChArUco board is visible to both cameras
4. **Lighting**: Maintain consistent, good lighting conditions

```python
# Example data collection with your robot API
def collect_calibration_data(robot, num_samples=20):
    ee_poses = {}
    
    for i in range(num_samples):
        sample_id = f"sample_{i:03d}"
        
        # Move to new position
        joint_angles = generate_random_joint_angles()
        robot.move_to_joint_angles(joint_angles)
        
        # Get pose
        ee_position, ee_orientation = robot.get_ee_pose()
        ee_poses[sample_id] = {
            "position": ee_position.tolist(),
            "orientation": ee_orientation.tolist()
        }
        
        # Capture images
        hand_image = robot.get_hand_camera_image()
        external_image = get_external_camera_image()
        
        # Save
        cv2.imwrite(f"calibration_data/hand_camera/{sample_id}.jpg", hand_image)
        cv2.imwrite(f"calibration_data/external_camera/{sample_id}.jpg", external_image)
    
    # Save poses
    with open("calibration_data/ee_poses.json", "w") as f:
        json.dump(ee_poses, f, indent=2)
```

## Usage

### 1. Basic Calibration

```python
from camera_robot_calibration import *

# Camera intrinsics (from your camera calibration)
hand_camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
hand_dist_coeffs = np.array([0.1, -0.2, 0, 0, 0])

external_camera_matrix = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]])
external_dist_coeffs = np.array([0.05, -0.1, 0, 0, 0])

# Initialize detector
detector = CharucoDetector(squares_x=7, squares_y=5, square_length=0.04, marker_length=0.02)

# Load data
data_loader = DataLoader("calibration_data")
samples = data_loader.load_calibration_samples(
    hand_camera_matrix, hand_dist_coeffs,
    external_camera_matrix, external_dist_coeffs,
    detector
)

# Perform calibration
calibrator = HandEyeCalibrator(samples)
result = calibrator.calibrate()

if result['success']:
    print("Calibration successful!")
    print(f"Final error: {result['final_error']:.6f}")
    
    # Save results
    np.save("T_hand_eye.npy", result['T_hand_eye'])
    np.save("T_base_external.npy", result['T_base_external'])
```

### 2. Data Quality Assessment

```python
from validation_utils import DataQualityChecker

# Check data quality before calibration
quality_checker = DataQualityChecker(samples)
report = quality_checker.generate_report()
print(report)
```

### 3. Validation and Visualization

```python
from validation_utils import CalibrationValidator

# Load calibration results
T_hand_eye = np.load("T_hand_eye.npy")
T_base_external = np.load("T_base_external.npy")

# Create validator
validator = CalibrationValidator(T_hand_eye, T_base_external)

# Visualize setup
validator.visualize_setup(samples)

# Evaluate calibration
evaluation = calibrator.evaluate_calibration(T_hand_eye, T_base_external)
validator.plot_error_analysis(evaluation)
```

### 4. Real-time Testing

```python
# Test with new robot position
test_result = validator.test_new_position(
    ee_position=new_ee_pos,
    ee_orientation=new_ee_quat,
    hand_camera_image=hand_img,
    external_camera_image=external_img,
    hand_camera_matrix=hand_camera_matrix,
    hand_dist_coeffs=hand_dist_coeffs,
    external_camera_matrix=external_camera_matrix,
    external_dist_coeffs=external_dist_coeffs,
    detector=detector
)

if test_result['success']:
    print(f"Rotation error: {test_result['rotation_error_deg']:.2f}°")
    print(f"Translation error: {test_result['translation_error_m']:.4f}m")
```

## Key Features

### 1. ChArUco Board Detection
- Uses OpenCV's `cv2.aruco.CharucoBoard` for robust detection
- Automatic corner refinement with subpixel accuracy
- Handles partial board visibility

### 2. Optimization Strategy
- **Rotation Representation**: Uses axis-angle (Rodrigues) representation for numerical stability
- **Objective Function**: Minimizes reprojection error between two camera paths
- **Solver**: BFGS optimization for fast convergence

### 3. Robust Error Handling
- Validates ChArUco detection success
- Checks for minimum number of detected corners
- Provides detailed error analysis

### 4. Comprehensive Validation
- Workspace coverage analysis
- Orientation diversity assessment
- Detection quality metrics
- Real-time testing capabilities

## Expected Accuracy

With good data collection:
- **Rotation Error**: < 1-2 degrees
- **Translation Error**: < 2-5 mm

## Troubleshooting

### Common Issues:

1. **Low Detection Success Rate**
   - Improve lighting conditions
   - Ensure board is flat and well-positioned
   - Check camera focus and exposure

2. **Poor Convergence**
   - Increase workspace coverage
   - Add more orientation diversity
   - Collect more samples (20-30 recommended)

3. **High Calibration Error**
   - Verify camera intrinsics are accurate
   - Check end-effector pose accuracy
   - Ensure ChArUco board dimensions are correct

### Validation Checklist:

- [ ] Camera intrinsics calibrated accurately
- [ ] ChArUco board printed at correct scale
- [ ] 15+ calibration samples collected
- [ ] Good workspace coverage achieved
- [ ] Both cameras detect board in all samples
- [ ] End-effector poses recorded accurately
- [ ] Final calibration error < 0.01

## Advanced Usage

### Custom Initial Guess
```python
# Provide better initial guess
T_hand_eye_init = np.eye(4)
T_hand_eye_init[:3, 3] = [0.05, 0.0, 0.1]  # Expected camera offset

T_base_external_init = np.eye(4)
T_base_external_init[:3, 3] = [0.5, 0.5, 1.0]  # Expected external camera position

result = calibrator.calibrate(initial_guess=(T_hand_eye_init, T_base_external_init))
```

### Multi-Board Calibration
For enhanced accuracy, you can modify the system to use multiple boards at different positions.

### Integration with Robot Control
```python
# Real-time calibration validation during robot operation
def validate_during_operation(robot, validator):
    while robot.is_running():
        ee_pos, ee_quat = robot.get_ee_pose()
        hand_img = robot.get_hand_camera_image()
        ext_img = get_external_camera_image()
        
        result = validator.test_new_position(ee_pos, ee_quat, hand_img, ext_img, ...)
        if result['success'] and result['translation_error_m'] > 0.01:
            print("Warning: Calibration may need updating")
```

## References

This implementation is based on the classic hand-eye calibration problem and extends it to dual-camera systems. The optimization uses axis-angle representation for rotations to ensure numerical stability and convergence.

Key papers:
- Tsai, R.Y. & Lenz, R.K. "A new technique for fully autonomous and efficient 3D robotics hand/eye calibration"
- Horaud, R. & Dornaika, F. "Hand-eye calibration"