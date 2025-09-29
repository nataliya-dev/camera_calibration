#!/usr/bin/env python3
"""
Hand-Eye Calibration using AXXB solver
Converts collected data and solves for camera-to-base transformation
"""

import numpy as np
import cv2
import json
import os
from scipy.spatial.transform import Rotation as R

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data directory containing collected images and poses
DATA_DIR = "calibration_data"
EE_POSES_FILE = os.path.join(DATA_DIR, "ee_poses.json")

# Camera directory to calibrate (from your data collection)
CAMERA_DIR = "ext2"  # or "r1", "r2", etc.

# Chessboard configuration (MUST match your calibration board)
CHESSBOARD_SIZE = (7, 6)  # Internal corners (width, height)
SQUARE_SIZE = 25.0  # Size of square in mm

# Calibration option
# "EH" = eye-on-hand (camera on robot end-effector, board is stationary)
# "EBCB" = eye-on-base (camera is stationary, board on end-effector)
CALIBRATION_OPTION = "EBCB"

# Output files
OUTPUT_DIR = DATA_DIR
CONVERTED_DATA_DIR = os.path.join(DATA_DIR, "converted_axxb_data")

# =============================================================================
# UTILITY FUNCTIONS (from calibration_toolbox)
# =============================================================================


def pose_inv(pose):
    """Invert a homogeneous transformation matrix"""
    R_inv = pose[:3, :3].T
    t_inv = -np.dot(R_inv, pose[:3, 3])
    pose_inv = np.eye(4)
    pose_inv[:3, :3] = R_inv
    pose_inv[:3, 3] = t_inv
    return pose_inv


def get_mat_log(R):
    """Get the logarithm map of rotation matrix"""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if np.abs(theta) < 1e-10:
        return np.zeros(3)
    log_R = theta / (2 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    return log_R


def rotm2quat(R):
    """Convert rotation matrix to quaternion [w, x, y, z]"""
    q = np.zeros(4)
    q[0] = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
    q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
    q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])
    return q


# =============================================================================
# CHESSBOARD DETECTION
# =============================================================================

def detect_chessboard_pose(image_path, chessboard_size, square_size, camera_matrix, dist_coeffs):
    """
    Detect chessboard and compute its pose in camera frame

    Returns:
        4x4 homogeneous transformation matrix or None if detection fails
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if not ret:
        print(f"Chessboard not found in: {image_path}")
        return None

    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Generate object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size / 1000.0  # Convert mm to meters

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        objp, corners, camera_matrix, dist_coeffs)

    if not success:
        print(f"PnP failed for: {image_path}")
        return None

    # Convert to homogeneous transformation
    rot_mat, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4)
    pose[:3, :3] = rot_mat
    pose[:3, 3] = tvec.flatten()

    return pose


# =============================================================================
# DATA CONVERSION
# =============================================================================

def convert_data_for_axxb(data_dir, camera_dir, ee_poses_file, chessboard_size,
                          square_size, output_dir):
    """
    Convert collected data to AXXB format
    Creates robotpose.txt and markerpose.txt files for each sample
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load end-effector poses
    with open(ee_poses_file, 'r') as f:
        ee_poses = json.load(f)

    # Load camera intrinsics (assuming you saved them)
    intrinsics_file = os.path.join(data_dir, "camera_intrinsics.json")
    if os.path.exists(intrinsics_file):
        with open(intrinsics_file, 'r') as f:
            intrinsics_data = json.load(f)
        camera_matrix = np.array(intrinsics_data['camera_matrix'])
        dist_coeffs = np.array(intrinsics_data['distortion_coefficients'])
        print(f"Loaded camera intrinsics from {intrinsics_file}")
    else:
        print("Warning: No camera intrinsics file found. Using default values.")
        print("Please run the intrinsics export script first!")
        # Default values - REPLACE WITH YOUR ACTUAL INTRINSICS
        camera_matrix = np.array([
            [1542.30, 0.0, 964.17],
            [0.0, 1542.30, 561.23],
            [0.0, 0.0, 1.0]
        ])
        dist_coeffs = np.array(
            [10.003107, -95.531288, 0.002026, 0.003359, 321.020020])

    print(f"\nCamera Matrix:\n{camera_matrix}")
    print(f"\nDistortion Coefficients:\n{dist_coeffs}")

    # Get image directory
    image_dir = os.path.join(data_dir, camera_dir)

    successful_conversions = 0
    failed_conversions = 0

    # Process each sample
    for sample_id, pose_data in sorted(ee_poses.items()):
        print(f"\nProcessing {sample_id}...")

        # Image path
        image_path = os.path.join(image_dir, f"{sample_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            failed_conversions += 1
            continue

        # Detect chessboard pose (marker in camera frame)
        marker_pose = detect_chessboard_pose(
            image_path, chessboard_size, square_size, camera_matrix, dist_coeffs
        )

        if marker_pose is None:
            failed_conversions += 1
            continue

        # Convert robot pose to homogeneous matrix (base to end-effector)
        position = np.array(pose_data['position'])
        quaternion = np.array(pose_data['orientation'])  # [x, y, z, w]

        # Convert quaternion to rotation matrix
        rot = R.from_quat(quaternion)
        robot_pose = np.eye(4)
        robot_pose[:3, :3] = rot.as_matrix()
        robot_pose[:3, 3] = position

        # Save robot pose
        robot_pose_file = os.path.join(
            output_dir, f"{sample_id}_robotpose.txt")
        with open(robot_pose_file, 'w') as f:
            f.write(' '.join(map(str, robot_pose.flatten())))

        # Save marker pose
        marker_pose_file = os.path.join(
            output_dir, f"{sample_id}_markerpose.txt")
        with open(marker_pose_file, 'w') as f:
            f.write(' '.join(map(str, marker_pose.flatten())))

        print(f"✓ Converted {sample_id}")
        successful_conversions += 1

    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"Successful: {successful_conversions}")
    print(f"Failed: {failed_conversions}")
    print(f"{'='*60}")

    return successful_conversions > 0


# =============================================================================
# AXXB SOLVER
# =============================================================================

def load_axxb_data(data_dir):
    """Load robot and marker poses from directory"""
    robot_poses = []
    marker_poses = []
    pose_indices = []

    for f in sorted(os.listdir(data_dir)):
        if 'robotpose.txt' in f:
            pose_idx = f.split('_')[0]
            pose_indices.append(pose_idx)

            # Load robot pose
            with open(os.path.join(data_dir, f), 'r') as file:
                robotpose_str = file.readline().split(' ')
                robotpose = [float(x) for x in robotpose_str if x != '']
                assert len(robotpose) == 16
                robotpose = np.reshape(np.array(robotpose), (4, 4))
            robot_poses.append(robotpose)

            # Load marker pose
            marker_file = f.replace('robotpose.txt', 'markerpose.txt')
            with open(os.path.join(data_dir, marker_file), 'r') as file:
                markerpose_str = file.readline().split(' ')
                markerpose = [float(x) for x in markerpose_str if x != '']
                assert len(markerpose) == 16
                markerpose = np.reshape(np.array(markerpose), (4, 4))
            marker_poses.append(markerpose)

    return robot_poses, marker_poses, pose_indices


def solve_axxb(robot_poses, marker_poses, option="EH"):
    """
    Solve AX=XB using Park & Martin's method

    option: "EH" (eye-on-hand), "EBCB" (eye-on-base camera to base), "EBME" (eye-on-base marker to ee)
    """
    n = len(robot_poses)
    print(f"\nSolving AXXB with {n} poses...")

    pose_inds = np.arange(n)
    np.random.shuffle(pose_inds)

    A = np.zeros((4, 4, n-1))
    B = np.zeros((4, 4, n-1))

    M = np.zeros((3, 3))

    for i in range(n-1):
        if option == "EH":
            A[:, :, i] = np.matmul(
                pose_inv(robot_poses[pose_inds[i+1]]), robot_poses[pose_inds[i]])
            B[:, :, i] = np.matmul(
                marker_poses[pose_inds[i+1]], pose_inv(marker_poses[pose_inds[i]]))
        elif option == "EBME":
            A[:, :, i] = np.matmul(
                pose_inv(robot_poses[pose_inds[i+1]]), robot_poses[pose_inds[i]])
            B[:, :, i] = np.matmul(
                pose_inv(marker_poses[pose_inds[i+1]]), marker_poses[pose_inds[i]])
        elif option == "EBCB":
            A[:, :, i] = np.matmul(
                robot_poses[pose_inds[i+1]], pose_inv(robot_poses[pose_inds[i]]))
            B[:, :, i] = np.matmul(
                marker_poses[pose_inds[i+1]], pose_inv(marker_poses[pose_inds[i]]))

        alpha = get_mat_log(A[:3, :3, i])
        beta = get_mat_log(B[:3, :3, i])

        if np.sum(np.isnan(alpha)) + np.sum(np.isnan(beta)) > 0:
            continue

        M += np.outer(beta, alpha)

    # Get rotation matrix
    mtm = np.matmul(M.T, M)
    u_mtm, s_mtm, vh_mtm = np.linalg.svd(mtm)
    R_calib = np.matmul(
        np.matmul(np.matmul(u_mtm, np.diag(np.power(s_mtm, -0.5))), vh_mtm), M.T)

    # Get translation vector
    I_Ra_Left = np.zeros((3*(n-1), 3))
    ta_Rtb_Right = np.zeros((3*(n-1), 1))

    for i in range(n-1):
        I_Ra_Left[(3*i):(3*(i+1)), :] = np.eye(3) - A[:3, :3, i]
        ta_Rtb_Right[(3*i):(3*(i+1)), :] = np.reshape(A[:3, 3,
                                                        i] - np.dot(R_calib, B[:3, 3, i]), (3, 1))

    t = np.linalg.lstsq(I_Ra_Left, ta_Rtb_Right, rcond=-1)[0]

    calib_pose = np.c_[R_calib, t]
    calib_pose = np.r_[calib_pose, [[0, 0, 0, 1]]]

    return calib_pose


def test_calibration(robot_poses, marker_poses, calib_pose, pose_indices, option="EH"):
    """Test calibration accuracy"""
    print(f"\n{'='*60}")
    print("Testing Calibration Accuracy")
    print(f"{'='*60}")

    errors = []
    for i, pose_idx in enumerate(pose_indices):
        if option == "EH":
            check_pose = np.matmul(
                np.matmul(robot_poses[i], calib_pose), marker_poses[i])
        elif option == "EBME":
            check_pose = np.matmul(
                np.matmul(robot_poses[i], calib_pose), pose_inv(marker_poses[i]))
        elif option == "EBCB":
            check_pose = np.matmul(
                np.matmul(pose_inv(robot_poses[i]), calib_pose), marker_poses[i])

        # Calculate error (deviation from identity in position)
        # For a perfect calibration, check_pose should be constant across all samples
        if i == 0:
            reference_pose = check_pose.copy()

        pos_error = np.linalg.norm(check_pose[:3, 3] - reference_pose[:3, 3])
        errors.append(pos_error)

        print(f"Sample {pose_idx}: position error = {pos_error:.6f} m")

    print(f"\nMean error: {np.mean(errors):.6f} m")
    print(f"Max error: {np.max(errors):.6f} m")
    print(f"Std error: {np.std(errors):.6f} m")


def save_calibration(calib_pose, output_dir, option="EH"):
    """Save calibration results"""
    option_map = {"EBCB": "cam_to_base",
                  "EBME": "marker_to_ee", "EH": "cam_to_ee"}
    option_str = option_map[option]

    # Save as text file
    calib_file = os.path.join(output_dir, f"pose_{option_str}.txt")
    with open(calib_file, 'w') as f:
        f.write(' '.join(map(str, calib_pose.flatten())))

    # Save as JSON
    calib_json = os.path.join(output_dir, f"pose_{option_str}.json")

    position = calib_pose[:3, 3].tolist()
    rotation_matrix = calib_pose[:3, :3]
    quaternion = rotm2quat(rotation_matrix).tolist()  # [w, x, y, z]

    calib_data = {
        option_str: {
            "translation": {
                "x": position[0],
                "y": position[1],
                "z": position[2]
            },
            "quaternion": {
                "w": quaternion[0],
                "x": quaternion[1],
                "y": quaternion[2],
                "z": quaternion[3]
            },
            "matrix": calib_pose.tolist()
        }
    }

    with open(calib_json, 'w') as f:
        json.dump(calib_data, f, indent=2)

    print(f"\n✓ Calibration saved to:")
    print(f"  {calib_file}")
    print(f"  {calib_json}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("AXXB Hand-Eye Calibration")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Camera directory: {CAMERA_DIR}")
    print(f"Calibration option: {CALIBRATION_OPTION}")
    print(
        f"Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}, {SQUARE_SIZE}mm squares")

    # Step 1: Convert data
    print(f"\n{'='*60}")
    print("Step 1: Converting data to AXXB format")
    print(f"{'='*60}")

    success = convert_data_for_axxb(
        DATA_DIR, CAMERA_DIR, EE_POSES_FILE,
        CHESSBOARD_SIZE, SQUARE_SIZE, CONVERTED_DATA_DIR
    )

    if not success:
        print("Error: Data conversion failed")
        return

    # Step 2: Load converted data
    print(f"\n{'='*60}")
    print("Step 2: Loading converted data")
    print(f"{'='*60}")

    robot_poses, marker_poses, pose_indices = load_axxb_data(
        CONVERTED_DATA_DIR)
    print(f"Loaded {len(robot_poses)} pose pairs")

    if len(robot_poses) < 3:
        print("Error: Need at least 3 valid poses for calibration")
        return

    # Step 3: Solve AXXB
    print(f"\n{'='*60}")
    print("Step 3: Solving AXXB")
    print(f"{'='*60}")

    calib_pose = solve_axxb(robot_poses, marker_poses, CALIBRATION_OPTION)

    print(f"\nCalibration Result:")
    print(calib_pose)

    # Step 4: Test calibration
    test_calibration(robot_poses, marker_poses, calib_pose,
                     pose_indices, CALIBRATION_OPTION)

    # Step 5: Save results
    save_calibration(calib_pose, OUTPUT_DIR, CALIBRATION_OPTION)

    print(f"\n{'='*60}")
    print("✓ Calibration Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
