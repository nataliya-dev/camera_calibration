#!/usr/bin/env python3
"""
Hand-Eye Calibration Script using Chessboard Pattern
Performs hand-eye calibration optimization using collected robot poses and camera images.
"""

import numpy as np
import cv2
import json
import os
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Data paths
DATA_DIR = "calibration_data"
EE_POSES_FILE = os.path.join(DATA_DIR, "ee_poses.json")
HAND_IMAGES_DIR = os.path.join(DATA_DIR, "hand_camera")
EXTERNAL_IMAGES_DIR = os.path.join(DATA_DIR, "external_camera")

# Chessboard configuration
CHESSBOARD_SIZE = (7, 6)  # Internal corners (width, height)
SQUARE_SIZE = 25.0  # Size of chessboard square in mm (adjust to your board)

# Hand camera intrinsics
HAND_CAMERA_MATRIX = np.array([
    [
        630.5633384147305,
        0.0,
        316.77844020764405
    ],
    [
        0.0,
        632.2089483760437,
        253.40738678050528
    ],
    [
        0.0,
        0.0,
        1.0
    ]
])
HAND_DIST_COEFFS = np.array(

    [
        -0.0035377891570395194,
        0.039970849526369395,
        0.006702630497626305,
        -0.0021572475245587026,
        -0.31506016344623927
    ]
)

# External camera intrinsics
EXTERNAL_CAMERA_MATRIX = np.array([
    [
        800.6245546760713,
        0.0,
        348.6044665134533
    ],
    [
        0.0,
        798.872485234719,
        246.6400986859083
    ],
    [
        0.0,
        0.0,
        1.0
    ]]
)

EXTERNAL_DIST_COEFFS = np.array(
    [
        -0.0015595918327435448,
        0.5208775389617283,
        0.008307967420407177,
        0.006815589941204983,
        -1.7941428400123502
    ]
)

# Optimization parameters
MIN_SAMPLES = 5
MAX_ITERATIONS = 1000
ROTATION_WEIGHT = 1.0
TRANSLATION_WEIGHT = 1.0

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CameraPose:
    """Represents camera pose detection result"""
    success: bool
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray


@dataclass
class CalibrationSample:
    """Single calibration sample containing all necessary data"""
    ee_position: np.ndarray
    ee_orientation: np.ndarray  # quaternion [x, y, z, w]
    hand_camera_pose: CameraPose
    external_camera_pose: CameraPose
    sample_id: str

# =============================================================================
# CHESSBOARD DETECTOR
# =============================================================================


class ChessboardDetector:
    """Handles chessboard detection in images"""

    def __init__(self, board_size: Tuple[int, int], square_size: float):
        self.board_size = board_size
        self.square_size = square_size

        # Create 3D object points for chessboard
        self.object_points = np.zeros(
            (board_size[0] * board_size[1], 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:board_size[0],
                                             0:board_size[1]].T.reshape(-1, 2)
        self.object_points *= square_size / 1000.0  # Convert mm to meters

        # Criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def detect_pose(self, image: np.ndarray, camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray) -> CameraPose:
        """Detect chessboard pose in image"""

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)

        if not ret:
            return CameraPose(False, None, None, None)

        # Refine corners
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), self.criteria)

        # Solve PnP to get pose
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            corners_refined,
            camera_matrix,
            dist_coeffs
        )

        if not success:
            return CameraPose(False, None, None, None)

        return CameraPose(True, rvec.flatten(), tvec.flatten(), corners_refined)

# =============================================================================
# DATA LOADER
# =============================================================================


class DataLoader:
    """Handles loading of calibration data from files"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ee_poses_file = os.path.join(data_dir, "ee_poses.json")
        self.hand_images_dir = os.path.join(data_dir, "hand_camera")
        self.external_images_dir = os.path.join(data_dir, "external_camera")

    def load_ee_poses(self) -> Dict:
        """Load end-effector poses from JSON file"""
        if not os.path.exists(self.ee_poses_file):
            raise FileNotFoundError(
                f"End-effector poses file not found: {self.ee_poses_file}")

        with open(self.ee_poses_file, 'r') as f:
            return json.load(f)

    def load_calibration_samples(self, detector: ChessboardDetector) -> List[CalibrationSample]:
        """Load all calibration samples"""

        print("Loading calibration data...")
        ee_poses = self.load_ee_poses()
        samples = []

        for sample_id, pose_data in ee_poses.items():
            # Load images
            hand_img_path = os.path.join(
                self.hand_images_dir, f"{sample_id}.jpg")
            external_img_path = os.path.join(
                self.external_images_dir, f"{sample_id}.jpg")

            if not (os.path.exists(hand_img_path) and os.path.exists(external_img_path)):
                print(f"Warning: Missing images for sample {sample_id}")
                continue

            hand_img = cv2.imread(hand_img_path)
            external_img = cv2.imread(external_img_path)

            if hand_img is None or external_img is None:
                print(f"Warning: Failed to load images for sample {sample_id}")
                continue

            # Detect chessboard poses
            hand_pose = detector.detect_pose(
                hand_img, HAND_CAMERA_MATRIX, HAND_DIST_COEFFS)
            external_pose = detector.detect_pose(
                external_img, EXTERNAL_CAMERA_MATRIX, EXTERNAL_DIST_COEFFS)

            # Only include samples where both cameras detect the board
            if hand_pose.success and external_pose.success:
                ee_pos = np.array(pose_data["position"])
                ee_quat = np.array(pose_data["orientation"])  # [x, y, z, w]

                sample = CalibrationSample(
                    ee_position=ee_pos,
                    ee_orientation=ee_quat,
                    hand_camera_pose=hand_pose,
                    external_camera_pose=external_pose,
                    sample_id=sample_id
                )
                samples.append(sample)
            else:
                print(
                    f"Warning: Chessboard detection failed for sample {sample_id}")

        print(f"Loaded {len(samples)} valid calibration samples")
        return samples

# =============================================================================
# HAND-EYE CALIBRATOR
# =============================================================================


class HandEyeCalibrator:
    """Performs hand-eye calibration optimization"""

    def __init__(self, samples: List[CalibrationSample]):
        self.samples = samples

    def pack_parameters(self, T_hand_eye: np.ndarray, T_base_external: np.ndarray) -> np.ndarray:
        """Pack transformation matrices into optimization parameter vector"""

        # Extract rotation (as axis-angle) and translation
        R_he = T_hand_eye[:3, :3]
        t_he = T_hand_eye[:3, 3]

        R_be = T_base_external[:3, :3]
        t_be = T_base_external[:3, 3]

        # Convert rotations to axis-angle representation
        rvec_he = R.from_matrix(R_he).as_rotvec()
        rvec_be = R.from_matrix(R_be).as_rotvec()

        # Pack: [rvec_hand_eye(3), tvec_hand_eye(3), rvec_base_external(3), tvec_base_external(3)]
        params = np.concatenate([rvec_he, t_he, rvec_be, t_be])
        return params

    def unpack_parameters(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack optimization parameters into transformation matrices"""

        rvec_he = params[0:3]
        t_he = params[3:6]
        rvec_be = params[6:9]
        t_be = params[9:12]

        # Convert axis-angle to rotation matrices
        R_he = R.from_rotvec(rvec_he).as_matrix()
        R_be = R.from_rotvec(rvec_be).as_matrix()

        # Construct transformation matrices
        T_hand_eye = np.eye(4)
        T_hand_eye[:3, :3] = R_he
        T_hand_eye[:3, 3] = t_he

        T_base_external = np.eye(4)
        T_base_external[:3, :3] = R_be
        T_base_external[:3, 3] = t_be

        return T_hand_eye, T_base_external

    def pose_to_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert rotation vector and translation to transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(rvec).as_matrix()
        T[:3, 3] = tvec
        return T

    def ee_pose_to_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Convert end-effector pose to transformation matrix"""
        T = np.eye(4)
        # quaternion is [x, y, z, w]
        T[:3, :3] = R.from_quat(quaternion).as_matrix()
        T[:3, 3] = position
        return T

    def objective_function(self, params: np.ndarray) -> float:
        """Objective function for optimization"""

        T_hand_eye, T_base_external = self.unpack_parameters(params)

        total_error = 0.0

        for sample in self.samples:
            # Get transformations
            T_base_ee = self.ee_pose_to_matrix(
                sample.ee_position, sample.ee_orientation)
            T_hand_board = self.pose_to_matrix(sample.hand_camera_pose.rvec,
                                               sample.hand_camera_pose.tvec)
            T_external_board = self.pose_to_matrix(sample.external_camera_pose.rvec,
                                                   sample.external_camera_pose.tvec)

            # Compute board poses in base frame via two different paths
            # Path 1: base -> ee -> hand_camera -> board
            T_board_base_path1 = T_base_ee @ T_hand_eye @ T_hand_board

            # Path 2: base -> external_camera -> board
            T_board_base_path2 = T_base_external @ T_external_board

            # The two paths should give the same result
            # Compute error as the difference between the two transformations
            error_matrix = T_board_base_path1 @ np.linalg.inv(
                T_board_base_path2)

            # Extract rotation and translation errors
            R_error = error_matrix[:3, :3]
            t_error = error_matrix[:3, 3]

            # Rotation error (angle of rotation matrix)
            rotation_error = np.linalg.norm(R.from_matrix(R_error).as_rotvec())

            # Translation error
            translation_error = np.linalg.norm(t_error)

            # Combined error (weighted)
            total_error += ROTATION_WEIGHT * rotation_error**2 + \
                TRANSLATION_WEIGHT * translation_error**2

        return total_error

    def calibrate(self) -> Dict:
        """Perform calibration optimization"""

        # Initialize with identity transformations
        T_hand_eye_init = np.eye(4)
        T_base_external_init = np.eye(4)

        # Add small perturbation to avoid singular initial conditions
        T_hand_eye_init[:3, 3] = [0.1, 0.0, 0.0]
        T_base_external_init[:3, 3] = [0.5, 0.5, 0.5]

        # Pack initial parameters
        initial_params = self.pack_parameters(
            T_hand_eye_init, T_base_external_init)

        print("Starting optimization...")
        print(f"Initial error: {self.objective_function(initial_params):.6f}")

        # Optimize
        result = minimize(
            self.objective_function,
            initial_params,
            method='BFGS',
            options={'disp': True, 'maxiter': MAX_ITERATIONS}
        )

        # Unpack results
        T_hand_eye_opt, T_base_external_opt = self.unpack_parameters(result.x)

        return {
            'T_hand_eye': T_hand_eye_opt,
            'T_base_external': T_base_external_opt,
            'final_error': result.fun,
            'optimization_result': result,
            'success': result.success
        }

    def evaluate_calibration(self, T_hand_eye: np.ndarray, T_base_external: np.ndarray) -> Dict:
        """Evaluate calibration quality"""

        errors = []

        for sample in self.samples:
            T_base_ee = self.ee_pose_to_matrix(
                sample.ee_position, sample.ee_orientation)
            T_hand_board = self.pose_to_matrix(sample.hand_camera_pose.rvec,
                                               sample.hand_camera_pose.tvec)
            T_external_board = self.pose_to_matrix(sample.external_camera_pose.rvec,
                                                   sample.external_camera_pose.tvec)

            # Two paths to board pose
            T_board_path1 = T_base_ee @ T_hand_eye @ T_hand_board
            T_board_path2 = T_base_external @ T_external_board

            # Error
            error_matrix = T_board_path1 @ np.linalg.inv(T_board_path2)
            R_error = error_matrix[:3, :3]
            t_error = error_matrix[:3, 3]

            rotation_error = np.linalg.norm(R.from_matrix(R_error).as_rotvec())
            translation_error = np.linalg.norm(t_error)

            errors.append({
                'sample_id': sample.sample_id,
                'rotation_error_rad': rotation_error,
                'translation_error_m': translation_error,
                'rotation_error_deg': np.degrees(rotation_error)
            })

        errors_array = np.array(
            [[e['rotation_error_deg'], e['translation_error_m']] for e in errors])

        return {
            'individual_errors': errors,
            'mean_rotation_error_deg': np.mean(errors_array[:, 0]),
            'mean_translation_error_m': np.mean(errors_array[:, 1]),
            'std_rotation_error_deg': np.std(errors_array[:, 0]),
            'std_translation_error_m': np.std(errors_array[:, 1]),
            'max_rotation_error_deg': np.max(errors_array[:, 0]),
            'max_translation_error_m': np.max(errors_array[:, 1])
        }

# =============================================================================
# MAIN CALIBRATION PIPELINE
# =============================================================================


def save_results(calibration_result: Dict, evaluation: Dict):
    """Save calibration results to files"""

    # Save transformation matrices
    np.save("T_hand_eye.npy", calibration_result['T_hand_eye'])
    np.save("T_base_external.npy", calibration_result['T_base_external'])

    # Save detailed results
    results = {
        'T_hand_eye': calibration_result['T_hand_eye'].tolist(),
        'T_base_external': calibration_result['T_base_external'].tolist(),
        'final_error': float(calibration_result['final_error']),
        'success': calibration_result['success'],
        'evaluation': evaluation
    }

    with open("calibration_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved:")
    print("- T_hand_eye.npy")
    print("- T_base_external.npy")
    print("- calibration_results.json")


def print_results(calibration_result: Dict, evaluation: Dict):
    """Print calibration results"""

    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)

    if calibration_result['success']:
        print("✓ Calibration successful!")
        print(f"Final error: {calibration_result['final_error']:.6f}")

        print(f"\nCalibration Quality:")
        print(
            f"Mean rotation error: {evaluation['mean_rotation_error_deg']:.3f}° ± {evaluation['std_rotation_error_deg']:.3f}°")
        print(
            f"Mean translation error: {evaluation['mean_translation_error_m']:.4f}m ± {evaluation['std_translation_error_m']:.4f}m")
        print(
            f"Max rotation error: {evaluation['max_rotation_error_deg']:.3f}°")
        print(
            f"Max translation error: {evaluation['max_translation_error_m']:.4f}m")

        print(f"\nHand-Eye Transformation Matrix:")
        print(calibration_result['T_hand_eye'])

        print(f"\nBase-External Camera Transformation Matrix:")
        print(calibration_result['T_base_external'])

    else:
        print("✗ Calibration failed!")
        print("Optimization did not converge.")


def main():
    """Main calibration pipeline"""

    print("="*60)
    print("HAND-EYE CALIBRATION WITH CHESSBOARD")
    print("="*60)

    print(f"Configuration:")
    print(f"- Data directory: {DATA_DIR}")
    print(f"- Chessboard size: {CHESSBOARD_SIZE}")
    print(f"- Square size: {SQUARE_SIZE}mm")
    print(f"- Minimum samples: {MIN_SAMPLES}")

    # Initialize detector
    detector = ChessboardDetector(CHESSBOARD_SIZE, SQUARE_SIZE)

    # Load data
    data_loader = DataLoader(DATA_DIR)
    samples = data_loader.load_calibration_samples(detector)

    if len(samples) < MIN_SAMPLES:
        print(
            f"\nError: Need at least {MIN_SAMPLES} valid samples for calibration")
        print(f"Found only {len(samples)} samples")
        return

    # Perform calibration
    calibrator = HandEyeCalibrator(samples)
    calibration_result = calibrator.calibrate()

    # Evaluate calibration quality
    evaluation = calibrator.evaluate_calibration(
        calibration_result['T_hand_eye'],
        calibration_result['T_base_external']
    )

    # Print and save results
    print_results(calibration_result, evaluation)
    save_results(calibration_result, evaluation)


if __name__ == "__main__":
    main()
