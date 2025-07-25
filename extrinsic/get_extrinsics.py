#!/usr/bin/env python3
"""
Multi-Camera Hand-Eye Calibration System
Python implementation of the C++ hand-eye calibration library
Supports any number of cameras with individual intrinsics and image sets
"""

import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import glob
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Data paths
DATA_FOLDER = "calibration_data"
POSES_FILE = "ee_poses.json"
RESULTS_FOLDER = "results"

# Camera configuration - Add/modify cameras here
CAMERAS = [
    {
        "id": 1,
        "folder": "external_camera_1",
        "intrinsics_file": "calibration_results_20250721_135454.json"
    },
    {
        "id": 2, 
        "folder": "external_camera_2",
        "intrinsics_file": "calibration_results_camera_2.json"
    }
    # Add more cameras as needed:
    # {
    #     "id": 3,
    #     "folder": "hand_camera",
    #     "intrinsics_file": "hand_camera_intrinsics.json"
    # }
]

# Calibration setup
CALIBRATION_SETUP = 0  # 0: eye-in-hand, 1: eye-on-base
PATTERN_TYPE = "checkerboard"
CHECKERBOARD_ROWS = 6  # Inner corners
CHECKERBOARD_COLS = 7  # Inner corners
SQUARE_SIZE = 0.025  # 25mm in meters
RESIZE_FACTOR = 1.0
VISUAL_ERROR = True

# Optimization parameters
MAX_ITERATIONS = 3000
LOSS_SCALE = 1.0
CONVERGENCE_TOL = 1e-8

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CalibrationInfo:
    number_of_cameras: int
    camera_folder_prefix: str
    pattern_type: str
    number_of_rows: int
    number_of_columns: int
    size: float
    resize_factor: float
    visual_error: bool
    calibration_setup: int

@dataclass
class CameraInfo:
    camera_id: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray
    img_width: int
    img_height: int
    
    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]], dtype=np.float64)

@dataclass
class DetectionResults:
    correct_images: List[List[np.ndarray]]
    correct_poses: List[List[np.ndarray]]
    correct_corners: List[List[List[np.ndarray]]]
    cross_observation_matrix: List[List[int]]
    object_points: np.ndarray

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class Utils:
    @staticmethod
    def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
        """Convert rotation vector to rotation matrix"""
        return R.from_rotvec(rvec.flatten()).as_matrix()
    
    @staticmethod
    def matrix_to_rodrigues(rmat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to rotation vector"""
        return R.from_matrix(rmat).as_rotvec().reshape(-1, 1)
    
    @staticmethod
    def pose_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert rotation vector and translation to 4x4 transformation matrix"""
        rmat = Utils.rodrigues_to_matrix(rvec)
        T = np.eye(4)
        T[:3, :3] = rmat
        T[:3, 3] = tvec.flatten()
        return T
    
    @staticmethod
    def matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 4x4 transformation matrix to rotation vector and translation"""
        rmat = T[:3, :3]
        tvec = T[:3, 3].reshape(-1, 1)
        rvec = Utils.matrix_to_rodrigues(rmat)
        return rvec, tvec
    
    @staticmethod
    def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [x,y,z,w] to 4x4 transformation matrix"""
        return R.from_quat(quat).as_matrix()

# =============================================================================
# DATA READER
# =============================================================================

class DataReader:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.cameras = CAMERAS
        print(f"Configured {len(self.cameras)} cameras:")
        for cam in self.cameras:
            print(f"  Camera {cam['id']}: {cam['folder']} -> {cam['intrinsics_file']}")
    
    def read_calibration_info(self) -> CalibrationInfo:
        """Read calibration configuration"""
        return CalibrationInfo(
            number_of_cameras=len(self.cameras),
            camera_folder_prefix="",  # Not used anymore
            pattern_type=PATTERN_TYPE,
            number_of_rows=CHECKERBOARD_ROWS,
            number_of_columns=CHECKERBOARD_COLS,
            size=SQUARE_SIZE,
            resize_factor=RESIZE_FACTOR,
            visual_error=VISUAL_ERROR,
            calibration_setup=CALIBRATION_SETUP
        )
    
    def read_camera_info(self, camera_config: Dict) -> CameraInfo:
        """Read camera intrinsic parameters from JSON file"""
        intrinsics_path = os.path.join(self.data_folder, camera_config['intrinsics_file'])
        print(f"Reading intrinsics for camera {camera_config['id']}: {camera_config['intrinsics_file']}")
        
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
        
        with open(intrinsics_path, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['dist_coeffs']).flatten()
        
        return CameraInfo(
            camera_id=camera_config['id'],
            fx=camera_matrix[0, 0] * RESIZE_FACTOR,
            fy=camera_matrix[1, 1] * RESIZE_FACTOR,
            cx=camera_matrix[0, 2] * RESIZE_FACTOR,
            cy=camera_matrix[1, 2] * RESIZE_FACTOR,
            dist_coeffs=dist_coeffs,
            img_width=int(data['image_size'][0] * RESIZE_FACTOR),
            img_height=int(data['image_size'][1] * RESIZE_FACTOR)
        )
    
    def read_images(self, camera_config: Dict) -> List[np.ndarray]:
        """Read images for a specific camera"""
        camera_folder = os.path.join(self.data_folder, camera_config['folder'])
        
        if not os.path.exists(camera_folder):
            print(f"Warning: Camera folder not found: {camera_folder}")
            return []
        
        # Support multiple image formats and naming conventions
        image_patterns = [
            os.path.join(camera_folder, "sample_*.jpg"),
            os.path.join(camera_folder, "sample_*.png"),
            os.path.join(camera_folder, "image_*.jpg"),
            os.path.join(camera_folder, "image_*.png"),
            os.path.join(camera_folder, "*.jpg"),
            os.path.join(camera_folder, "*.png")
        ]
        
        image_paths = []
        for pattern in image_patterns:
            paths = glob.glob(pattern)
            if paths:
                image_paths = sorted(paths)
                break
        
        if not image_paths:
            print(f"Warning: No images found in {camera_folder}")
            return []
        
        print(f"Reading {len(image_paths)} images for camera {camera_config['id']} from {camera_config['folder']}")
        
        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                if RESIZE_FACTOR != 1.0:
                    new_size = (int(img.shape[1] * RESIZE_FACTOR), 
                               int(img.shape[0] * RESIZE_FACTOR))
                    img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
                images.append(img)
            else:
                print(f"Warning: Failed to read image: {img_path}")
        
        print(f"Successfully loaded {len(images)} images for camera {camera_config['id']}")
        return images
    
    def read_all_camera_images(self) -> List[List[np.ndarray]]:
        """Read images for all cameras"""
        all_images = []
        for camera_config in self.cameras:
            images = self.read_images(camera_config)
            all_images.append(images)
        return all_images
    
    def read_all_camera_infos(self) -> List[CameraInfo]:
        """Read intrinsics for all cameras"""
        camera_infos = []
        for camera_config in self.cameras:
            try:
                camera_info = self.read_camera_info(camera_config)
                camera_infos.append(camera_info)
                print(f"Camera {camera_info.camera_id}: fx={camera_info.fx:.1f}, fy={camera_info.fy:.1f}, "
                      f"resolution={camera_info.img_width}x{camera_info.img_height}")
            except Exception as e:
                print(f"Error reading camera {camera_config['id']} intrinsics: {e}")
                # Create a default camera info to maintain indexing
                camera_infos.append(self._create_default_camera_info(camera_config['id']))
        
        return camera_infos
    
    def _create_default_camera_info(self, camera_id: int) -> CameraInfo:
        """Create default camera info for missing intrinsics"""
        print(f"Creating default intrinsics for camera {camera_id}")
        return CameraInfo(
            camera_id=camera_id,
            fx=800.0, fy=800.0, cx=320.0, cy=240.0,
            dist_coeffs=np.zeros(5),
            img_width=640, img_height=480
        )
    
    def read_robot_poses(self) -> List[np.ndarray]:
        """Read robot poses from JSON file"""
        poses_path = os.path.join(self.data_folder, POSES_FILE)
        
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"Poses file not found: {poses_path}")
        
        with open(poses_path, 'r') as f:
            poses_data = json.load(f)
        
        poses = []
        for sample_id in sorted(poses_data.keys()):
            pose_data = poses_data[sample_id]
            position = np.array(pose_data['position'])
            quaternion = np.array(pose_data['orientation'])  # [x,y,z,w]
            
            # Convert to 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = Utils.quaternion_to_matrix(quaternion)
            T[:3, 3] = position
            
            poses.append(T.astype(np.float64))
        
        print(f"Read {len(poses)} robot poses")
        return poses

# =============================================================================
# PATTERN DETECTOR
# =============================================================================

class PatternDetector:
    def __init__(self, calib_info: CalibrationInfo, camera_infos: List[CameraInfo]):
        self.calib_info = calib_info
        self.camera_infos = camera_infos
        self.pattern_size = (calib_info.number_of_rows, calib_info.number_of_columns)
        
    def get_object_points(self) -> np.ndarray:
        """Generate 3D object points for the calibration pattern"""
        objp = np.zeros((self.calib_info.number_of_rows * self.calib_info.number_of_columns, 3), 
                       np.float32)
        objp[:, :2] = np.mgrid[0:self.calib_info.number_of_rows, 
                              0:self.calib_info.number_of_columns].T.reshape(-1, 2)
        objp *= self.calib_info.size
        return objp
    
    def detect_checkerboard(self, image: np.ndarray, camera_info: CameraInfo) -> Tuple[bool, np.ndarray]:
        """Detect checkerboard corners in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(
            gray, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        
        if found:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # RANSAC filter for robust detection
            object_points = self.get_object_points()
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, corners, camera_info.camera_matrix, 
                camera_info.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
                iterationsCount=50, reprojectionError=8.0, confidence=0.99
            )
            
            if success and len(inliers) == len(object_points):
                return True, corners.reshape(-1, 2)
        
        return False, None
    
    def detect_patterns(self, images_all: List[List[np.ndarray]], 
                       poses_all: List[np.ndarray]) -> DetectionResults:
        """Detect patterns in all images and filter valid detections"""
        print("Detecting calibration patterns...")
        
        num_cameras = len(images_all)
        max_poses = max(len(images) for images in images_all) if images_all else 0
        num_poses = min(len(poses_all), max_poses)
        
        print(f"Processing {num_cameras} cameras with up to {num_poses} poses")
        
        # Initialize data structures
        correct_images = [[] for _ in range(num_cameras)]
        correct_poses = [[] for _ in range(num_cameras)]
        correct_corners = [[[] for _ in range(num_cameras)] for _ in range(num_poses)]
        cross_observation_matrix = [[0 for _ in range(num_cameras)] for _ in range(num_poses)]
        
        object_points = self.get_object_points()
        
        for cam_id in range(num_cameras):
            print(f"Processing camera {cam_id + 1}...")
            images = images_all[cam_id]
            
            if not images:
                print(f"No images found for camera {cam_id + 1}")
                continue
            
            detection_count = 0
            for img_id, image in enumerate(images):
                if img_id >= num_poses:
                    break
                    
                success, corners = self.detect_checkerboard(image, self.camera_infos[cam_id])
                
                if success:
                    correct_images[cam_id].append(image.copy())
                    correct_poses[cam_id].append(poses_all[img_id].copy())
                    correct_corners[img_id][cam_id] = corners
                    cross_observation_matrix[img_id][cam_id] = 1
                    detection_count += 1
            
            print(f"Camera {cam_id + 1}: {detection_count}/{len(images)} valid detections")
        
        # Print cross-observation summary
        total_detections = sum(sum(row) for row in cross_observation_matrix)
        print(f"Total valid detections across all cameras: {total_detections}")
        
        return DetectionResults(
            correct_images=correct_images,
            correct_poses=correct_poses,
            correct_corners=correct_corners,
            cross_observation_matrix=cross_observation_matrix,
            object_points=object_points
        )

# =============================================================================
# HAND-EYE CALIBRATOR
# =============================================================================

class HandEyeCalibrator:
    def __init__(self, detection_results: DetectionResults, camera_infos: List[CameraInfo],
                 poses: List[np.ndarray], calib_info: CalibrationInfo):
        self.detection_results = detection_results
        self.camera_infos = camera_infos
        self.poses = poses
        self.calib_info = calib_info
        self.num_cameras = len(camera_infos)
        self.num_poses = len(poses)
        
    def set_initial_guess(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Set initial guess for hand-eye transformations"""
        print("Setting initial guess for optimization...")
        
        # Initialize hand-to-eye transforms for each camera
        h2e_initial = []
        for i in range(self.num_cameras):
            # Start with identity transform
            T_init = np.eye(4)
            # Add small random perturbation to avoid local minima
            T_init[:3, 3] = np.random.normal(0, 0.01, 3)  # Small translation
            h2e_initial.append(T_init)
            print(f"Camera {i+1} initial transform set")
        
        # Board to end-effector initial guess
        if self.calib_info.calibration_setup == 1:  # eye-on-base
            b2ee_initial = np.eye(4)
            # Position board slightly in front
            b2ee_initial[2, 3] = 0.1  # 10cm in Z
        else:  # eye-in-hand
            b2ee_initial = np.eye(4)
            # Position board slightly in front of end-effector
            b2ee_initial[2, 3] = 0.1  # 10cm in Z
            
        return h2e_initial, b2ee_initial
    
    def project_points(self, object_points: np.ndarray, rvec: np.ndarray, 
                      tvec: np.ndarray, camera_info: CameraInfo) -> np.ndarray:
        """Project 3D points to image plane"""
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, 
            camera_info.camera_matrix, camera_info.dist_coeffs
        )
        return projected.reshape(-1, 2)
    
    def pack_parameters(self, h2e_list: List[np.ndarray], b2ee: np.ndarray) -> np.ndarray:
        """Pack transformation matrices into parameter vector"""
        params = []
        
        # Pack board-to-end-effector (6 parameters)
        b2ee_rvec, b2ee_tvec = Utils.matrix_to_pose(b2ee)
        params.extend(b2ee_rvec.flatten())
        params.extend(b2ee_tvec.flatten())
        
        # Pack hand-to-eye for each camera (6 parameters each)
        for h2e in h2e_list:
            h2e_rvec, h2e_tvec = Utils.matrix_to_pose(h2e)
            params.extend(h2e_rvec.flatten())
            params.extend(h2e_tvec.flatten())
        
        return np.array(params)
    
    def unpack_parameters(self, params: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Unpack parameter vector into transformation matrices"""
        # Unpack board-to-end-effector
        b2ee_rvec = params[:3].reshape(-1, 1)
        b2ee_tvec = params[3:6].reshape(-1, 1)
        b2ee = Utils.pose_to_matrix(b2ee_rvec, b2ee_tvec)
        
        # Unpack hand-to-eye for each camera
        h2e_list = []
        for i in range(self.num_cameras):
            start_idx = 6 + i * 6
            h2e_rvec = params[start_idx:start_idx+3].reshape(-1, 1)
            h2e_tvec = params[start_idx+3:start_idx+6].reshape(-1, 1)
            h2e = Utils.pose_to_matrix(h2e_rvec, h2e_tvec)
            h2e_list.append(h2e)
        
        return h2e_list, b2ee
    
    def compute_residuals_single_camera(self, h2e: np.ndarray, b2ee: np.ndarray,
                                      camera_id: int, pose_id: int, 
                                      corners: np.ndarray) -> np.ndarray:
        """Compute reprojection residuals for single camera"""
        # Get robot pose
        robot_pose = self.poses[pose_id]
        
        # Compute transformation chain based on calibration setup
        if self.calib_info.calibration_setup == 0:  # eye-in-hand
            # Point in camera frame = H2E^-1 * Robot^-1 * B2EE * Point_board
            transform_chain = np.linalg.inv(h2e) @ np.linalg.inv(robot_pose) @ b2ee
        else:  # eye-on-base
            # Point in camera frame = H2E * Robot * B2EE * Point_board
            transform_chain = h2e @ robot_pose @ b2ee
        
        # Extract rotation and translation for projection
        final_rvec, final_tvec = Utils.matrix_to_pose(transform_chain)
        
        # Project object points
        projected = self.project_points(
            self.detection_results.object_points, 
            final_rvec, final_tvec, 
            self.camera_infos[camera_id]
        )
        
        # Compute residuals
        residuals = (projected - corners).flatten()
        return residuals
    
    def objective_function(self, params: np.ndarray) -> np.ndarray:
        """Objective function for optimization"""
        # Unpack parameters
        h2e_list, b2ee = self.unpack_parameters(params)
        
        all_residuals = []
        
        for pose_id in range(self.num_poses):
            for camera_id in range(self.num_cameras):
                if self.detection_results.cross_observation_matrix[pose_id][camera_id]:
                    corners = self.detection_results.correct_corners[pose_id][camera_id]
                    
                    # Single camera residuals
                    residuals = self.compute_residuals_single_camera(
                        h2e_list[camera_id], b2ee, camera_id, pose_id, corners
                    )
                    all_residuals.extend(residuals)
        
        return np.array(all_residuals)
    
    def calibrate(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Perform hand-eye calibration optimization"""
        print("Starting hand-eye calibration optimization...")
        
        # Set initial guess
        h2e_initial, b2ee_initial = self.set_initial_guess()
        
        # Pack parameters for optimization
        initial_params = self.pack_parameters(h2e_initial, b2ee_initial)
        
        print(f"Optimizing {len(initial_params)} parameters:")
        print(f"  - Board-to-end-effector: 6 parameters")
        print(f"  - Hand-to-eye transforms: {self.num_cameras} cameras × 6 parameters = {self.num_cameras * 6}")
        
        # Count total observations for optimization info
        total_observations = sum(
            sum(self.detection_results.cross_observation_matrix[pose_id][cam_id] 
                for cam_id in range(self.num_cameras))
            for pose_id in range(self.num_poses)
        )
        total_residuals = total_observations * len(self.detection_results.object_points) * 2
        print(f"Total observations: {total_observations}")
        print(f"Total residuals: {total_residuals}")
        
        # Optimization
        result = least_squares(
            self.objective_function,
            initial_params,
            method='lm',
            max_nfev=MAX_ITERATIONS,
            ftol=CONVERGENCE_TOL,
            verbose=2
        )
        
        print(f"Optimization finished:")
        print(f"  Success: {result.success}")
        print(f"  Final cost: {result.cost:.6f}")
        print(f"  Iterations: {result.nfev}")
        print(f"  Final residual norm: {np.linalg.norm(result.fun):.6f}")
        
        # Extract optimized parameters
        h2e_optimal, b2ee_optimal = self.unpack_parameters(result.x)
        
        return h2e_optimal, b2ee_optimal

# =============================================================================
# METRICS AND EVALUATION
# =============================================================================

class Metrics:
    def __init__(self, detection_results: DetectionResults, camera_infos: List[CameraInfo],
                 h2e_optimal: List[np.ndarray], b2ee_optimal: np.ndarray, 
                 calib_info: CalibrationInfo):
        self.detection_results = detection_results
        self.camera_infos = camera_infos
        self.h2e_optimal = h2e_optimal
        self.b2ee_optimal = b2ee_optimal
        self.calib_info = calib_info
    
    def compute_reprojection_error(self, poses: List[np.ndarray]) -> Dict:
        """Compute reprojection errors for all cameras"""
        print("Computing reprojection errors...")
        
        errors_per_camera = []
        summary_stats = {}
        
        for camera_id in range(len(self.camera_infos)):
            camera_errors = []
            
            for pose_id in range(len(poses)):
                if self.detection_results.cross_observation_matrix[pose_id][camera_id]:
                    corners_detected = self.detection_results.correct_corners[pose_id][camera_id]
                    
                    # Compute transformation chain
                    robot_pose = poses[pose_id]
                    
                    if self.calib_info.calibration_setup == 0:  # eye-in-hand
                        transform_chain = (np.linalg.inv(self.h2e_optimal[camera_id]) @ 
                                         np.linalg.inv(robot_pose) @ self.b2ee_optimal)
                    else:  # eye-on-base
                        transform_chain = (self.h2e_optimal[camera_id] @ 
                                         robot_pose @ self.b2ee_optimal)
                    
                    # Project points
                    rvec, tvec = Utils.matrix_to_pose(transform_chain)
                    projected = self.project_points(
                        self.detection_results.object_points,
                        rvec, tvec, self.camera_infos[camera_id]
                    )
                    
                    # Compute per-point errors
                    point_errors = np.linalg.norm(projected - corners_detected, axis=1)
                    mean_error = np.mean(point_errors)
                    camera_errors.append(mean_error)
            
            errors_per_camera.append(camera_errors)
            
            if camera_errors:
                avg_error = np.mean(camera_errors)
                std_error = np.std(camera_errors)
                max_error = np.max(camera_errors)
                min_error = np.min(camera_errors)
                
                summary_stats[f'camera_{camera_id + 1}'] = {
                    'mean_error': avg_error,
                    'std_error': std_error,
                    'max_error': max_error,
                    'min_error': min_error,
                    'num_observations': len(camera_errors)
                }
                
                print(f"Camera {camera_id + 1}: {len(camera_errors)} observations")
                print(f"  Mean error: {avg_error:.4f} ± {std_error:.4f} pixels")
                print(f"  Range: [{min_error:.4f}, {max_error:.4f}] pixels")
            else:
                print(f"Camera {camera_id + 1}: No valid observations")
                summary_stats[f'camera_{camera_id + 1}'] = {
                    'mean_error': 0, 'std_error': 0, 'max_error': 0, 'min_error': 0,
                    'num_observations': 0
                }
        
        return {
            'per_camera_errors': errors_per_camera,
            'summary_statistics': summary_stats
        }
    
    def project_points(self, object_points: np.ndarray, rvec: np.ndarray, 
                      tvec: np.ndarray, camera_info: CameraInfo) -> np.ndarray:
        """Project 3D points to image plane"""
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec,
            camera_info.camera_matrix, camera_info.dist_coeffs
        )
        return projected.reshape(-1, 2)

# =============================================================================
# MAIN CALIBRATION CLASS
# =============================================================================

class MultiCameraHandEyeCalibration:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.reader = DataReader(data_folder)
        
    def run_calibration(self):
        """Run the complete calibration pipeline"""
        print("=== Multi-Camera Hand-Eye Calibration ===")
        
        # Create results directory
        results_dir = os.path.join(self.data_folder, RESULTS_FOLDER)
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Read configuration
        print("\n1. Reading calibration configuration...")
        calib_info = self.reader.read_calibration_info()
        print(f"Calibration setup: {'Eye-in-hand' if calib_info.calibration_setup == 0 else 'Eye-on-base'}")
        print(f"Number of cameras: {calib_info.number_of_cameras}")
        print(f"Pattern: {calib_info.pattern_type} ({calib_info.number_of_rows}x{calib_info.number_of_columns})")
        print(f"Square size: {calib_info.size}m")
        
        if calib_info.number_of_cameras == 0:
            print("Error: No cameras detected!")
            return
        
        # 2. Read camera information
        print("\n2. Reading camera intrinsics...")
        camera_infos = self.reader.read_all_camera_infos()
        
        if len(camera_infos) != calib_info.number_of_cameras:
            print(f"Warning: Expected {calib_info.number_of_cameras} cameras, got {len(camera_infos)}")
        
        for i, camera_info in enumerate(camera_infos):
            print(f"Camera {camera_info.camera_id}: fx={camera_info.fx:.1f}, fy={camera_info.fy:.1f}, "
                  f"resolution={camera_info.img_width}x{camera_info.img_height}")
        
        # 3. Read images and poses
        print("\n3. Reading images and poses...")
        images_all = self.reader.read_all_camera_images()
        poses = self.reader.read_robot_poses()
        
        # Verify data consistency
        print(f"Images per camera: {[len(imgs) for imgs in images_all]}")
        print(f"Total robot poses: {len(poses)}")
        
        # Check if we have enough data
        min_images = min(len(imgs) for imgs in images_all if imgs)
        if min_images < 5:
            print("Warning: Very few images available. Consider collecting more data.")
        
        # 4. Detect calibration patterns
        print("\n4. Detecting calibration patterns...")
        detector = PatternDetector(calib_info, camera_infos)
        detection_results = detector.detect_patterns(images_all, poses)
        
        # Check detection results
        total_detections = sum(sum(row) for row in detection_results.cross_observation_matrix)
        if total_detections < 10:
            print("Warning: Very few pattern detections. Check pattern visibility and camera calibration.")
        
        # 5. Perform hand-eye calibration
        print("\n5. Performing hand-eye calibration...")
        calibrator = HandEyeCalibrator(detection_results, camera_infos, poses, calib_info)
        h2e_optimal, b2ee_optimal = calibrator.calibrate()
        
        # 6. Evaluate results
        print("\n6. Evaluating calibration results...")
        metrics = Metrics(detection_results, camera_infos, h2e_optimal, b2ee_optimal, calib_info)
        errors = metrics.compute_reprojection_error(poses)
        
        # 7. Save results
        print("\n7. Saving results...")
        self.save_results(results_dir, h2e_optimal, b2ee_optimal, errors, calib_info, camera_infos)
        
        print("\n=== Calibration Complete ===")
        self.print_results(h2e_optimal, b2ee_optimal, calib_info, camera_infos, errors)
    
    def save_results(self, results_dir: str, h2e_optimal: List[np.ndarray], 
                    b2ee_optimal: np.ndarray, errors: Dict, calib_info: CalibrationInfo,
                    camera_infos: List[CameraInfo]):
        """Save calibration results"""
        results = {
            'calibration_info': {
                'setup': 'eye-in-hand' if calib_info.calibration_setup == 0 else 'eye-on-base',
                'number_of_cameras': calib_info.number_of_cameras,
                'pattern_type': calib_info.pattern_type,
                'pattern_size': [calib_info.number_of_rows, calib_info.number_of_columns],
                'square_size_m': calib_info.size
            },
            'camera_info': [
                {
                    'camera_id': cam.camera_id,
                    'fx': cam.fx, 'fy': cam.fy, 'cx': cam.cx, 'cy': cam.cy,
                    'distortion_coeffs': cam.dist_coeffs.tolist(),
                    'image_size': [cam.img_width, cam.img_height]
                }
                for cam in camera_infos
            ],
            'hand_to_eye_transforms': [
                {
                    'camera_id': i + 1,
                    'transform_matrix': h2e.tolist(),
                    'rotation_matrix': h2e[:3, :3].tolist(),
                    'translation_vector': h2e[:3, 3].tolist()
                }
                for i, h2e in enumerate(h2e_optimal)
            ],
            'board_to_end_effector_transform': {
                'transform_matrix': b2ee_optimal.tolist(),
                'rotation_matrix': b2ee_optimal[:3, :3].tolist(),
                'translation_vector': b2ee_optimal[:3, 3].tolist()
            },
            'reprojection_errors': errors
        }
        
        results_file = os.path.join(results_dir, 'hand_eye_calibration_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        # Save individual transform files for each camera
        for i, h2e in enumerate(h2e_optimal):
            camera_file = os.path.join(results_dir, f'camera_{i+1}_transform.json')
            camera_result = {
                'camera_id': i + 1,
                'hand_to_eye_transform': h2e.tolist(),
                'setup': 'eye-in-hand' if calib_info.calibration_setup == 0 else 'eye-on-base'
            }
            with open(camera_file, 'w') as f:
                json.dump(camera_result, f, indent=2)
        
        # Save board transform
        board_file = os.path.join(results_dir, 'board_transform.json')
        board_result = {
            'board_to_end_effector_transform': b2ee_optimal.tolist(),
            'setup': 'eye-in-hand' if calib_info.calibration_setup == 0 else 'eye-on-base'
        }
        with open(board_file, 'w') as f:
            json.dump(board_result, f, indent=2)
        
        print(f"Individual transform files saved to: {results_dir}")
    
    def print_results(self, h2e_optimal: List[np.ndarray], b2ee_optimal: np.ndarray, 
                     calib_info: CalibrationInfo, camera_infos: List[CameraInfo], 
                     errors: Dict):
        """Print calibration results"""
        print("\nCalibration Results:")
        print("=" * 80)
        
        setup_name = "Eye-in-hand" if calib_info.calibration_setup == 0 else "Eye-on-base"
        print(f"Setup: {setup_name}")
        print(f"Number of cameras: {len(h2e_optimal)}")
        
        for i, (h2e, camera_info) in enumerate(zip(h2e_optimal, camera_infos)):
            print(f"\nCamera {camera_info.camera_id} ({camera_info.img_width}x{camera_info.img_height}):")
            print("-" * 40)
            
            if calib_info.calibration_setup == 0:  # eye-in-hand
                print("Hand-to-Eye Transform:")
                print(h2e)
                
                # Also show Eye-to-Hand (inverse) for clarity
                print("\nEye-to-Hand Transform (inverse):")
                print(np.linalg.inv(h2e))
            else:  # eye-on-base
                print("Base-to-Eye Transform:")
                print(h2e)
                
                # Show Eye-to-Base (inverse) for clarity
                print("\nEye-to-Base Transform (inverse):")
                print(np.linalg.inv(h2e))
            
            # Print translation and rotation separately for easier interpretation
            translation = h2e[:3, 3]
            rotation_matrix = h2e[:3, :3]
            euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
            
            print(f"\nTranslation (m): [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
            print(f"Rotation (deg): [{euler_angles[0]:.2f}, {euler_angles[1]:.2f}, {euler_angles[2]:.2f}]")
            
            # Print error statistics if available
            if 'summary_statistics' in errors:
                cam_key = f'camera_{camera_info.camera_id}'
                if cam_key in errors['summary_statistics']:
                    stats = errors['summary_statistics'][cam_key]
                    print(f"Reprojection error: {stats['mean_error']:.4f} ± {stats['std_error']:.4f} pixels "
                          f"({stats['num_observations']} observations)")
        
        print(f"\n{'Board-to-End-Effector' if calib_info.calibration_setup == 0 else 'Board-to-Hand'} Transform:")
        print("-" * 50)
        print(b2ee_optimal)
        
        # Print board transform details
        board_translation = b2ee_optimal[:3, 3]
        board_rotation = b2ee_optimal[:3, :3]
        board_euler = R.from_matrix(board_rotation).as_euler('xyz', degrees=True)
        
        print(f"\nBoard Translation (m): [{board_translation[0]:.4f}, {board_translation[1]:.4f}, {board_translation[2]:.4f}]")
        print(f"Board Rotation (deg): [{board_euler[0]:.2f}, {board_euler[1]:.2f}, {board_euler[2]:.2f}]")
        
        # Overall statistics
        if 'summary_statistics' in errors:
            print(f"\nOverall Statistics:")
            print("-" * 30)
            all_errors = []
            total_observations = 0
            
            for cam_stats in errors['summary_statistics'].values():
                if cam_stats['num_observations'] > 0:
                    all_errors.extend([cam_stats['mean_error']] * cam_stats['num_observations'])
                    total_observations += cam_stats['num_observations']
            
            if all_errors:
                overall_mean = np.mean(all_errors)
                overall_std = np.std(all_errors)
                print(f"Overall reprojection error: {overall_mean:.4f} ± {overall_std:.4f} pixels")
                print(f"Total observations: {total_observations}")
                
                if overall_mean < 1.0:
                    print("✓ Excellent calibration quality (< 1.0 pixel)")
                elif overall_mean < 2.0:
                    print("✓ Good calibration quality (< 2.0 pixels)")
                elif overall_mean < 3.0:
                    print("⚠ Acceptable calibration quality (< 3.0 pixels)")
                else:
                    print("⚠ Poor calibration quality (≥ 3.0 pixels) - consider recalibrating")

# =============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL USE
# =============================================================================

def load_calibration_results(results_file: str) -> Dict:
    """Load calibration results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def get_camera_transform(results: Dict, camera_id: int) -> np.ndarray:
    """Extract hand-to-eye transform for specific camera"""
    for cam_data in results['hand_to_eye_transforms']:
        if cam_data['camera_id'] == camera_id:
            return np.array(cam_data['transform_matrix'])
    raise ValueError(f"Camera {camera_id} not found in results")

def get_board_transform(results: Dict) -> np.ndarray:
    """Extract board-to-end-effector transform"""
    return np.array(results['board_to_end_effector_transform']['transform_matrix'])

def transform_point_to_camera(point_board: np.ndarray, robot_pose: np.ndarray,
                            h2e_transform: np.ndarray, b2ee_transform: np.ndarray,
                            setup: int = 0) -> np.ndarray:
    """Transform a point from board frame to camera frame
    
    Args:
        point_board: 3D point in board coordinate system
        robot_pose: 4x4 robot end-effector pose matrix
        h2e_transform: 4x4 hand-to-eye transform matrix
        b2ee_transform: 4x4 board-to-end-effector transform matrix
        setup: 0 for eye-in-hand, 1 for eye-on-base
    
    Returns:
        3D point in camera coordinate system
    """
    # Add homogeneous coordinate
    if point_board.shape[0] == 3:
        point_board_h = np.append(point_board, 1)
    else:
        point_board_h = point_board
    
    if setup == 0:  # eye-in-hand
        # Point in camera = H2E^-1 * Robot^-1 * B2EE * Point_board
        transform_chain = np.linalg.inv(h2e_transform) @ np.linalg.inv(robot_pose) @ b2ee_transform
    else:  # eye-on-base
        # Point in camera = H2E * Robot * B2EE * Point_board
        transform_chain = h2e_transform @ robot_pose @ b2ee_transform
    
    point_camera_h = transform_chain @ point_board_h
    return point_camera_h[:3]

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function"""
    print("Multi-Camera Hand-Eye Calibration System")
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Auto-detect cameras: {AUTO_DETECT_CAMERAS}")
    print(f"Use shared intrinsics: {USE_SHARED_INTRINSICS}")
    
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder '{DATA_FOLDER}' not found!")
        print("Please ensure the data folder exists and contains:")
        print("  - Camera folders (external_camera_1, external_camera_2, etc.)")
        print("  - Robot poses file (ee_poses.json)")
        print("  - Camera intrinsics file(s)")
        return
    
    # Verify required files
    poses_file = os.path.join(DATA_FOLDER, POSES_FILE)
    if not os.path.exists(poses_file):
        print(f"Error: Robot poses file '{poses_file}' not found!")
        return
    
    if USE_SHARED_INTRINSICS:
        intrinsics_file = os.path.join(DATA_FOLDER, SHARED_INTRINSICS_FILE)
        if not os.path.exists(intrinsics_file):
            print(f"Error: Shared intrinsics file '{intrinsics_file}' not found!")
            return
    
    print("\nStarting calibration...")
    calibration = MultiCameraHandEyeCalibration(DATA_FOLDER)
    calibration.run_calibration()

if __name__ == "__main__":
    main()
