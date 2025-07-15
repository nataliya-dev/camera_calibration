import numpy as np
import cv2
import json
import os
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from dataclasses import dataclass
import glob

@dataclass
class CameraPose:
    """Represents camera pose detection result"""
    success: bool
    rvec: np.ndarray  # Rotation vector
    tvec: np.ndarray  # Translation vector
    corners: np.ndarray
    ids: np.ndarray

@dataclass
class CalibrationSample:
    """Single calibration sample containing all necessary data"""
    ee_position: np.ndarray
    ee_orientation: np.ndarray  # quaternion [x, y, z, w]
    hand_camera_pose: CameraPose
    external_camera_pose: CameraPose
    sample_id: str

class CharucoDetector:
    """Handles ChArUco board detection in images"""
    
    def __init__(self, 
                 squares_x: int = 7, 
                 squares_y: int = 5, 
                 square_length: float = 0.03,  # 4cm
                 marker_length: float = 0.015,  # 2cm
                 dictionary: int = cv2.aruco.DICT_6X6_250):
        
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        
        # Create ChArUco board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), 
            square_length, 
            marker_length, 
            self.aruco_dict
        )
        
        # Detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
    def detect_pose(self, image: np.ndarray, camera_matrix: np.ndarray, 
                   dist_coeffs: np.ndarray) -> CameraPose:
        """Detect ChArUco board pose in image"""
        
        # Detect ArUco markers
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        corners, ids, _ = detector.detectMarkers(image)
        
        if len(corners) == 0:
            return CameraPose(False, None, None, None, None)
        
        # Interpolate ChArUco corners
        charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, image, self.board
        )
        
        if charuco_corners is None or len(charuco_corners) < 4:
            return CameraPose(False, None, None, None, None)
        
        # Estimate pose
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board,
            camera_matrix, dist_coeffs, None, None
        )
        
        if not success:
            return CameraPose(False, None, None, None, None)
        
        return CameraPose(True, rvec.flatten(), tvec.flatten(), 
                         charuco_corners, charuco_ids)

class DataLoader:
    """Handles loading of calibration data from files"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ee_poses_file = os.path.join(data_dir, "ee_poses.json")
        self.hand_images_dir = os.path.join(data_dir, "hand_camera")
        self.external_images_dir = os.path.join(data_dir, "external_camera")
        
    def load_ee_poses(self) -> Dict:
        """Load end-effector poses from JSON file"""
        with open(self.ee_poses_file, 'r') as f:
            return json.load(f)
    
    def load_calibration_samples(self, 
                               hand_camera_matrix: np.ndarray,
                               hand_dist_coeffs: np.ndarray,
                               external_camera_matrix: np.ndarray,
                               external_dist_coeffs: np.ndarray,
                               detector: CharucoDetector) -> List[CalibrationSample]:
        """Load all calibration samples"""
        
        ee_poses = self.load_ee_poses()
        samples = []
        
        for sample_id, pose_data in ee_poses.items():
            # Load images
            hand_img_path = os.path.join(self.hand_images_dir, f"{sample_id}.jpg")
            external_img_path = os.path.join(self.external_images_dir, f"{sample_id}.jpg")
            
            if not (os.path.exists(hand_img_path) and os.path.exists(external_img_path)):
                print(f"Warning: Missing images for sample {sample_id}")
                continue
            
            hand_img = cv2.imread(hand_img_path)
            external_img = cv2.imread(external_img_path)
            
            # Detect ChArUco poses
            hand_pose = detector.detect_pose(hand_img, hand_camera_matrix, hand_dist_coeffs)
            external_pose = detector.detect_pose(external_img, external_camera_matrix, external_dist_coeffs)
            
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
                print(f"Warning: ChArUco detection failed for sample {sample_id}")
        
        print(f"Loaded {len(samples)} valid calibration samples")
        return samples

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
        T[:3, :3] = R.from_quat(quaternion).as_matrix()  # quaternion is [x, y, z, w]
        T[:3, 3] = position
        return T
    
    def objective_function(self, params: np.ndarray) -> float:
        """Objective function for optimization"""
        
        T_hand_eye, T_base_external = self.unpack_parameters(params)
        
        total_error = 0.0
        
        for sample in self.samples:
            # Get transformations
            T_base_ee = self.ee_pose_to_matrix(sample.ee_position, sample.ee_orientation)
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
            error_matrix = T_board_base_path1 @ np.linalg.inv(T_board_base_path2)
            
            # Extract rotation and translation errors
            R_error = error_matrix[:3, :3]
            t_error = error_matrix[:3, 3]
            
            # Rotation error (angle of rotation matrix)
            rotation_error = np.linalg.norm(R.from_matrix(R_error).as_rotvec())
            
            # Translation error
            translation_error = np.linalg.norm(t_error)
            
            # Combined error (weighted)
            total_error += rotation_error**2 + 0.1 * translation_error**2
        
        return total_error
    
    def calibrate(self, initial_guess: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
        """Perform calibration optimization"""
        
        if initial_guess is None:
            # Initialize with identity transformations
            T_hand_eye_init = np.eye(4)
            T_base_external_init = np.eye(4)
            # Add small perturbation to avoid singular initial conditions
            T_hand_eye_init[:3, 3] = [0.1, 0.0, 0.0]
            T_base_external_init[:3, 3] = [0.5, 0.5, 0.5]
        else:
            T_hand_eye_init, T_base_external_init = initial_guess
        
        # Pack initial parameters
        initial_params = self.pack_parameters(T_hand_eye_init, T_base_external_init)
        
        print("Starting optimization...")
        print(f"Initial error: {self.objective_function(initial_params):.6f}")
        
        # Optimize
        result = minimize(
            self.objective_function,
            initial_params,
            method='BFGS',
            options={'disp': True, 'maxiter': 1000}
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
            T_base_ee = self.ee_pose_to_matrix(sample.ee_position, sample.ee_orientation)
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
        
        errors_array = np.array([[e['rotation_error_deg'], e['translation_error_m']] for e in errors])
        
        return {
            'individual_errors': errors,
            'mean_rotation_error_deg': np.mean(errors_array[:, 0]),
            'mean_translation_error_m': np.mean(errors_array[:, 1]),
            'std_rotation_error_deg': np.std(errors_array[:, 0]),
            'std_translation_error_m': np.std(errors_array[:, 1]),
            'max_rotation_error_deg': np.max(errors_array[:, 0]),
            'max_translation_error_m': np.max(errors_array[:, 1])
        }

def main():
    """Main calibration pipeline"""
    
    # Configuration
    data_dir = "calibration_data"
    
    # Camera intrinsics (you need to provide these from camera calibration)
    # Hand camera intrinsics
    hand_camera_matrix = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    hand_dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])
    
    # External camera intrinsics  
    external_camera_matrix = np.array([
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 480.0],
        [0.0, 0.0, 1.0]
    ])
    external_dist_coeffs = np.array([0.05, -0.1, 0.0, 0.0, 0.0])
    
    # Initialize components
    detector = CharucoDetector(
        squares_x=7, squares_y=5,
        square_length=0.04,  # 4cm squares
        marker_length=0.02   # 2cm markers
    )
    
    data_loader = DataLoader(data_dir)
    
    # Load calibration data
    samples = data_loader.load_calibration_samples(
        hand_camera_matrix, hand_dist_coeffs,
        external_camera_matrix, external_dist_coeffs,
        detector
    )
    
    if len(samples) < 5:
        print("Error: Need at least 5 valid samples for calibration")
        return
    
    # Perform calibration
    calibrator = HandEyeCalibrator(samples)
    result = calibrator.calibrate()
    
    if result['success']:
        print("\nCalibration successful!")
        print(f"Final error: {result['final_error']:.6f}")
        
        # Evaluate calibration quality
        evaluation = calibrator.evaluate_calibration(
            result['T_hand_eye'], 
            result['T_base_external']
        )
        
        print(f"\nCalibration Quality:")
        print(f"Mean rotation error: {evaluation['mean_rotation_error_deg']:.3f}° ± {evaluation['std_rotation_error_deg']:.3f}°")
        print(f"Mean translation error: {evaluation['mean_translation_error_m']:.4f}m ± {evaluation['std_translation_error_m']:.4f}m")
        print(f"Max rotation error: {evaluation['max_rotation_error_deg']:.3f}°")
        print(f"Max translation error: {evaluation['max_translation_error_m']:.4f}m")
        
        # Save results
        np.save("T_hand_eye.npy", result['T_hand_eye'])
        np.save("T_base_external.npy", result['T_base_external'])
        
        print(f"\nHand-Eye Transformation Matrix:")
        print(result['T_hand_eye'])
        print(f"\nBase-External Camera Transformation Matrix:")
        print(result['T_base_external'])
        
    else:
        print("Calibration failed!")
        print(result['optimization_result'])

if __name__ == "__main__":
    main()