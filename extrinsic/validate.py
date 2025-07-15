import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from typing import List, Dict
import json

class CalibrationValidator:
    """Utilities for validating and testing calibration results"""
    
    def __init__(self, T_hand_eye: np.ndarray, T_base_external: np.ndarray):
        self.T_hand_eye = T_hand_eye
        self.T_base_external = T_base_external
    
    def visualize_setup(self, samples: List, figsize=(12, 8)):
        """Visualize the calibration setup and results"""
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot robot base
        ax.scatter(0, 0, 0, c='red', s=100, marker='o', label='Robot Base')
        
        # Plot end-effector positions
        ee_positions = np.array([s.ee_position for s in samples])
        ax.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                  c='blue', s=50, marker='o', alpha=0.7, label='End-Effector Positions')
        
        # Plot external camera position
        external_pos = self.T_base_external[:3, 3]
        ax.scatter(external_pos[0], external_pos[1], external_pos[2], 
                  c='green', s=100, marker='^', label='External Camera')
        
        # Plot hand camera positions (transformed through kinematic chain)
        hand_camera_positions = []
        for sample in samples:
            T_base_ee = self.ee_pose_to_matrix(sample.ee_position, sample.ee_orientation)
            T_base_hand = T_base_ee @ self.T_hand_eye
            hand_camera_positions.append(T_base_hand[:3, 3])
        
        hand_camera_positions = np.array(hand_camera_positions)
        ax.scatter(hand_camera_positions[:, 0], hand_camera_positions[:, 1], 
                  hand_camera_positions[:, 2], c='orange', s=50, marker='s', 
                  alpha=0.7, label='Hand Camera Positions')
        
        # Plot board positions as detected by both cameras
        board_positions_hand = []
        board_positions_external = []
        
        for sample in samples:
            # Board position from hand camera
            T_base_ee = self.ee_pose_to_matrix(sample.ee_position, sample.ee_orientation)
            T_hand_board = self.pose_to_matrix(sample.hand_camera_pose.rvec, 
                                             sample.hand_camera_pose.tvec)
            T_base_board_hand = T_base_ee @ self.T_hand_eye @ T_hand_board
            board_positions_hand.append(T_base_board_hand[:3, 3])
            
            # Board position from external camera
            T_external_board = self.pose_to_matrix(sample.external_camera_pose.rvec, 
                                                 sample.external_camera_pose.tvec)
            T_base_board_external = self.T_base_external @ T_external_board
            board_positions_external.append(T_base_board_external[:3, 3])
        
        board_positions_hand = np.array(board_positions_hand)
        board_positions_external = np.array(board_positions_external)
        
        ax.scatter(board_positions_hand[:, 0], board_positions_hand[:, 1], 
                  board_positions_hand[:, 2], c='purple', s=30, marker='o', 
                  alpha=0.7, label='Board (Hand Camera)')
        
        ax.scatter(board_positions_external[:, 0], board_positions_external[:, 1], 
                  board_positions_external[:, 2], c='pink', s=30, marker='o', 
                  alpha=0.7, label='Board (External Camera)')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title('Calibration Setup Visualization')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_error_analysis(self, evaluation_results: Dict):
        """Plot error analysis"""
        
        errors = evaluation_results['individual_errors']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Rotation errors
        rotation_errors = [e['rotation_error_deg'] for e in errors]
        ax1.bar(range(len(rotation_errors)), rotation_errors, alpha=0.7, color='skyblue')
        ax1.axhline(y=evaluation_results['mean_rotation_error_deg'], 
                   color='red', linestyle='--', label=f"Mean: {evaluation_results['mean_rotation_error_deg']:.2f}°")
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Rotation Error (degrees)')
        ax1.set_title('Rotation Error per Sample')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Translation errors
        translation_errors = [e['translation_error_m'] for e in errors]
        ax2.bar(range(len(translation_errors)), translation_errors, alpha=0.7, color='lightcoral')
        ax2.axhline(y=evaluation_results['mean_translation_error_m'], 
                   color='red', linestyle='--', label=f"Mean: {evaluation_results['mean_translation_error_m']:.4f}m")
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Translation Error (m)')
        ax2.set_title('Translation Error per Sample')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def ee_pose_to_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Convert end-effector pose to transformation matrix"""
        from scipy.spatial.transform import Rotation as R
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quaternion).as_matrix()
        T[:3, 3] = position
        return T
    
    def pose_to_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert rotation vector and translation to transformation matrix"""
        from scipy.spatial.transform import Rotation as R
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(rvec).as_matrix()
        T[:3, 3] = tvec
        return T
    
    def test_new_position(self, ee_position: np.ndarray, ee_orientation: np.ndarray,
                         hand_camera_image: np.ndarray, external_camera_image: np.ndarray,
                         hand_camera_matrix: np.ndarray, hand_dist_coeffs: np.ndarray,
                         external_camera_matrix: np.ndarray, external_dist_coeffs: np.ndarray,
                         detector) -> Dict:
        """Test calibration with a new robot position"""
        
        # Detect board poses
        hand_pose = detector.detect_pose(hand_camera_image, hand_camera_matrix, hand_dist_coeffs)
        external_pose = detector.detect_pose(external_camera_image, external_camera_matrix, external_dist_coeffs)
        
        if not (hand_pose.success and external_pose.success):
            return {"success": False, "reason": "ChArUco detection failed"}
        
        # Compute board positions using calibrated transformations
        T_base_ee = self.ee_pose_to_matrix(ee_position, ee_orientation)
        T_hand_board = self.pose_to_matrix(hand_pose.rvec, hand_pose.tvec)
        T_external_board = self.pose_to_matrix(external_pose.rvec, external_pose.tvec)
        
        # Board position from hand camera
        T_base_board_hand = T_base_ee @ self.T_hand_eye @ T_hand_board
        
        # Board position from external camera
        T_base_board_external = self.T_base_external @ T_external_board
        
        # Compute error
        error_matrix = T_base_board_hand @ np.linalg.inv(T_base_board_external)
        
        from scipy.spatial.transform import Rotation as R
        R_error = error_matrix[:3, :3]
        t_error = error_matrix[:3, 3]
        
        rotation_error = np.linalg.norm(R.from_matrix(R_error).as_rotvec())
        translation_error = np.linalg.norm(t_error)
        
        return {
            "success": True,
            "rotation_error_deg": np.degrees(rotation_error),
            "translation_error_m": translation_error,
            "board_position_hand": T_base_board_hand[:3, 3],
            "board_position_external": T_base_board_external[:3, 3],
            "position_difference": np.linalg.norm(T_base_board_hand[:3, 3] - T_base_board_external[:3, 3])
        }

class DataQualityChecker:
    """Check quality of calibration data before running optimization"""
    
    def __init__(self, samples: List):
        self.samples = samples
    
    def check_data_quality(self) -> Dict:
        """Comprehensive data quality check"""
        
        results = {
            "num_samples": len(self.samples),
            "workspace_coverage": self.check_workspace_coverage(),
            "orientation_diversity": self.check_orientation_diversity(),
            "detection_quality": self.check_detection_quality(),
            "motion_analysis": self.check_motion_analysis(),
            "recommendations": []
        }
        
        # Generate recommendations
        if results["num_samples"] < 10:
            results["recommendations"].append("Consider collecting more samples (recommended: 15-30)")
        
        if results["workspace_coverage"]["volume"] < 0.01:  # Less than 0.01 m³
            results["recommendations"].append("Increase workspace coverage - move robot to more diverse positions")
        
        if results["orientation_diversity"]["std"] < 0.5:  # Low orientation diversity
            results["recommendations"].append("Increase orientation diversity - rotate end-effector more")
        
        if results["detection_quality"]["avg_corners"] < 20:
            results["recommendations"].append("Improve ChArUco detection - ensure better lighting/board visibility")
        
        return results
    
    def check_workspace_coverage(self) -> Dict:
        """Check how well the workspace is covered"""
        
        positions = np.array([s.ee_position for s in self.samples])
        
        # Compute workspace volume (bounding box)
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        volume = np.prod(max_pos - min_pos)
        
        # Compute average pairwise distance
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                distances.append(np.linalg.norm(positions[i] - positions[j]))
        
        return {
            "volume": volume,
            "min_position": min_pos.tolist(),
            "max_position": max_pos.tolist(),
            "avg_pairwise_distance": np.mean(distances) if distances else 0,
            "std_pairwise_distance": np.std(distances) if distances else 0
        }
    
    def check_orientation_diversity(self) -> Dict:
        """Check orientation diversity"""
        
        from scipy.spatial.transform import Rotation as R
        
        orientations = [s.ee_orientation for s in self.samples]
        
        # Convert to rotation matrices and compute angular distances
        rotations = [R.from_quat(q) for q in orientations]
        
        angular_distances = []
        for i in range(len(rotations)):
            for j in range(i+1, len(rotations)):
                # Angular distance between rotations
                rel_rot = rotations[i].inv() * rotations[j]
                angle = np.linalg.norm(rel_rot.as_rotvec())
                angular_distances.append(angle)
        
        return {
            "mean_angular_distance": np.mean(angular_distances) if angular_distances else 0,
            "std": np.std(angular_distances) if angular_distances else 0,
            "max_angular_distance": np.max(angular_distances) if angular_distances else 0
        }
    
    def check_detection_quality(self) -> Dict:
        """Check ChArUco detection quality"""
        
        hand_corner_counts = []
        external_corner_counts = []
        
        for sample in self.samples:
            if sample.hand_camera_pose.success:
                hand_corner_counts.append(len(sample.hand_camera_pose.corners))
            if sample.external_camera_pose.success:
                external_corner_counts.append(len(sample.external_camera_pose.corners))
        
        return {
            "hand_camera_success_rate": len(hand_corner_counts) / len(self.samples),
            "external_camera_success_rate": len(external_corner_counts) / len(self.samples),
            "avg_corners": np.mean(hand_corner_counts + external_corner_counts) if hand_corner_counts + external_corner_counts else 0,
            "min_corners": np.min(hand_corner_counts + external_corner_counts) if hand_corner_counts + external_corner_counts else 0
        }
    
    def check_motion_analysis(self) -> Dict:
        """Analyze motion patterns"""
        
        positions = np.array([s.ee_position for s in self.samples])
        
        # Compute motion smoothness (consecutive position differences)
        motion_distances = []
        for i in range(len(positions) - 1):
            motion_distances.append(np.linalg.norm(positions[i+1] - positions[i]))
        
        return {
            "avg_motion_distance": np.mean(motion_distances) if motion_distances else 0,
            "max_motion_distance": np.max(motion_distances) if motion_distances else 0,
            "motion_smoothness": np.std(motion_distances) if motion_distances else 0
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive quality report"""
        
        quality_results = self.check_data_quality()
        
        report = f"""
DATA QUALITY REPORT
==================

Dataset Overview:
- Number of samples: {quality_results['num_samples']}
- Workspace volume: {quality_results['workspace_coverage']['volume']:.4f} m³
- Average sample distance: {quality_results['workspace_coverage']['avg_pairwise_distance']:.3f} m

Detection Quality:
- Hand camera success rate: {quality_results['detection_quality']['hand_camera_success_rate']:.2%}
- External camera success rate: {quality_results['detection_quality']['external_camera_success_rate']:.2%}
- Average corners detected: {quality_results['detection_quality']['avg_corners']:.1f}

Orientation Diversity:
- Mean angular distance: {np.degrees(quality_results['orientation_diversity']['mean_angular_distance']):.1f}°
- Max angular distance: {np.degrees(quality_results['orientation_diversity']['max_angular_distance']):.1f}°

Motion Analysis:
- Average motion distance: {quality_results['motion_analysis']['avg_motion_distance']:.3f} m
- Motion smoothness (std): {quality_results['motion_analysis']['motion_smoothness']:.3f}

Recommendations:
"""
        
        for rec in quality_results['recommendations']:
            report += f"- {rec}\n"
        
        if not quality_results['recommendations']:
            report += "- Data quality looks good for calibration!\n"
        
        return report

# Usage example and testing utilities
def run_validation_pipeline(data_dir: str = "calibration_data"):
    """Complete validation pipeline"""
    
    # Load results
    if not (os.path.exists("T_hand_eye.npy") and os.path.exists("T_base_external.npy")):
        print("Error: Calibration results not found. Run calibration first.")
        return
    
    T_hand_eye = np.load("T_hand_eye.npy")
    T_base_external = np.load("T_base_external.npy")
    
    # Load samples (simplified - you'd need to reload them)
    print("Note: This example assumes you have samples loaded.")
    print("In practice, you'd reload the samples using the DataLoader class.")
    
    # Create validator
    validator = CalibrationValidator(T_hand_eye, T_base_external)
    
    # Example of testing with new position
    print("\nTo test calibration with new position:")
    print("test_result = validator.test_new_position(ee_pos, ee_quat, hand_img, ext_img, ...)")
    print("This will give you real-time validation of calibration accuracy.")
    
    return validator

# Save/load calibration results with metadata
def save_calibration_results(result: Dict, filename: str = "calibration_results.json"):
    """Save calibration results with metadata"""
    
    # Convert numpy arrays to lists for JSON serialization
    result_json = {
        "T_hand_eye": result["T_hand_eye"].tolist(),
        "T_base_external": result["T_base_external"].tolist(),
        "final_error": float(result["final_error"]),
        "success": bool(result["success"]),
        "timestamp": str(np.datetime64('now')),
        "optimization_info": {
            "iterations": int(result["optimization_result"].nit) if result["success"] else 0,
            "function_evaluations": int(result["optimization_result"].nfev) if result["success"] else 0
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(result_json, f, indent=2)
    
    print(f"Calibration results saved to {filename}")

def load_calibration_results(filename: str = "calibration_results.json") -> Dict:
    """Load calibration results from JSON file"""
    
    with open(filename, 'r') as f:
        result_json = json.load(f)
    
    result = {
        "T_hand_eye": np.array(result_json["T_hand_eye"]),
        "T_base_external": np.array(result_json["T_base_external"]),
        "final_error": result_json["final_error"],
        "success": result_json["success"],
        "timestamp": result_json["timestamp"]
    }
    
    return result